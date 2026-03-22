#!/usr/bin/env python3
"""
Text JEPA: BPE masked span prediction in latent space.
Adapted from train_jepa.py (Clojure JEPA) for natural language text.
Trains per-language models for cross-lingual alignment experiment.

Usage:
  python3 train_text_jepa.py --lang en   # train on English Wikipedia
  python3 train_text_jepa.py --lang fi   # train on Finnish Wikipedia
"""

import gc
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from prepare_text_jepa import (
    BOS_ID,
    MAX_CTX_LEN,
    MAX_TGT_LEN,
    BPETokenizer,
    load_sentences,
    load_val_pairs,
    make_text_dataloader,
)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

N_EMBD = 128
DEPTH = 4
N_HEAD = 4
EMA_TAU_START = 0.996
EMA_TAU_END = 0.9999
VICREG_LAMBDA = 1.0
VICREG_GAMMA = 0.5
VICREG_COV = 0.15
INFONCE_WEIGHT = 0.5
INFONCE_TEMP = 0.07
QUEUE_SIZE = 2048
DEVICE_BATCH_SIZE = 32
MATRIX_LR = 0.0012
EMBEDDING_LR = 0.012
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.4
TIME_BUDGET = 300

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "text")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def l2_normalize(x):
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-8)


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


# ---------------------------------------------------------------------------
# Model components (reused from train_jepa.py)
# ---------------------------------------------------------------------------


class BidirAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        assert n_embd % n_head == 0
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        q = norm(self.rope(q))
        k = norm(self.rope(k))
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class EncoderMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReLU^2
        return self.c_proj(x)


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = BidirAttention(n_embd, n_head)
        self.mlp = EncoderMLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, q_x, kv_x):
        B, T, _ = kv_x.shape
        q = self.c_q(q_x).reshape(B, 1, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(kv_x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(kv_x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        y = y.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return self.c_proj(y)


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = CrossAttention(n_embd, n_head)
        self.mlp = EncoderMLP(n_embd)

    def __call__(self, q, kv):
        q = q + self.attn(norm(q), norm(kv))
        q = q + self.mlp(norm(q))
        return q


class AttentionPool(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.score = nn.Linear(n_embd, 1, bias=False)

    def __call__(self, x, mask):
        scores = self.score(x).squeeze(-1)
        scores = scores + (1.0 - mask) * -1e9
        weights = mx.softmax(scores, axis=-1)
        return mx.sum(x * weights[..., None], axis=1)


# ---------------------------------------------------------------------------
# Text Encoder (simplified from ClojureEncoder — no depth/numeric embeddings)
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = [EncoderBlock(n_embd, n_head) for _ in range(n_layer)]
        self.pool = AttentionPool(n_embd)

    def _run_transformer(self, tokens):
        x = self.wte(tokens)
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        return norm(x)

    def __call__(self, tokens, mask):
        x = self._run_transformer(tokens)
        return self.pool(x, mask.astype(mx.float32))

    def encode_full(self, tokens, mask):
        x = self._run_transformer(tokens)
        z = self.pool(x, mask.astype(mx.float32))
        return z, x


# ---------------------------------------------------------------------------
# Text JEPA model
# ---------------------------------------------------------------------------


class TextJEPA(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.ctx_encoder = TextEncoder(vocab_size, n_embd, n_layer, n_head)
        self.res_query = mx.random.normal((1, 1, n_embd)) * 0.02
        self.pred_blocks = [CrossAttentionBlock(n_embd, n_head) for _ in range(3)]

    def predict(self, hidden_ctx):
        B, T, D = hidden_ctx.shape
        q = mx.broadcast_to(self.res_query, (B, 1, D))
        for block in self.pred_blocks:
            q = block(q, hidden_ctx)
        return norm(q.squeeze(1))

    def __call__(self, ctx_tokens, ctx_mask):
        z_ctx, hidden_ctx = self.ctx_encoder.encode_full(ctx_tokens, ctx_mask)
        z_pred = self.predict(hidden_ctx)
        return z_pred


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------


def ema_update(target_enc, ctx_enc, tau):
    target_params = dict(tree_flatten(target_enc.parameters()))
    source_params = dict(tree_flatten(ctx_enc.parameters()))
    for path, t_param in target_params.items():
        if path in source_params:
            s_param = source_params[path]
            new_param = tau * t_param.astype(mx.float32) + (1.0 - tau) * s_param.astype(mx.float32)
            _set_nested(target_enc, path, new_param.astype(t_param.dtype))


def _set_nested(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def compute_loss(model, target_enc, ctx_tokens, ctx_mask, tgt_tokens, tgt_mask, queue_keys):
    z_ctx, hidden_ctx = model.ctx_encoder.encode_full(ctx_tokens, ctx_mask)
    z_pred = model.predict(hidden_ctx)
    z_tgt = mx.stop_gradient(target_enc(tgt_tokens, tgt_mask))

    z_pred_n = l2_normalize(z_pred)
    z_tgt_n = l2_normalize(z_tgt)

    # Cosine loss
    cos_sim = mx.sum(z_pred_n * z_tgt_n, axis=-1)
    main_loss = 1.0 - mx.mean(cos_sim)

    # VICReg variance on z_pred
    z_pred_centered = z_pred - mx.mean(z_pred, axis=0, keepdims=True)
    std_pred = mx.sqrt(mx.mean(z_pred_centered * z_pred_centered, axis=0) + 1e-4)
    var_pred_loss = mx.mean(mx.maximum(VICREG_GAMMA - std_pred, 0.0))

    # VICReg variance on z_ctx
    z_ctx_centered = z_ctx - mx.mean(z_ctx, axis=0, keepdims=True)
    std_ctx = mx.sqrt(mx.mean(z_ctx_centered * z_ctx_centered, axis=0) + 1e-4)
    var_ctx_loss = mx.mean(mx.maximum(VICREG_GAMMA - std_ctx, 0.0))

    # Covariance regularization on z_ctx
    N = z_ctx_centered.shape[0]
    cov_ctx = mx.matmul(mx.transpose(z_ctx_centered), z_ctx_centered) / max(N - 1, 1)
    diag_mask = mx.eye(N_EMBD)
    off_diag_sq = mx.square(cov_ctx * (1.0 - diag_mask))
    cov_loss = mx.sum(off_diag_sq) / N_EMBD

    # MoCo InfoNCE
    queue_sg = mx.stop_gradient(queue_keys)
    all_keys = mx.concatenate([queue_sg, z_tgt_n], axis=0)
    B_size = ctx_tokens.shape[0]
    logits = mx.matmul(z_pred_n, mx.transpose(all_keys)) / INFONCE_TEMP
    labels = mx.arange(QUEUE_SIZE, QUEUE_SIZE + B_size, dtype=mx.int32)
    infonce_loss = mx.mean(nn.losses.cross_entropy(logits, labels))

    return (
        main_loss
        + VICREG_LAMBDA * (var_pred_loss + var_ctx_loss)
        + VICREG_COV * cov_loss
        + INFONCE_WEIGHT * infonce_loss
    )


# ---------------------------------------------------------------------------
# Optimizer (from train_jepa.py)
# ---------------------------------------------------------------------------


class AdamW:
    def __init__(self, model, matrix_lr, embedding_lr, weight_decay):
        self.param_config = {}
        self.adam_state = {}
        adam_betas = (0.9, 0.999)

        for path, param in tree_flatten(model.parameters()):
            if "wte" in path:
                lr = embedding_lr
                wd = 0.0
            elif param.ndim == 2:
                lr = matrix_lr
                wd = weight_decay
            else:
                lr = embedding_lr
                wd = 0.0
            self.param_config[path] = {"lr": lr, "betas": adam_betas, "eps": 1e-8, "weight_decay": wd}

        self.initial_lrs = {p: c["lr"] for p, c in self.param_config.items()}

    def _step(self, path, grad, param, config):
        grad_f32 = grad.astype(mx.float32)
        param_f32 = param.astype(mx.float32)
        lr = config["lr"]
        beta1, beta2 = config["betas"]
        eps = config["eps"]
        wd = config["weight_decay"]

        if path not in self.adam_state:
            self.adam_state[path] = {"m": mx.zeros_like(grad_f32), "v": mx.zeros_like(grad_f32), "t": 0}

        state = self.adam_state[path]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad_f32
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad_f32 * grad_f32)
        bias1 = 1 - beta1 ** state["t"]
        bias2 = 1 - beta2 ** state["t"]
        denom = mx.sqrt(state["v"] / bias2) + eps
        step_size = lr / bias1
        param_f32 = param_f32 * (1 - lr * wd)
        param_f32 = param_f32 - step_size * (state["m"] / denom)
        return param_f32.astype(param.dtype)

    def _set_path_value(self, model, path, value):
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            setattr(obj, last, value)

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params = dict(tree_flatten(model.parameters()))
        for path, grad in flat_grads.items():
            if path not in self.param_config:
                continue
            config = self.param_config[path]
            param = flat_params[path]
            new_param = self._step(path, grad, param, config)
            self._set_path_value(model, path, new_param)

    @property
    def state(self):
        arrays = []
        for state in self.adam_state.values():
            arrays.extend([state["m"], state["v"]])
        return arrays


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_val(model, target_enc, lang, batch_size=256):
    """Evaluate on validation pairs. Returns (mean_cos, recall_at_1)."""
    ctx, ctx_mask, tgt, tgt_mask = load_val_pairs(lang)
    N = len(ctx)
    cos_sims = []
    all_z_pred = []
    all_z_tgt = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        c = mx.array(ctx[start:end], dtype=mx.int32)
        cm = mx.array(ctx_mask[start:end], dtype=mx.float32)
        t = mx.array(tgt[start:end], dtype=mx.int32)
        tm = mx.array(tgt_mask[start:end], dtype=mx.float32)

        z_pred = model(c, cm)
        z_tgt = mx.stop_gradient(target_enc(t, tm))

        z_pred_n = l2_normalize(z_pred)
        z_tgt_n = l2_normalize(z_tgt)

        cos_sim = mx.sum(z_pred_n * z_tgt_n, axis=-1)
        mx.eval(cos_sim, z_pred_n, z_tgt_n)
        cos_sims.append(cos_sim)
        all_z_pred.append(np.array(z_pred_n))
        all_z_tgt.append(np.array(z_tgt_n))

    cos_all = mx.concatenate(cos_sims, axis=0)
    mx.eval(cos_all)
    mean_cos = float(np.mean(np.array(cos_all)))

    # Recall@1
    Z_pred = np.concatenate(all_z_pred, axis=0)
    Z_tgt = np.concatenate(all_z_tgt, axis=0)
    sims = Z_pred @ Z_tgt.T
    retrieved = np.argmax(sims, axis=1)
    true_emb = Z_tgt[np.arange(N)]
    retrieved_emb = Z_tgt[retrieved]
    cos_match = np.sum(true_emb * retrieved_emb, axis=1)
    recall_at_1 = float(np.mean(cos_match > 0.9999))

    return mean_cos, recall_at_1


# ---------------------------------------------------------------------------
# Save/load model
# ---------------------------------------------------------------------------


def save_model(model, target_enc, lang):
    """Save model and target encoder weights."""
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = os.path.join(DATA_DIR, f"model_{lang}.npz")

    # Flatten and save all parameters
    model_params = {f"model.{k}": v for k, v in tree_flatten(model.ctx_encoder.parameters())}
    target_params = {f"target.{k}": v for k, v in tree_flatten(target_enc.parameters())}
    all_params = {**model_params, **target_params}

    # Convert to numpy for saving
    np_params = {}
    for k, v in all_params.items():
        mx.eval(v)
        np_params[k] = np.array(v)
    np.savez_compressed(model_path, **np_params)
    print(f"Model saved to {model_path}")


def load_encoder(lang, vocab_size=None):
    """Load a trained TextEncoder for embedding extraction."""
    if vocab_size is None:
        tokenizer = BPETokenizer.load()
        vocab_size = tokenizer.vocab_size
    model_path = os.path.join(DATA_DIR, f"model_{lang}.npz")
    data = np.load(model_path)

    encoder = TextEncoder(vocab_size, N_EMBD, DEPTH, N_HEAD)
    mx.eval(encoder.parameters())

    # Load model parameters
    for key in data.files:
        if key.startswith("model."):
            param_path = key[len("model."):]
            _set_nested(encoder, param_path, mx.array(data[key]))

    mx.eval(encoder.parameters())
    return encoder


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    # Parse language argument
    lang = "en"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--lang" and i < len(sys.argv) - 1:
            lang = sys.argv[i + 1]

    print(f"Training Text JEPA for language: {lang}")
    print(f"Time budget: {TIME_BUDGET}s")

    t_start = time.time()
    mx.random.seed(42)

    # Load BPE tokenizer and data
    tokenizer = BPETokenizer.load()
    VOCAB_SIZE = tokenizer.vocab_size
    sentences = load_sentences(lang)
    train_loader = make_text_dataloader(sentences, tokenizer, DEVICE_BATCH_SIZE)

    t_data = time.time()
    print(f"Data ready in {t_data - t_start:.1f}s ({len(sentences)} sentences, vocab={VOCAB_SIZE})")

    # Create model
    model = TextJEPA(VOCAB_SIZE, N_EMBD, DEPTH, N_HEAD)
    target_enc = TextEncoder(VOCAB_SIZE, N_EMBD, DEPTH, N_HEAD)
    mx.eval(model.parameters(), target_enc.parameters())

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    num_target_params = sum(p.size for _, p in tree_flatten(target_enc.parameters()))
    print(f"Model params (trainable): {num_params / 1e6:.2f}M")
    print(f"Target encoder params (EMA only): {num_target_params / 1e6:.2f}M")

    # Optimizer
    optimizer = AdamW(model, matrix_lr=MATRIX_LR, embedding_lr=EMBEDDING_LR, weight_decay=WEIGHT_DECAY)

    loss_grad_fn = nn.value_and_grad(
        model,
        lambda m, ct, cm, tt, tm, qk: compute_loss(m, target_enc, ct, cm, tt, tm, qk),
    )

    # MoCo queue
    _queue_np = np.random.randn(QUEUE_SIZE, N_EMBD).astype(np.float32)
    _queue_np /= np.linalg.norm(_queue_np, axis=-1, keepdims=True) + 1e-8
    _queue_ptr = 0

    # Training
    total_training_time = 0.0
    step = 0
    t_compiled = None
    smooth_loss = 0.0
    total_samples = 0
    next_log_pct = 0

    while True:
        t0 = time.time()

        ctx_np, ctx_mask_np, tgt_np, tgt_mask_np = next(train_loader)

        ctx_t = mx.array(ctx_np, dtype=mx.int32)
        ctx_m = mx.array(ctx_mask_np, dtype=mx.float32)
        tgt_t = mx.array(tgt_np, dtype=mx.int32)
        tgt_m = mx.array(tgt_mask_np, dtype=mx.float32)

        queue_snapshot = mx.array(_queue_np, dtype=mx.float32)
        loss, grads = loss_grad_fn(model, ctx_t, ctx_m, tgt_t, tgt_m, queue_snapshot)
        mx.eval(loss, grads)

        if t_compiled is None:
            t_compiled = time.time()
            print(f"Compiled in {t_compiled - t_data:.1f}s")

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        if progress < WARMUP_RATIO:
            lrm = progress / WARMUP_RATIO
        elif progress < (1.0 - WARMDOWN_RATIO):
            lrm = 1.0
        else:
            decay_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
            lrm = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

        if step == 0:
            for cfg in optimizer.param_config.values():
                cfg["_base_lr"] = cfg["lr"]
        for cfg in optimizer.param_config.values():
            cfg["lr"] = cfg["_base_lr"] * lrm

        optimizer.update(model, grads)
        mx.eval(model.parameters(), *optimizer.state)

        # EMA update
        tau = EMA_TAU_END - (EMA_TAU_END - EMA_TAU_START) * (math.cos(math.pi * progress) + 1) / 2
        ema_update(target_enc, model.ctx_encoder, tau)
        mx.eval(target_enc.parameters())

        # Update MoCo queue
        z_new = mx.stop_gradient(target_enc(tgt_t, tgt_m))
        z_new_n = l2_normalize(z_new)
        mx.eval(z_new_n)
        new_keys = np.array(z_new_n)
        B_q = new_keys.shape[0]
        end_ptr = _queue_ptr + B_q
        if end_ptr <= QUEUE_SIZE:
            _queue_np[_queue_ptr:end_ptr] = new_keys
        else:
            first = QUEUE_SIZE - _queue_ptr
            _queue_np[_queue_ptr:] = new_keys[:first]
            _queue_np[:end_ptr % QUEUE_SIZE] = new_keys[first:]
        _queue_ptr = end_ptr % QUEUE_SIZE

        loss_f = float(loss.item())
        if loss_f > 100 or math.isnan(loss_f):
            print("FAIL: loss exploded")
            raise SystemExit(1)

        dt = time.time() - t0
        if step >= 1:
            total_training_time += dt

        total_samples += DEVICE_BATCH_SIZE
        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased = smooth_loss / (1 - ema_beta ** (step + 1))
        pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)

        # Log at step 0 and every 20%
        if step == 0 or pct >= next_log_pct:
            print(f"step {step:05d} ({pct:.0f}%) | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms")
            next_log_pct = (int(pct / 20) + 1) * 20

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 1000 == 0:
            gc.collect()

        step += 1
        if step >= 2 and total_training_time >= TIME_BUDGET:
            break

    print()
    t_train = time.time()
    print(f"Training completed in {t_train - (t_compiled or t_start):.1f}s ({step} steps)")

    # Save model
    save_model(model, target_enc, lang)

    # Final evaluation
    print("Starting final eval...")
    val_mean_cos, val_recall = evaluate_val(model, target_enc, lang)
    t_eval = time.time()
    print(f"Final eval completed in {t_eval - t_train:.1f}s")

    peak_vram_mb = get_peak_memory_mb()

    print("---")
    print(f"lang:             {lang}")
    print(f"val_mean_cos:     {val_mean_cos:.4f}")
    print(f"val_recall_at_1:  {val_recall * 100:.2f}%")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_eval - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_samples_M:    {total_samples / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")


if __name__ == "__main__":
    main()
