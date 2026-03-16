#!/usr/bin/env python3

"""
nanoJEPA: Clojure execution prediction in latent space.
Given embedding of a Clojure expression, predict embedding of its result.
Apple Silicon MLX implementation.
Usage: uv run train_jepa.py
"""

import gc
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from prepare_jepa import (
    MAX_EXPR_LEN,
    MAX_RESULT_LEN,
    TIME_BUDGET,
    ClojureVocab,
    load_val_pairs,
    make_jepa_dataloader,
)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Hyperparameters (editable by autoresearch loop)
# ---------------------------------------------------------------------------

N_EMBD = 256
DEPTH = 4
N_HEAD = 8            # Doubled from 4
EMA_TAU = 0.996
VICREG_LAMBDA = 1.0
VICREG_GAMMA = 1.0
VICREG_COV = 0.1        # Covariance regularization weight (decorrelates z_ctx dims)
DEVICE_BATCH_SIZE = 96
MATRIX_LR = 0.001
EMBEDDING_LR = 0.01
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05   # 5% warmup
WARMDOWN_RATIO = 0.4  # Cosine decay over final 40%

# ---------------------------------------------------------------------------
# Utilities (copied from train.py)
# ---------------------------------------------------------------------------


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


# ---------------------------------------------------------------------------
# Model components (bidirectional — mask=None)
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
        # mask=None → full bidirectional attention
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
        x = mx.maximum(x, 0) ** 2
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


class ClojureEncoder(nn.Module):
    """Bidirectional transformer encoder with mean-pooling over non-pad tokens."""

    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = [EncoderBlock(n_embd, n_head) for _ in range(n_layer)]

    def __call__(self, tokens, mask):
        # tokens: (B, T), mask: (B, T) float 0/1
        x = self.wte(tokens)
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        x = norm(x)
        # Mean pool over non-pad positions
        mask_f = mask.astype(mx.float32)[..., None]  # (B, T, 1)
        denom = mx.maximum(mx.sum(mask_f, axis=1), 1.0)
        z = mx.sum(x * mask_f, axis=1) / denom  # (B, n_embd)
        return z


class NanoJEPA(nn.Module):
    """Context encoder + predictor MLP."""

    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.ctx_encoder = ClojureEncoder(vocab_size, n_embd, n_layer, n_head)
        # Predictor: 3-layer MLP (256 → 1024 → 1024 → 256)
        self.pred_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.pred_fc2 = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
        self.pred_fc3 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def predict(self, z):
        h = norm(mx.maximum(self.pred_fc1(z), 0))
        h = norm(mx.maximum(self.pred_fc2(h), 0))
        return self.pred_fc3(h)

    def __call__(self, expr_tokens, expr_mask):
        z_ctx = self.ctx_encoder(expr_tokens, expr_mask)
        z_pred = self.predict(z_ctx)
        return z_pred


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------


def ema_update(target_enc, ctx_enc, tau):
    """Update target_enc params via EMA from ctx_enc. No gradients flow."""
    target_params = dict(tree_flatten(target_enc.parameters()))
    source_params = dict(tree_flatten(ctx_enc.parameters()))
    for path, t_param in target_params.items():
        if path in source_params:
            s_param = source_params[path]
            new_param = tau * t_param.astype(mx.float32) + (1.0 - tau) * s_param.astype(mx.float32)
            # Set the value back
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
# Loss
# ---------------------------------------------------------------------------


def compute_loss(model, target_enc, expr_tokens, expr_mask, res_tokens, res_mask):
    z_ctx = model.ctx_encoder(expr_tokens, expr_mask)
    z_pred = model.predict(z_ctx)
    z_tgt = mx.stop_gradient(target_enc(res_tokens, res_mask))

    # L2-normalize for cosine loss
    z_pred_n = z_pred * mx.rsqrt(mx.sum(z_pred * z_pred, axis=-1, keepdims=True) + 1e-8)
    z_tgt_n = z_tgt * mx.rsqrt(mx.sum(z_tgt * z_tgt, axis=-1, keepdims=True) + 1e-8)

    # Cosine loss
    cos_sim = mx.sum(z_pred_n * z_tgt_n, axis=-1)  # (B,)
    main_loss = 1.0 - mx.mean(cos_sim)

    # VICReg variance on z_pred (penalise collapsed predictor dimensions)
    z_pred_centered = z_pred - mx.mean(z_pred, axis=0, keepdims=True)
    std_pred = mx.sqrt(mx.mean(z_pred_centered * z_pred_centered, axis=0) + 1e-4)
    var_pred_loss = mx.mean(mx.maximum(VICREG_GAMMA - std_pred, 0.0))

    # VICReg variance on z_ctx (prevents ctx_encoder collapse; target_enc inherits via EMA)
    z_ctx_centered = z_ctx - mx.mean(z_ctx, axis=0, keepdims=True)
    std_ctx = mx.sqrt(mx.mean(z_ctx_centered * z_ctx_centered, axis=0) + 1e-4)
    var_ctx_loss = mx.mean(mx.maximum(VICREG_GAMMA - std_ctx, 0.0))

    # Covariance regularization on z_ctx: decorrelate encoder dimensions
    N = z_ctx_centered.shape[0]
    cov_ctx = mx.matmul(mx.transpose(z_ctx_centered), z_ctx_centered) / max(N - 1, 1)
    diag_mask = mx.eye(N_EMBD)
    off_diag_sq = mx.square(cov_ctx * (1.0 - diag_mask))
    cov_loss = mx.sum(off_diag_sq) / N_EMBD

    return (
        main_loss
        + VICREG_LAMBDA * (var_pred_loss + var_ctx_loss)
        + VICREG_COV * cov_loss
    )


# ---------------------------------------------------------------------------
# Optimizer (simplified AdamW — adapted from train.py)
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


def evaluate_val_retrieval(model, target_enc, batch_size=256):
    """Recall@1 on 2000 val pairs: does predict(encode(expr)) retrieve the correct result?

    1.0 = perfect retrieval, ~0.0005 = random (1/2000). Unlike cosine similarity,
    this metric cannot be gamed by representation collapse — collapsed embeddings
    give near-zero recall because all predictions tie and argmax picks arbitrarily.
    """
    expr, expr_mask, res, res_mask = load_val_pairs()
    N = len(expr)
    all_z_pred, all_z_tgt = [], []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = mx.array(expr[start:end], dtype=mx.int32)
        em = mx.array(expr_mask[start:end], dtype=mx.float32)
        r = mx.array(res[start:end], dtype=mx.int32)
        rm = mx.array(res_mask[start:end], dtype=mx.float32)

        z_pred = model(e, em)
        z_tgt = mx.stop_gradient(target_enc(r, rm))
        mx.eval(z_pred, z_tgt)
        all_z_pred.append(z_pred)
        all_z_tgt.append(z_tgt)

    z_pred_all = mx.concatenate(all_z_pred, axis=0)   # (N, D)
    z_tgt_all = mx.concatenate(all_z_tgt, axis=0)     # (N, D)

    # L2-normalize
    z_pred_n = z_pred_all * mx.rsqrt(mx.sum(z_pred_all * z_pred_all, axis=-1, keepdims=True) + 1e-8)
    z_tgt_n = z_tgt_all * mx.rsqrt(mx.sum(z_tgt_all * z_tgt_all, axis=-1, keepdims=True) + 1e-8)

    # (N, N) similarity matrix → Recall@1
    sim_matrix = mx.matmul(z_pred_n, mx.transpose(z_tgt_n))
    labels = mx.arange(N, dtype=mx.int32)
    predicted = mx.argmax(sim_matrix, axis=1).astype(mx.int32)
    recall_at_1 = mx.mean((predicted == labels).astype(mx.float32))
    mx.eval(recall_at_1)
    return float(recall_at_1.item())


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

vocab = ClojureVocab()
from prepare_jepa import VOCAB_SIZE

train_loader = make_jepa_dataloader(vocab, DEVICE_BATCH_SIZE)

t_data = time.time()
print(f"Data ready in {t_data - t_start:.1f}s")

model = NanoJEPA(VOCAB_SIZE, N_EMBD, DEPTH, N_HEAD)
# Target encoder: intentionally initialized independently (different random state).
# This acts as a warm-up curriculum: the model first learns to map encoder-A → encoder-B
# (structured warm-up), then transitions to the real semantic task as EMA converges.
# Empirically better than copy-init for short (~6000 step) training budgets.
target_enc = ClojureEncoder(VOCAB_SIZE, N_EMBD, DEPTH, N_HEAD)

mx.eval(model.parameters(), target_enc.parameters())

num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
num_target_params = sum(p.size for _, p in tree_flatten(target_enc.parameters()))
print(f"Model params (trainable): {num_params / 1e6:.2f}M")
print(f"Target encoder params (EMA only): {num_target_params / 1e6:.2f}M")

optimizer = AdamW(model, matrix_lr=MATRIX_LR, embedding_lr=EMBEDDING_LR, weight_decay=WEIGHT_DECAY)

loss_grad_fn = nn.value_and_grad(model, lambda m, et, em, rt, rm: compute_loss(m, target_enc, et, em, rt, rm))

print(f"Time budget: {TIME_BUDGET}s")

total_training_time = 0.0
step = 0
t_compiled = None
smooth_loss = 0.0
total_pairs = 0

while True:
    t0 = time.time()

    expr_np, expr_mask_np, res_np, res_mask_np = next(train_loader)
    expr_t = mx.array(expr_np, dtype=mx.int32)
    expr_m = mx.array(expr_mask_np, dtype=mx.float32)
    res_t = mx.array(res_np, dtype=mx.int32)
    res_m = mx.array(res_mask_np, dtype=mx.float32)

    loss, grads = loss_grad_fn(model, expr_t, expr_m, res_t, res_m)
    mx.eval(loss, grads)

    if t_compiled is None:
        t_compiled = time.time()
        print(f"Compiled in {t_compiled - t_data:.1f}s")

    # LR schedule: Linear warmup + Cosine decay
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    if progress < WARMUP_RATIO:
        lrm = progress / WARMUP_RATIO
    elif progress < (1.0 - WARMDOWN_RATIO):
        lrm = 1.0
    else:
        # Cosine decay from 1.0 to 0.0
        decay_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
        lrm = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    if step == 0:
        for cfg in optimizer.param_config.values():
            cfg["_base_lr"] = cfg["lr"]
    for cfg in optimizer.param_config.values():
        cfg["lr"] = cfg["_base_lr"] * lrm

    optimizer.update(model, grads)
    mx.eval(model.parameters(), *optimizer.state)

    # EMA update of target encoder
    ema_update(target_enc, model.ctx_encoder, EMA_TAU)
    mx.eval(target_enc.parameters())

    loss_f = float(loss.item())
    if loss_f > 100 or math.isnan(loss_f):
        print("FAIL")
        raise SystemExit(1)

    dt = time.time() - t0
    if step >= 1:
        total_training_time += dt

    total_pairs += DEVICE_BATCH_SIZE
    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | "
        f"dt: {dt*1000:.0f}ms | pairs: {total_pairs/1e6:.2f}M | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

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
print(f"Training completed in {t_train - (t_compiled or t_start):.1f}s")

print("Starting final eval...")
val_recall_at_1 = evaluate_val_retrieval(model, target_enc)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_recall_at_1_pct: {val_recall_at_1 * 100:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_pairs_M:      {total_pairs / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
