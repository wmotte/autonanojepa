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
    VOCAB_SIZE,
    BOS_ID,
    ClojureVocab,
    load_val_pairs,
    make_jepa_dataloader,
)

TIME_BUDGET = 60  # reduced from 120s — model already saturates at 120s

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# Hyperparameters (editable by autoresearch loop)
# ---------------------------------------------------------------------------

N_EMBD = 160
DEPTH = 4
N_HEAD = 4            # 160 // 4 = 40 (even for RoPE)
EMA_TAU = 0.9995  # Slower EMA target update
VICREG_LAMBDA = 1.0
VICREG_GAMMA = 0.2  # Reduced from 0.5 (from original best config)
VICREG_COV = 0.05   # Reduced from 0.1 (from original best config)
SUBEXPR_WEIGHT = 0.25   # Sub-expression auxiliary loss weight (Gap 2)
DEVICE_BATCH_SIZE = 28  # Best batch size for time budget
MATRIX_LR = 0.0005
EMBEDDING_LR = 0.005
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
# Depth embedding (Gap 1: structural bias)
# ---------------------------------------------------------------------------

MAX_DEPTH = 8  # Maximum nesting depth tracked (vocab rarely exceeds 5)


class DepthEmbedding(nn.Module):
    """Encodes syntactic nesting depth for each token position.

    For `( + ( count [ 1 2 ] ) 2 )`:
      depth:  0  1   1  2    2 2 2  1  1  1  0
    Forces attention heads to distinguish tokens at different scoping levels.
    Token IDs: ( = 4, ) = 5, [ = 6, ] = 7, { = 8, } = 9  (from ClojureVocab)
    """

    def __init__(self, max_depth, n_embd):
        super().__init__()
        self.emb = nn.Embedding(max_depth + 1, n_embd)
        self.max_depth = max_depth

    def __call__(self, tokens):
        # tokens: (B, T) int32
        is_open  = ((tokens == 4) | (tokens == 6) | (tokens == 8)).astype(mx.int32)
        is_close = ((tokens == 5) | (tokens == 7) | (tokens == 9)).astype(mx.int32)
        delta = is_open - is_close
        cum   = mx.cumsum(delta, axis=-1)
        # depth[t] = sum of deltas *before* position t = cum[t] - delta[t]
        depth = mx.maximum(cum - delta, 0)
        depth = mx.minimum(depth, self.max_depth)
        return self.emb(depth)


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
        self.depth_emb = DepthEmbedding(MAX_DEPTH, n_embd)  # Gap 1
        self.blocks = [EncoderBlock(n_embd, n_head) for _ in range(n_layer)]

    def _run_transformer(self, tokens):
        """Full transformer pass; returns per-token hidden states (B, T, D)."""
        x = self.wte(tokens) + self.depth_emb(tokens)  # Gap 1: depth signal
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        return norm(x)

    def __call__(self, tokens, mask):
        # tokens: (B, T), mask: (B, T) float 0/1
        x = self._run_transformer(tokens)
        mask_f = mask.astype(mx.float32)[..., None]  # (B, T, 1)
        denom = mx.maximum(mx.sum(mask_f, axis=1), 1.0)
        return mx.sum(x * mask_f, axis=1) / denom  # (B, n_embd)

    def encode_full(self, tokens, mask):
        """Returns (z, hidden): pooled embedding + per-token hidden states (B, T, D).
        Used by sub-expression auxiliary loss (Gap 2).
        """
        x = self._run_transformer(tokens)
        mask_f = mask.astype(mx.float32)[..., None]
        denom = mx.maximum(mx.sum(mask_f, axis=1), 1.0)
        z = mx.sum(x * mask_f, axis=1) / denom
        return z, x


class NanoJEPA(nn.Module):
    """Context encoder + predictor MLP + reverse predictor (symmetric JEPA)."""

    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.ctx_encoder = ClojureEncoder(vocab_size, n_embd, n_layer, n_head)
        # Forward predictor: expr → result
        self.pred_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.pred_fc2 = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
        self.pred_fc3 = nn.Linear(4 * n_embd, n_embd, bias=False)
        # Iterative refinement (Gap 3): shared residual step applied 2x
        self.refine_fc = nn.Linear(n_embd, n_embd, bias=False)
        # Reverse predictor: result → expr (symmetric JEPA)
        self.rev_pred_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.rev_pred_fc2 = nn.Linear(4 * n_embd, 4 * n_embd, bias=False)
        self.rev_pred_fc3 = nn.Linear(4 * n_embd, n_embd, bias=False)
        # Sub-expression predictor (Gap 2): hidden-at-root → span embedding
        self.subexpr_fc1 = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.subexpr_fc2 = nn.Linear(2 * n_embd, n_embd, bias=False)

    def predict(self, z):
        h = norm(mx.maximum(self.pred_fc1(z), 0))
        h = norm(mx.maximum(self.pred_fc2(h), 0))
        z_pred = self.pred_fc3(h)
        # Gap 3: iterative refinement — 2 residual steps with shared weights.
        # Simulates multi-step computation (e.g. reduce accumulation).
        for _ in range(2):
            z_pred = z_pred + self.refine_fc(norm(z_pred))
        return z_pred

    def predict_reverse(self, z):
        """Reverse predictor: predict expression embedding from result embedding."""
        h = norm(mx.maximum(self.rev_pred_fc1(z), 0))
        h = norm(mx.maximum(self.rev_pred_fc2(h), 0))
        return self.rev_pred_fc3(h)

    def predict_subexpr(self, z_span_root):
        """Gap 2: predict sub-expression embedding from its root hidden state."""
        h = norm(mx.maximum(self.subexpr_fc1(z_span_root), 0))
        return self.subexpr_fc2(h)

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
# Sub-expression span extraction (Gap 2: compositional generalization)
# ---------------------------------------------------------------------------

_OPEN_IDS  = {4, 6, 8}  # (, [, {  (token IDs from ClojureVocab)
_CLOSE_IDS = {5, 7, 9}  # ), ], }


def compute_span_info(expr_np, expr_mask_np):
    """CPU-side bracket matching: sample one sub-expression span per batch item.

    Returns:
      span_starts : (B,) int32  — start index of sampled span (-1 if none found)
      span_mask   : (B, T) float32 — 1.0 within sampled span, 0.0 elsewhere

    Only non-root spans (start > 1, skipping BOS + outer paren) of length ≥ 3
    are considered, so simple flat expressions yield no span (start stays -1).
    """
    B, T = expr_np.shape
    span_starts = np.full(B, -1, dtype=np.int32)
    span_mask   = np.zeros((B, T), dtype=np.float32)

    for b in range(B):
        toks  = expr_np[b]
        spans = []
        stack = []
        for t in range(T):
            tok = int(toks[t])
            if tok in _OPEN_IDS:
                stack.append(t)
            elif tok in _CLOSE_IDS and stack:
                start = stack.pop()
                end   = t + 1           # exclusive
                # start > 1: skip outermost ( at position 1 (right after BOS)
                if end - start >= 3 and start > 1:
                    spans.append((start, end))

        if spans:
            i = np.random.randint(len(spans))
            start, end = spans[i]
            span_starts[b] = start
            # Intersect with expression mask so PAD positions stay 0
            span_mask[b, start:end] = expr_mask_np[b, start:end]

    return span_starts, span_mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(model, target_enc, expr_tokens, expr_mask, res_tokens, res_mask,
                 span_starts, span_mask):
    # Gap 2: encode_full returns per-token hidden states needed for subexpr loss
    z_ctx, hidden_ctx = model.ctx_encoder.encode_full(expr_tokens, expr_mask)
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

    # Symmetric JEPA: reverse loss (result → expression)
    z_rev_pred = model.predict_reverse(z_tgt)
    z_ctx_n = z_ctx * mx.rsqrt(mx.sum(z_ctx * z_ctx, axis=-1, keepdims=True) + 1e-8)
    z_rev_pred_n = z_rev_pred * mx.rsqrt(mx.sum(z_rev_pred * z_rev_pred, axis=-1, keepdims=True) + 1e-8)
    rev_cos_sim = mx.sum(z_rev_pred_n * z_ctx_n, axis=-1)
    rev_loss = 1.0 - mx.mean(rev_cos_sim)

    # Gap 2: sub-expression auxiliary loss.
    # Target: target encoder mean-pooled over the sampled span positions.
    # Prediction: from the hidden state at the span's opening bracket.
    # This forces each ( hidden state to encode what its sub-expression means.
    B = expr_tokens.shape[0]
    z_span_tgt = mx.stop_gradient(target_enc(expr_tokens, span_mask))
    clamped_starts = mx.maximum(span_starts, 0)          # clamp -1 → 0 (masked out below)
    z_span_root = hidden_ctx[mx.arange(B), clamped_starts]  # (B, D)
    z_span_pred = model.predict_subexpr(z_span_root)

    z_span_tgt_n  = z_span_tgt  * mx.rsqrt(mx.sum(z_span_tgt  * z_span_tgt,  axis=-1, keepdims=True) + 1e-8)
    z_span_pred_n = z_span_pred * mx.rsqrt(mx.sum(z_span_pred * z_span_pred, axis=-1, keepdims=True) + 1e-8)
    span_cos  = mx.sum(z_span_tgt_n * z_span_pred_n, axis=-1)   # (B,)
    has_span  = (span_starts >= 0).astype(mx.float32)            # (B,)
    n_valid   = mx.maximum(mx.sum(has_span), 1.0)
    subexpr_loss = mx.sum((1.0 - span_cos) * has_span) / n_valid

    return (
        main_loss
        + 0.5 * rev_loss  # Symmetric JEPA weight
        + VICREG_LAMBDA * (var_pred_loss + var_ctx_loss)
        + VICREG_COV * cov_loss
        + SUBEXPR_WEIGHT * subexpr_loss
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
# Evaluation helpers
# ---------------------------------------------------------------------------


def _val_span_info(expr_np, expr_mask_np):
    """Deterministic span extraction for validation: first innermost closed span.

    Unlike compute_span_info (random), this always picks the first span whose
    closing bracket appears earliest — i.e. the shallowest innermost sub-expression.
    Returns span_starts (B,) int32 and span_masks (B, T) float32.
    """
    B, T = expr_np.shape
    span_starts = np.full(B, -1, dtype=np.int32)
    span_masks  = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        toks  = expr_np[b]
        stack = []
        for t in range(T):
            tok = int(toks[t])
            if tok in _OPEN_IDS:
                stack.append(t)
            elif tok in _CLOSE_IDS and stack:
                start = stack.pop()
                end   = t + 1
                if end - start >= 3 and start > 1:
                    span_starts[b] = start
                    span_masks[b, start:end] = expr_mask_np[b, start:end]
                    break  # first valid innermost span only
    return span_starts, span_masks


def _evaluate_subexpr_cos(model, target_enc, expr, expr_mask, batch_size=256):
    """Cosine similarity for sub-expression prediction head (Gap 2).

    For each val item with a valid interior span, tests whether
    predict_subexpr(hidden_at_open_bracket) ≈ target_enc(span_tokens).
    Returns mean cosine over val items that have a sub-expression.
    """
    span_starts_np, span_masks_np = _val_span_info(expr, expr_mask)
    N = len(expr)
    total_cos = 0.0
    total_n   = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e   = mx.array(expr[start:end],         dtype=mx.int32)
        em  = mx.array(expr_mask[start:end],    dtype=mx.float32)
        ss  = mx.array(span_starts_np[start:end], dtype=mx.int32)
        smk = mx.array(span_masks_np[start:end],  dtype=mx.float32)

        _, hidden   = model.ctx_encoder.encode_full(e, em)
        z_span_tgt  = mx.stop_gradient(target_enc(e, smk))
        clamped     = mx.maximum(ss, 0)
        B           = e.shape[0]
        z_span_root = hidden[mx.arange(B), clamped]         # (B, D)
        z_span_pred = model.predict_subexpr(z_span_root)

        z_tgt_n  = z_span_tgt  * mx.rsqrt(mx.sum(z_span_tgt  * z_span_tgt,  axis=-1, keepdims=True) + 1e-8)
        z_pred_n = z_span_pred * mx.rsqrt(mx.sum(z_span_pred * z_span_pred, axis=-1, keepdims=True) + 1e-8)
        cos = mx.sum(z_tgt_n * z_pred_n, axis=-1)  # (B,)
        mx.eval(cos)

        cos_np_b = np.array(cos)
        has      = span_starts_np[start:end] >= 0
        total_cos += float(cos_np_b[has].sum()) if has.any() else 0.0
        total_n   += int(has.sum())

    return total_cos / total_n if total_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_val_retrieval(model, target_enc, batch_size=256):
    """Evaluate on 2000 masked-JEPA val pairs.

    Returns (val_mean_cos, val_class_acc, cos_by_masklen, val_subexpr_cos):
      - val_mean_cos:    mean cosine similarity (all pairs)
      - val_class_acc:   nearest-neighbour token accuracy on mask_len==1 pairs
      - cos_by_masklen:  {1: float, 2: float, 3: float} cosine by mask length
      - val_subexpr_cos: cosine for sub-expression prediction head (Gap 2)
    """
    expr, expr_mask, res, res_mask, mask_len = load_val_pairs()
    N = len(expr)
    cos_sims = []
    all_z_pred = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        e = mx.array(expr[start:end], dtype=mx.int32)
        em = mx.array(expr_mask[start:end], dtype=mx.float32)
        r = mx.array(res[start:end], dtype=mx.int32)
        rm = mx.array(res_mask[start:end], dtype=mx.float32)

        z_pred = model(e, em)
        z_tgt = mx.stop_gradient(target_enc(r, rm))

        z_pred_n = z_pred * mx.rsqrt(mx.sum(z_pred * z_pred, axis=-1, keepdims=True) + 1e-8)
        z_tgt_n = z_tgt * mx.rsqrt(mx.sum(z_tgt * z_tgt, axis=-1, keepdims=True) + 1e-8)

        cos_sim = mx.sum(z_pred_n * z_tgt_n, axis=-1)  # (batch,)
        mx.eval(cos_sim, z_pred_n)
        cos_sims.append(cos_sim)
        all_z_pred.append(z_pred_n)

    cos_all = mx.concatenate(cos_sims, axis=0)  # (N,)
    mx.eval(cos_all)
    cos_np   = np.array(cos_all)
    mean_cos = float(np.mean(cos_np))

    # --- Cosine breakdown by mask length (1 / 2 / 3 tokens masked) ---
    cos_by_masklen = {}
    for ml in [1, 2, 3]:
        idx = np.where(mask_len == ml)[0]
        cos_by_masklen[ml] = float(np.mean(cos_np[idx])) if len(idx) > 0 else 0.0

    # --- Class accuracy on single-token mask pairs ---
    # Build one prototype embedding per vocab token: target_enc([BOS, tok, PAD...], [1,1,0...])
    proto_tok = np.zeros((VOCAB_SIZE, MAX_RESULT_LEN), dtype=np.int32)
    proto_msk = np.zeros((VOCAB_SIZE, MAX_RESULT_LEN), dtype=np.float32)
    proto_tok[:, 0] = BOS_ID
    proto_tok[np.arange(VOCAB_SIZE), 1] = np.arange(VOCAB_SIZE)
    proto_msk[:, 0] = 1.0
    proto_msk[:, 1] = 1.0
    z_proto = mx.stop_gradient(target_enc(mx.array(proto_tok), mx.array(proto_msk)))
    z_proto_n = z_proto * mx.rsqrt(mx.sum(z_proto * z_proto, axis=-1, keepdims=True) + 1e-8)
    mx.eval(z_proto_n)

    single_idx = np.where(mask_len == 1)[0]
    class_acc = 0.0
    if len(single_idx) > 0:
        z_pred_all = mx.concatenate(all_z_pred, axis=0)             # (N, D)
        z_single   = z_pred_all[mx.array(single_idx, dtype=mx.int32)]  # (M, D)
        sims       = mx.matmul(z_single, mx.transpose(z_proto_n))   # (M, 96)
        pred_toks  = mx.argmax(sims, axis=1)                        # (M,)
        true_toks  = mx.array(res[single_idx, 1], dtype=mx.int32)   # token after BOS
        class_acc  = float(mx.mean((pred_toks == true_toks).astype(mx.float32)).item())

    # --- Sub-expression prediction (Gap 2) ---
    val_subexpr_cos = _evaluate_subexpr_cos(model, target_enc, expr, expr_mask, batch_size)

    return mean_cos, class_acc, cos_by_masklen, val_subexpr_cos


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

vocab = ClojureVocab()

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

loss_grad_fn = nn.value_and_grad(
    model,
    lambda m, et, em, rt, rm, ss, smk: compute_loss(m, target_enc, et, em, rt, rm, ss, smk),
)

print(f"Time budget: {TIME_BUDGET}s")

total_training_time = 0.0
step = 0
t_compiled = None
smooth_loss = 0.0
total_pairs = 0

while True:
    t0 = time.time()

    expr_np, expr_mask_np, res_np, res_mask_np = next(train_loader)
    # Gap 2: bracket matching on CPU before sending to GPU
    span_starts_np, span_mask_np = compute_span_info(expr_np, expr_mask_np)

    expr_t = mx.array(expr_np, dtype=mx.int32)
    expr_m = mx.array(expr_mask_np, dtype=mx.float32)
    res_t  = mx.array(res_np, dtype=mx.int32)
    res_m  = mx.array(res_mask_np, dtype=mx.float32)
    span_starts_t = mx.array(span_starts_np, dtype=mx.int32)
    span_mask_t   = mx.array(span_mask_np,   dtype=mx.float32)

    loss, grads = loss_grad_fn(model, expr_t, expr_m, res_t, res_m, span_starts_t, span_mask_t)
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
val_mean_cos, val_class_acc, cos_by_masklen, val_subexpr_cos = evaluate_val_retrieval(model, target_enc)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

peak_vram_mb = get_peak_memory_mb()

print("---")
print(f"val_mean_cos:        {val_mean_cos:.4f}")
print(f"val_recall_at_1_pct: {val_class_acc * 100:.4f}")
print(f"val_cos_mask1:       {cos_by_masklen[1]:.4f}")
print(f"val_cos_mask2:       {cos_by_masklen[2]:.4f}")
print(f"val_cos_mask3:       {cos_by_masklen[3]:.4f}")
print(f"val_subexpr_cos:     {val_subexpr_cos:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_pairs_M:      {total_pairs / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")