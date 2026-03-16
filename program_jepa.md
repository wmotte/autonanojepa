# nanoJEPA autoresearch protocol

This is the autoresearch protocol for nanoJEPA — Clojure execution prediction in latent space.
Adapted from `program.md`. The editable file is `train_jepa.py`; `prepare_jepa.py` is fixed.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autoresearch-jepa/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-jepa/<tag>` from current main.
3. **Read the in-scope files**:
   - `prepare_jepa.py` — fixed: vocab, data generator, val cache, dataloader. Do not modify.
   - `train_jepa.py` — the file you modify: model architecture, loss, optimizer, hyperparameters.
4. **Verify val cache**: Check `~/.cache/autoresearch_jepa/val_pairs_v2.npz` exists. If not: `uv run prepare_jepa.py`.
5. **Initialize results_jepa.tsv**: Run `uv run train_jepa.py` once to establish baseline.
6. **Confirm and go**.

## Task

**Clojure execution prediction**: given the embedding of a Clojure expression, predict the embedding of its result. The model works entirely in latent space — no token generation, no vocabulary projection.

**Architecture (JEPA):**
- Context Encoder: bidirectional transformer → mean-pool → z_context
- Predictor: 3-layer MLP → z_pred
- Target Encoder: EMA copy of context encoder (no gradient) → z_target
- Loss: `1 - cosine_sim(z_pred, z_target)` + VICReg variance (z_pred + z_ctx) + covariance reg (z_ctx)

## Experimentation

**What you CAN do:**
- Modify `train_jepa.py` — architecture, optimizer, hyperparameters, loss, EMA tau, everything.

**What you CANNOT do:**
- Modify `prepare_jepa.py`. Fixed. Contains evaluation and data generation.
- Install new packages. Only mlx + numpy available.

**The goal: maximize `val_recall_at_1_pct`** (Recall@1 retrieval accuracy on 2000 held-out pairs, as a percentage). Higher = better. Range: [0, 100]; random chance = 0.05%.

**Memory** is a soft constraint. `peak_vram_mb < 1000` is the target (fits every MacBook).

## Output format

```
---
val_recall_at_1_pct: 41.2345
training_seconds:    120.1
total_seconds:       121.0
peak_vram_mb:        312.4
num_pairs_M:         0.4
num_steps:           5831
num_params_M:        4.7
depth:               4
```

```
grep "^val_recall_at_1_pct:\|^peak_vram_mb:" run.log
```

## Logging results

Log to `results_jepa.tsv` (tab-separated):

```
commit	val_recall_at_1	memory_gb	status	description
```

> **Note**: entries before the mar16 session used `val_pred_sim` (cosine similarity, ~0.97 ceiling) — not comparable to `val_recall_at_1_pct`. Treat those rows as historical only.

1. git commit hash (7 chars)
2. val_recall_at_1_pct achieved (e.g. `1.3500`) — use `0.0000` for crashes
3. peak memory in GB (peak_vram_mb / 1024, .1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

## The experiment loop

Branch: `autoresearch-jepa/<tag>`

LOOP FOREVER:

1. Check git state (branch/commit).
2. Modify `train_jepa.py` with an experimental idea.
3. `git add autoresearch-mlx/train_jepa.py && git commit -m "experiment: <description>"`
4. Run: `uv run train_jepa.py > run.log 2>&1`
5. Check: `grep "^val_recall_at_1_pct:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Run `tail -n 50 run.log` for stack trace.
7. Record in `results_jepa.tsv`.
8. If `val_recall_at_1_pct` **improved** (higher): `git add autoresearch-mlx/results_jepa.tsv && git commit --amend --no-edit`
9. If equal or worse: record hash, then `git reset --hard <previous kept commit>`

**Timeout**: ~3 min per experiment. Kill and discard if >8 min.

**NEVER STOP**: Run autonomously until manually interrupted. If out of ideas, think harder.

## Architectural ambition

Hyperparameter tuning alone is unlikely to produce a meaningful contribution. At the current trajectory (1.40% → 1.85% over a handful of hyperparameter experiments), the model is making incremental gains but has not yet demonstrated genuine execution reasoning. To matter as a contribution to neural execution modelling, the architecture itself must change.

**Directions worth trying, in rough order of invasiveness:**

1. **Deeper / wider predictor**: The 3-layer MLP may be too shallow to simulate multi-step reduction. Try 4–6 layers, residual connections, or a small transformer predictor that operates over the context embedding as a sequence of "reduction steps."

2. **Explicit intermediate state representations**: Add auxiliary loss terms that encourage the predictor's hidden layers to represent identifiable intermediate values (e.g., the result of the inner sub-expression before the outer one is applied). This can be implemented by generating expression–subresult pairs from the data generator's existing structure and supervising intermediate activations.

3. **Curriculum from easy to hard families**: Train first on families A–B (arithmetic, single let), then introduce C–H (HOF, conditionals, multi-binding), and only later expose the model to I–J–K (multiplication, sequential let, cross-product). A staged curriculum lets the model build compositional representations before facing cases where the result is not a literal token in the input.

4. **Separate encoder per syntactic role**: Rather than mean-pooling all tokens together, pool sub-expressions separately (operator, arguments, bindings) and combine via a small attention layer. This injects structural bias without requiring changes to `prepare_jepa.py`.

5. **Contrastive hard-negative mining**: In the retrieval loss, instead of random negatives from the batch, mine hard negatives — results that are close in embedding space but correspond to different expressions. This sharpens the representation boundary between similar-but-distinct execution outcomes.

6. **Recursive / hierarchical encoding**: Clojure expressions are S-expression trees. Instead of treating the token sequence as flat, build a bottom-up composition pass: encode each leaf token, then compose sibling sub-expressions via a small MLP or single-layer attention, then compose at the next level up. This mirrors the actual evaluation order and gives the encoder an inductive bias toward tree structure — which the current flat mean-pooling discards entirely.

7. **Multi-step predictor unrolling**: Replace the single MLP forward pass with K unrolled steps, each refining the predicted embedding. Concretely, the predictor receives `(z_ctx, h_{k-1})` and outputs `h_k`; the final `h_K` is compared to `z_target`. This recurrent structure directly mimics iterative reduction and gives the model a mechanism to perform multi-hop reasoning within the predictor rather than expecting a single MLP to do it all at once.

8. **Momentum contrast queue (MoCo-style)**: The current batch of 96 provides only 95 negatives per anchor. Maintain a first-in-first-out queue of recent `z_target` embeddings (e.g., 2048–4096 entries) as a memory bank. The predictor must retrieve the correct result from this much larger pool, producing a far stronger gradient signal and forcing sharper embeddings without requiring a larger batch.

9. **Auxiliary result-type classification head**: Add a small classification head on top of `z_pred` that predicts a coarse category of the result (e.g., negative integer, zero-to-ten, large positive, boolean, list). Train this with cross-entropy alongside the primary cosine loss. The auxiliary signal provides a direct semantic anchor — the model cannot satisfy the type head via representation collapse — and costs almost nothing computationally.

10. **Symmetric / bidirectional JEPA**: Also train a reverse predictor that maps `z_target → z_context`. This symmetric objective forces both encoders to embed expressions and results in a mutually aligned space rather than two independent manifolds that happen to be pulled together only from one direction. The reverse predictor shares no weights with the forward one; it adds roughly 0.8M parameters and doubles the gradient signal per batch.

The bar for a meaningful result is approximately **30–50% recall@1 on the full validation set**, sustained across the hard families (I, J, K). Until then, architectural changes — not hyperparameter search — are the primary lever.
