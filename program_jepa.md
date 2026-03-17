# nanoJEPA autoresearch protocol

This is the autoresearch protocol for nanoJEPA — Clojure execution prediction in latent space.
Adapted from `program.md`. The editable file is `train_jepa.py`; `prepare_jepa.py` is fixed.

## Setup

To set up a new experiment, work with the user to:

1. **Ensure you are on main**: `git checkout main && git pull origin main`.
2. **Read the in-scope files**:
   - `prepare_jepa.py` — fixed: vocab, data generator, val cache, dataloader. Do not modify.
   - `train_jepa.py` — the file you modify: model architecture, loss, optimizer, hyperparameters.
3. **Verify val cache**: Check `data/val_pairs.txt` exists. If not: `uv run prepare_jepa.py`.
4. **Initialize results_jepa.tsv**: Run `uv run train_jepa.py` once to establish baseline.
5. **Confirm and go**.

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
val_mean_cos:        0.8123
val_recall_at_1_pct: 69.9600
val_cos_mask1:       0.8500
val_cos_mask2:       0.7900
val_cos_mask3:       0.7400
val_subexpr_cos:     0.7100
training_seconds: 60.1
total_seconds:    62.0
peak_vram_mb:     312.4
num_pairs_M:      0.1
num_steps:        2100
num_params_M:     2.1
depth:            4
```

```
grep "^val_recall_at_1_pct:\|^val_mean_cos:\|^peak_vram_mb:" run.log
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

Branch: `main` — commit all changes directly to main.

LOOP FOREVER:

### Step 0 — Plan before you code (MANDATORY)

Before touching `train_jepa.py`, write `plan.md` with the following three sections:

**1. Idea**
State the proposed change in one paragraph. Be specific about what you will modify and why you expect it to help.

**2. Not low-hanging fruit — justify this**
Explicitly argue why this idea is *not* one of the following boring moves:
- Trivial hyperparameter sweep (LR, BS, depth, width, dropout, weight decay)
- Obvious scaling (bigger model, more steps, more data)
- Direct copy of a setting already tried in `results_jepa.tsv`
- Generic "let's try X" with no mechanistic reasoning

Your argument must reference *why* the idea is mechanistically novel, structurally different, or borrowed from a different domain in a non-obvious way. If you cannot write two sentences defending this, pick a different idea.

**3. Evidence / references**
Search the internet if you need to. Look for:
- Papers or blog posts describing this technique in another domain (NLP, vision, RL, audio, protein folding, etc.)
- Known failure modes or success conditions for this idea
- Prior art in JEPA, BYOL, SimCLR, VICReg, I-JEPA, or related self-supervised methods

Summarize what you found in 2–5 bullet points. If a web search turns up nothing useful, say so explicitly and explain why you still think the idea is worth trying.

Commit `plan.md` separately: `git add plan.md && git commit -m "plan: <idea title>"`

---

### Step 1–9 — Execute

1. Check git state: `git status` — confirm you are on `main`.
2. Modify `train_jepa.py` with the idea from `plan.md`.
3. `git add train_jepa.py && git commit -m "experiment: <description>"`
4. Run: `uv run train_jepa.py > run.log 2>&1`
5. Check: `grep "^val_recall_at_1_pct:\|^val_mean_cos:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Run `tail -n 50 run.log` for stack trace.
7. Record in `results_jepa.tsv`.
8. If `val_recall_at_1_pct` **improved** (higher): `git add results_jepa.tsv && git commit --amend --no-edit`
9. If equal or worse: record hash, then `git reset --hard <previous kept commit>`

**Timeout**: ~2 min per experiment (TIME_BUDGET=60s + eval). Kill and discard if >5 min.

**NEVER STOP**: Run autonomously until manually interrupted. If out of ideas, search the web for recent self-supervised learning tricks and think harder.
