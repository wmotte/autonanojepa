# nanoJEPA — Clojure Masked-JEPA on Apple Silicon

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

An MLX implementation of a Joint Embedding Predictive Architecture (JEPA) whose task is **masked code understanding**: given a Clojure expression with a span of tokens replaced by a single opaque `MASK`, predict the embedding of the masked span — without ever generating tokens.

This runs natively on Apple Silicon via MLX (unified memory, no PyTorch, no CUDA). The autoresearch loop (`program_jepa.md`) iteratively improves the model within a fixed time budget.

---

## Why JEPA for Clojure?

Traditional language model evaluation requires:
1. A vocabulary projection head (lm_head)
2. Token-by-token generation
3. Cross-entropy loss over a discrete distribution

JEPA sidesteps all of this. The model works **entirely in latent space**:

```
(+ MASK 5)          →  Context Encoder  →  z_context
                                                ↓
                                           Predictor  →  z_pred
                                                              ↓
                                             Loss: 1 − cosine_sim(z_pred, z_target)
                                                    + VICReg variance (z_pred + z_ctx)
                                                    + covariance reg (z_ctx)

[3]  →  Target Encoder (EMA)  →  z_target      (the masked span)
```

The masked span is replaced by a **single opaque `MASK` token** regardless of span length (1–5 tokens), so the model cannot infer how many tokens are missing — it must predict both the content and the extent of the gap. This is harder than BERT-style masking (which uses one `[MASK]` per token and reveals span length) and avoids token-level classification in favour of latent-space prediction.

---

## Architecture

### Context Encoder
A 4-layer **bidirectional** Transformer (no causal mask). Input tokens attend to all positions — appropriate since we process a complete expression, not generate autoregressively.

- Embedding: `nn.Embedding(96, 256)`
- 4 × `EncoderBlock`: bidirectional attention (8 heads, head_dim=32) + MLP (256→1024→256)
- RoPE positional encoding on Q and K, followed by QK-norm
- **Mean pooling** over non-pad positions → single vector `z ∈ ℝ²⁵⁶`
- Final LayerNorm on pooled output

### Target Encoder
Identical architecture to the context encoder, but **updated only via Exponential Moving Average** of the context encoder's weights (τ=0.996). It never receives gradients — it's the "teacher" that provides stable training targets.

### Predictor
A **3-layer MLP**: `256 → 1024 → 1024 → 256` (ReLU activations). Maps the context embedding to the predicted target embedding.

### Parameter count

| Component | Parameters | Optimizer state |
|---|---|---|
| Context encoder | ~3.17M | Yes |
| Predictor (fc1 + fc2 + fc3) | ~0.79M | Yes |
| **Total trainable** | **~3.96M** | Yes |
| Target encoder | ~3.17M | **No** (EMA only) |

Total model footprint: well under 100 MB. Fits on any MacBook.

---

## Anti-collapse mechanisms

JEPA without safeguards collapses: the predictor learns to output a constant vector, and the EMA encoder learns to encode everything as the same constant — achieving zero loss trivially.

Four layers prevent this:

1. **EMA target encoder** (primary): The target encoder drifts slower than the context encoder, providing a stable but moving target. The predictor cannot "catch up" to a collapsing constant because the target itself keeps evolving.

2. **VICReg variance on `z_pred`**: Penalises any dimension of the predictor output whose standard deviation drops below `VICREG_GAMMA=1.0`. Directly prevents predictor dimensional collapse.

3. **VICReg variance on `z_ctx`**: Same penalty applied to the context encoder output. Because the target encoder is an EMA shadow of the context encoder, forcing `z_ctx` to maintain spread prevents the silent collapse that can occur in the target encoder even when `z_pred` looks healthy.

4. **Covariance regularization on `z_ctx`** (`VICREG_COV=0.04`): Penalises the off-diagonal elements of the `z_ctx` covariance matrix. Decorrelates the 256 encoder dimensions so each dimension carries independent information rather than redundant projections of the same collapsed subspace.

5. **L2-normalisation before cosine loss**: Prevents the trivially-zero solution of predicting the zero vector.

### Why `z_ctx` regularization matters

The target encoder has no direct gradient path — it is updated purely via EMA — so its collapse cannot be penalised directly. Regularising `z_ctx` (which is gradient-connected) solves this indirectly: the EMA update propagates spread from the context encoder into the target encoder. The covariance term additionally prevents all dimensions from collapsing onto a low-dimensional manifold while still passing the variance check.

---

## Vocabulary (96 tokens)

| Range | Tokens | Count |
|---|---|---|
| 0–3 | `PAD BOS SEP EOS` | 4 |
| 4–9 | `( ) [ ] { }` | 6 |
| 10–19 | `+ - * / mod max min abs inc dec` | 10 |
| 20–27 | `map filter reduce count first last rest cons` | 8 |
| 28–32 | `assoc get keys vals merge` | 5 |
| 33–37 | `let fn if cond def` | 5 |
| 38–44 | `x y z a b n m` | 7 |
| 45–85 | integers −10 to 30 as discrete tokens | 41 |
| 86–89 | `true false nil pos?` | 4 |
| 90–95 | `MASK -> ->> when not =` | 6 |

No external tokenizer. No subword merges. Tokens map 1:1 to Clojure syntax elements.

---

## Expression families

All expressions are generated **purely in Python** — no Clojure runtime required. The generator maintains the distribution at runtime, so training data is infinite and non-repeating.

A random contiguous span (1–5 tokens) is masked before encoding. The single opaque `MASK` token replaces the whole span, so the model must infer both content and length.

### A — Arithmetic (20%)
```clojure
(+ 3 (* 2 5))          → 13
(abs -7)               → 7
(max -4 (max 9 12))    → 12
```
Simple binary ops, single nesting, and unary ops (`abs`, `inc`, `dec`). Values clipped to [−10, 30].

### B — Let bindings (12%)
```clojure
(let [x 3] (+ x 2))   → 5
(let [a 13] (* a 9))  → 30
(let [n 7] (dec n))   → 6
```
Single binding; body applies one operation to the bound variable.

### C — Higher-order functions (8%)
```clojure
(reduce + [1 2 3])     → 6
(count [1 2 3 4 5])    → 5
(first [7 3 9])        → 7
(last [2 4 6])         → 6
```
Covers `count`, `reduce` (+/max/min), `first`, `last` on literal vectors.

### D — Conditionals (7%)
```clojure
(if (pos? -3) 8 5)                → 5
(if (pos? (+ 3 -5)) 4 -2)        → 4
```
`pos?` predicate; then/else branches return literal integers.

### E — Multi-binding let (9%)
```clojure
(let [x 3 y 4] (+ x y))          → 7
(let [a 5 b 2] (* a b))          → 10
(let [n 3 m 7] (max n m))        → 7
```
Two variables bound simultaneously; body uses both. Tests that the model tracks multiple bindings.

### F — HOF → arithmetic (7%)
```clojure
(inc (first [7 3 9]))             → 8
(+ (count [1 2 3]) 2)             → 5
(- (reduce + [4 3]) 3)            → 4
```
Arithmetic applied to a HOF result. Requires composing two semantic steps.

### G — Let + conditional (5%)
```clojure
(let [x 5] (if (pos? x) 1 0))            → 1
(let [a 4] (if (pos? (- a 8)) 5 -4))     → -4
```
Binding followed by a conditional test on the bound variable.

### H — Depth-3 arithmetic (8%)
```clojure
(+ (+ 2 3) (* 4 1))              → 9
(max (- 8 -4) (+ -2 2))         → 12
```
Two nested binary ops under a root op; no variable references.

### I — Multiplication / products (12%) **hard**
```clojure
(reduce * [2 3 4])               → 24
(* 3 (* 2 4))                    → 24
(let [x 4] (* x x))             → 16
```
Products are typically absent from the expression's literal tokens. `(* x x)` = x² where the result has no syntactic connection to the input.

### J — Sequential let (8%) **hard**
```clojure
(let [a 3 b (* a 4)] (+ b 2))   → 14
(let [x 2 y (+ x 3)] (* y 2))  → 10
```
Second binding uses the first (true Clojure `let` semantics). Result requires two levels of computation; the answer virtually never appears as a token in the expression.

### K — HOF cross-product (4%) **hard**
```clojure
(* (reduce + [1 3]) 2)           → 8
(mod (reduce + [3 5 7]) 4)       → 3
```
Aggregate a sequence, then apply a second operation. The `mod` variant produces a small residue that is maximally distant from all input tokens.

### L — Threading macro (5%)
```clojure
(-> 3 (inc) (+ 2))               → 6
(-> 5 (dec) (dec))               → 3
```
Uses the `->` threading macro. The value threads through each step; tests whether the model understands positional data flow.

### M — When form (4%)
```clojure
(when (pos? 4) 7)                → 7
(when (pos? -2) 5)               → nil
```
`when` returns the body when the condition is truthy, `nil` otherwise. Introduces `nil` as a meaningful result.

### N — Equality (2.5%)
```clojure
(= 3 3)                          → true
(= 4 7)                          → false
```
Simple equality check; result is boolean. Tests the model's ability to predict `true`/`false` from numeric comparison.

### O — Logical not (2%)
```clojure
(not (pos? -3))                  → true
(not (pos? 5))                   → false
```
Logical negation of a `pos?` predicate.

### P — map with fn (7%) — 14–18 tokens
```clojure
(map (fn [x] (inc x)) [1 2 3 4])      → [2 3 4 5]
(map (fn [x] (+ x 2)) [3 -1 5])       → [5 1 7]
(map (fn [x] (* x 3)) [1 2 3])        → [3 6 9]
```
Anonymous function mapped over a literal vector. The result is a collection; in masked-JEPA the execution result is irrelevant — the masked span is the prediction target.

### Q — filter with fn (4%) — 16–21 tokens
```clojure
(filter (fn [x] (pos? x)) [1 -2 3 -4 5])         → [1 3 5]
(filter (fn [x] (pos? (+ x 2))) [0 -1 -3 1])     → [0 1]
```
Filters a vector by an anonymous predicate. Longer than map; the predicate can include arithmetic.

### R — reduce with fn (3%) — 17–20 tokens
```clojure
(reduce (fn [a b] (+ a b)) 0 [1 2 3 4])    → 10
(reduce (fn [a b] (* a b)) 1 [2 3 4])      → 24
(reduce (fn [a b] (max a b)) 0 [3 1 5 2])  → 5
```
Explicit three-argument `reduce` with an anonymous combining function. The initial value is always present.

### S — nested let (5%) — 17–20 tokens
```clojure
(let [x 3] (let [y (+ x 2)] (* y 4)))    → 20
(let [a 4] (let [b (inc a)] (- b 3)))    → 2
```
Two nested `let` bindings; the inner binding refers to the outer variable.

### T — triple let (4%) — 16–19 tokens
```clojure
(let [x 2 y 3 z (+ x y)] (inc z))        → 6
(let [a 4 b 2 c (* a b)] (+ c 5))        → 13
```
Three bindings in one `let`; the third binding computes from the first two.

### U — cond form (3%) — 22–25 tokens
```clojure
(let [x 5]  (cond (pos? x) 8 (pos? (inc x)) 1 true -3))  → 8
(let [n -1] (cond (pos? n) 8 (pos? (inc n)) 1 true -3))  → 1
(let [a -5] (cond (pos? a) 8 (pos? (inc a)) 1 true -3))  → -3
```
Multi-branch `cond` inside a `let`. The longest family at ~24 tokens; the three branches exercise different evaluation paths through the same expression structure.

---

## Running the agent

Spin up Claude Code (or any capable coding agent) in this repo with all permissions disabled, then prompt:

> Have a look at `program_jepa.md` and let's kick off a new experiment — let's do the setup first.

`program_jepa.md` acts as a lightweight skill: it gives the agent everything it needs to run the autoresearch loop autonomously — branch naming, what to edit, how to run, how to log results, and when to keep or discard a run.

---

## Setup

```bash
# One-time: generate and cache 2000 validation pairs
uv run prepare_jepa.py

# Train for 2 minutes
uv run train_jepa.py
```

No data downloads. No internet access required after setup. The validation set is deterministic (seed=12345) and cached at `data/val_pairs.txt`.

### Sample training pairs

Each training sample is a masked Clojure expression paired with the hidden span. The number in parentheses is the span length; the model does not see it — only a single `MASK` token appears in the expression regardless.

```
masked expression                                           → target span
─────────────────────────────────────────────────────────────────────────
( let [ y MASK if ( pos? y ) 3 -4 ) )              (3)  →  11 ] (
( reduce ( fn [ a MASK ( + a b ) ) 0 [ 4 2 4 6 ] ) (2)  →  b ]
( let [ a 7 ] ( let [ b ( - a 2 ) ] ( inc MASK )   (3)  →  b ) )
( let [ b MASK 10 ] ( * b m ) )                    (2)  →  -3 m
( MASK y 4 b ( + y 3 ) ] ( dec b ) )               (2)  →  let [
( inc MASK first [ 9 0 11 ] ) )                    (1)  →  (
( MASK ] ( * x x ) )                               (4)  →  let [ x 2
( let [ b -5 ] MASK max b 2 ) )                    (1)  →  (
( last [ MASK 1 ] )                                (1)  →  15
( let [ MASK -3 z -1 a ( min y z ) ] ( - a 4 ) )   (1)  →  y
```

Span lengths 1–5 are sampled at 40/25/20/10/5%. Single-token masks (like rows 6, 8, 9, 10) are used for the `val_class_acc_pct` metric; longer masks (rows 1–5, 7) create harder prediction tasks that force the model to reason about structural context.

### What the training loop does

Each step draws a fresh batch of masked Clojure expressions. For every expression in the batch a random span has already been replaced by a single opaque `MASK` token; the original tokens of that span are the target:

```python
# One training step (simplified from train_jepa.py)

# 1. Encode the masked expression → context embedding
z_ctx  = model.ctx_encoder(expr_tokens, expr_mask)   # (B, D)

# 2. Predict the target embedding from context
z_pred = model.predict(z_ctx)                         # (B, D)

# 3. Encode the masked span with the EMA target encoder (no gradients)
z_tgt  = target_enc(span_tokens, span_mask)           # (B, D)  stop_gradient

# 4. Cosine loss: push predicted embedding toward target
z_pred_n = normalize(z_pred)
z_tgt_n  = normalize(z_tgt)
loss = 1.0 - mean(dot(z_pred_n, z_tgt_n))

# 5. VICReg: prevent dimensional collapse of z_pred and z_ctx
loss += VICREG_LAMBDA * (variance_penalty(z_pred) + variance_penalty(z_ctx))
loss += VICREG_COV    * covariance_penalty(z_ctx)

# 6. Symmetric JEPA: also predict expression embedding from span embedding
z_rev  = model.predict_reverse(z_tgt)
loss  += 0.5 * (1.0 - mean(dot(normalize(z_rev), normalize(z_ctx))))

# 7. Gradient step on model; EMA update of target encoder (no gradients)
optimizer.update(model, grads)
target_enc ← τ · target_enc + (1−τ) · model.ctx_encoder   # τ = 0.999
```

The target encoder is updated only through the EMA — it never receives gradients directly. This prevents trivial solutions: the predictor cannot simply copy the input because the target keeps evolving as the EMA slowly pulls it toward the context encoder.

---

## Files

| File | Role | Editable? |
|---|---|---|
| `prepare_jepa.py` | Vocab, generator, eval cache, dataloader | **Fixed** — do not modify (autoresearch loop) |
| `train_jepa.py` | Model, loss, optimizer, training loop | **Yes** — autoresearch target |
| `program_jepa.md` | Autoresearch loop protocol | Reference |
| `results_jepa.tsv` | Experiment log | Append-only |
| `README.md` | This file | Reference |

---

## Evaluation metrics

Two complementary metrics are reported after training:

### `val_mean_cos`
Mean cosine similarity between `predict(encode(masked_expr))` and `target_enc(masked_span)` over 2000 held-out pairs.

Range: [−1, 1]. Higher is better. ~0 = random; 1.0 = perfect alignment. Unaffected by duplicate target values (unlike retrieval metrics) because it scores each pair independently.

### `val_class_acc_pct`
Token-level accuracy on the ~40% of val pairs where `mask_len == 1` (single masked token). For each such pair, the predicted embedding is compared against the prototype embeddings of all 96 vocabulary tokens (computed by passing `[BOS, token]` through the target encoder); the nearest neighbour must match the true masked token.

Range: [0, 100]. Random baseline = 1/96 ≈ **1%**.

This metric is collapse-resistant: a model that outputs a constant vector maps to the same token for all inputs, giving near-random accuracy even if `val_mean_cos` looks inflated.

```bash
grep "^val_mean_cos:\|^val_class_acc_pct:" run.log
```

---

## Training dynamics and the time budget

The model converges **rapidly** due to the small, structured vocabulary and synthetic data. With harder expression families and full VICReg, the loss curve is less smooth but more meaningful:

| Step | % done | Notes |
|---|---|---|
| 0 | 0% | Loss starts ~2–3 (random) |
| ~10 | ~0.2% | Fast initial descent |
| ~500 | ~8% | EMA encoder beginning to stabilise |
| ~1,000 | ~17% | Typical plateau / brief rise as EMA drifts |
| ~6,000 | 100% | End of 120 s budget |

The rise-then-plateau pattern is expected JEPA behaviour: the EMA target encoder keeps drifting, so the prediction target is never static. The covariance regularizer slows initial loss descent slightly but prevents the false convergence seen in earlier runs.

### Time budget

`TIME_BUDGET = 120 s` (2 minutes). Reasoning:
- ~20 ms/step → 120 s ≈ 6,000 steps
- Shorter budget raises experiment throughput for the autoresearch loop (~7/hour → ~20/hour)
- 120 s gives the EMA encoder time to properly stabilise (τ=0.996 → half-life ≈ 170 steps)
- LR schedule: 5% linear warmup + flat + 40% cosine decay

---

## The autoresearch loop

See `program_jepa.md` for the full protocol. In short:

1. Create branch `autoresearch-jepa/<tag>`
2. Edit `train_jepa.py` (hyperparameters at the top, or deeper architectural changes)
3. Commit → run → grep → log → keep/discard

Things worth trying:
- Larger encoder (N_EMBD=512, DEPTH=6)
- Stronger VICReg (higher VICREG_LAMBDA or VICREG_COV)
- Lower EMA tau (faster target drift, e.g. τ=0.99)
- Batch size sweep (64 → 128 → 256)
- Separate encoder for expressions vs. results (asymmetric JEPA)
- Recall@5 or MRR as secondary metrics

---

## Relationship to `train.py`

| | `train.py` (GPT) | `train_jepa.py` (nanoJEPA) |
|---|---|---|
| Task | Next-token prediction | Masked span prediction in latent space |
| Attention | Causal (sliding window) | Bidirectional (no mask) |
| Output head | `lm_head` (vocab projection) | None — no token generation |
| Loss | Cross-entropy | Cosine + VICReg (pred + ctx) + covariance |
| Metric | `val_bpb` (lower = better) | `val_mean_cos` + `val_class_acc_pct` (higher = better) |
| Target | Fixed (ground truth tokens) | Moving (EMA encoder) |
| Data | Web text (parquet shards) | Synthetic Clojure pairs (infinite) |
| Memory | ~860 MB | ~860 MB |
| Params | ~3.7M at DEPTH=4 | ~4.0M at DEPTH=4 |
