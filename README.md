# nanoJEPA — Clojure Execution Prediction on Apple Silicon

An MLX implementation of a Joint Embedding Predictive Architecture (JEPA) whose task is to **evaluate Clojure expressions in latent space**: given the embedding of a Clojure expression, predict the embedding of its result — without ever generating tokens.

This runs natively on Apple Silicon via MLX (unified memory, no PyTorch, no CUDA). The autoresearch loop (`program_jepa.md`) iteratively improves the model within a fixed time budget.

---

## Why JEPA for Clojure?

Traditional language model evaluation requires:
1. A vocabulary projection head (lm_head)
2. Token-by-token generation
3. Cross-entropy loss over a discrete distribution

JEPA sidesteps all of this. The model works **entirely in latent space**:

```
Clojure expression  →  Context Encoder  →  z_context
                                                ↓
                                           Predictor  →  z_pred
                                                              ↓
                                             Loss: 1 − cosine_sim(z_pred, z_target)
                                                    + VICReg variance (z_pred + z_ctx)
                                                    + covariance reg (z_ctx)

Clojure result  →  Target Encoder (EMA)  →  z_target
```

This is a natural fit for functional/immutable Clojure semantics: a Clojure expression and its result have a **deterministic semantic relationship** that is easier to encode as a vector transformation than as a sequence of output tokens.

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
| 90–95 | reserved | 6 |

No external tokenizer. No subword merges. Tokens map 1:1 to Clojure syntax elements.

---

## Expression families

All expressions are generated **purely in Python** — no Clojure runtime required. The generator maintains the distribution at runtime, so training data is infinite and non-repeating.

The distribution is intentionally shifted toward families where the result does **not** appear as a literal token in the expression (marked **hard** below), making surface-statistics shortcuts insufficient.

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

---

## Setup

```bash
# One-time: generate and cache 2000 validation pairs
uv run prepare_jepa.py

# Train for 2 minutes
uv run train_jepa.py
```

No data downloads. No internet access required after setup. The validation set is deterministic (seed=12345) and cached at `~/.cache/autoresearch_jepa/val_pairs_v2.npz`.

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

## Evaluation metric

`val_recall_at_1_pct` — retrieval accuracy on 2000 held-out (expression, result) pairs, expressed as a percentage:

```
For each expr_i in val set:
  rank all 2000 target embeddings by cosine_sim(predict(encode(expr_i)), target_enc(result_j))
  Recall@1 = fraction where argmax_j == i  (multiplied by 100 for output)
```

Range: [0, 100]. Higher is better. A value of 100 means every expression retrieves its exact result as the nearest neighbour among 2000 candidates. Random chance = 1/2000 = **0.05%**.

**Why not cosine similarity?** The old `val_pred_sim` metric (average cosine similarity against the paired target) is gameable by representation collapse: if all embeddings collapse to the same vector, cosine similarity is trivially ~1.0 even though the model has learned nothing. Retrieval accuracy cannot be gamed this way — collapsed embeddings yield near-zero recall because all 2000 predictions tie and `argmax` picks the first index for all queries.

```bash
grep "^val_recall_at_1_pct:" run.log
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
| Task | Next-token prediction | Execution prediction in latent space |
| Attention | Causal (sliding window) | Bidirectional (no mask) |
| Output head | `lm_head` (vocab projection) | None — no token generation |
| Loss | Cross-entropy | Cosine + VICReg (pred + ctx) + covariance |
| Metric | `val_bpb` (lower = better) | `val_recall_at_1_pct` (higher = better, %) |
| Target | Fixed (ground truth tokens) | Moving (EMA encoder) |
| Data | Web text (parquet shards) | Synthetic Clojure pairs (infinite) |
| Memory | ~860 MB | ~860 MB |
| Params | ~3.7M at DEPTH=4 | ~4.0M at DEPTH=4 |
