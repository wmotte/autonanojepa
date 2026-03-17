# Plan: Attention Pooling to Replace Mean Pooling

## 1. Idea

Replace the mean-pooling aggregation in `ClojureEncoder` with a learned attention-pooling layer.
Currently, all non-padding tokens receive equal weight when computing the expression embedding:

```python
return mx.sum(x * mask_f, axis=1) / denom  # (B, n_embd)
```

The change: add a single linear projection `D → 1` that scores each token position; apply
softmax over non-padding positions to get per-token weights; then take a weighted sum.
This lets the model learn that **operators and function names** (`+`, `map`, `reduce`, at
position 2 right after BOS) carry more semantic content for execution prediction than
structural bracket tokens (`(`, `)`, `[`, `]`), which are currently counted equally.

The pooling layer adds exactly `N_EMBD = 168` parameters. It is applied in both
`__call__` and `encode_full` so that the sub-expression auxiliary loss also benefits.

## 2. Not Low-Hanging Fruit — Justification

- **Not a hyperparameter sweep.** Changes the aggregation function itself, not a scalar coefficient.
- **Not scaling.** Adds only 168 parameters out of ~2.1M — negligible size increase.
- **Not a copy of anything tried.** Mean-pooling has been unchanged since project start.
- **Mechanistic novelty.** Clojure expressions have highly non-uniform token importance:
  the operator at position 2 determines the computation type entirely, while brackets are
  structural scaffolding. Mean-pooling conflates both. Attention pooling can learn a prior
  over token roles that mean-pooling cannot express. This is structurally different from the
  depth embedding (which only adds a positional bias, not a weighting).

## 3. Evidence / References

- **Sentence-BERT** (Reimers & Gurevych, 2019): compared mean-pool, max-pool, and CLS-token
  pooling across BERT tasks. Mean pooling performs well on symmetric tasks but is outperformed
  by attention-based pooling on asymmetric sequence-to-property tasks — directly analogous
  to expression → result.

- **ESM-2 / ProtTrans** protein language models: attention-weighted pooling over residue
  representations outperforms mean pooling for sequence-level property prediction, an
  analogous "map variable-length sequence → compact embedding" task.

- **Code embedding literature** (CodeBERT, GraphCodeBERT): mean pooling over code tokens
  has been shown to dilute semantic signal with structural noise; token-level attention
  pooling or [CLS]-token extraction is preferred in downstream retrieval tasks.

- **Known failure mode:** attention weights can collapse onto a single token, degrading to
  argmax pooling. The existing VICReg variance regularisation on `z_ctx` provides implicit
  protection since collapsed pooling would reduce variance across the batch — VICReg would
  penalise this.

- Web search found no direct prior art applying attention pooling specifically to JEPA-style
  execution prediction, but the above analogues from NLP and protein modeling justify
  the hypothesis that attention pooling should strictly dominate mean pooling here.
