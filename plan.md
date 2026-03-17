# Plan: In-batch InfoNCE Contrastive Loss

## 1. Idea

Add an InfoNCE (NT-Xent) contrastive term to the loss. For each `z_pred[i]` in the
batch, treat `z_tgt[i]` as the positive and all other `z_tgt[j≠i]` as negatives:

```
L_nce = mean_i { -log( exp(sim(z_pred_i, z_tgt_i)/T) /
                        sum_j exp(sim(z_pred_i, z_tgt_j)/T) ) }
```

Temperature `T = 0.1`. Added at weight `INFONCE_WEIGHT = 0.5` on top of the existing
cosine + VICReg + symmetric + subexpr losses.

## 2. Not Low-Hanging Fruit — Justification

- **Not a hyperparameter sweep.** Introduces a structurally different loss term with a
  qualitatively different gradient signal.
- **Not scaling.** Zero additional parameters; O(B²) extra matmul at batch size 26 is negligible.
- **Not tried before.** All prior runs use only cosine attraction + VICReg variance/covariance.
- **Mechanistic novelty.** VICReg operates *per-dimension* (variance of each feature across
  the batch). InfoNCE operates *per-sample* (distance between all pairs). They are
  complementary: VICReg prevents feature collapse, InfoNCE prevents sample-level
  similarity collapse (different expressions mapping to the same z_pred). In retrieval
  terms, VICReg spreads the embedding space dimensionally, InfoNCE spreads it sample-wise.

## 3. Evidence / References

- **SimCLR** (Chen et al., 2020): NT-Xent (= InfoNCE) is the core loss; shown to dramatically
  outperform margin-based losses and plain cosine loss for representation learning.
- **MoCo** (He et al., 2020): InfoNCE with a momentum encoder (our EMA target encoder is
  structurally identical) is the defining contribution.
- **CLIP** (Radford et al., 2021): InfoNCE on image–text pairs is exactly analogous to our
  expression–result pairs — asymmetric modalities, alignment via contrastive loss.
- Known risk: with only 26 in-batch negatives, the gradient signal is noisier than large-batch
  SimCLR (4096+). Choosing T=0.1 rather than 0.07 softens the peakiness, trading some
  discrimination for stability at small batch size.
