# Plan: MoCo Momentum Queue for InfoNCE (2048 negatives)

## 1. Idea

Replace in-batch InfoNCE (25 negatives) with a MoCo-style momentum queue InfoNCE
(2048 negatives). Maintain a circular buffer of 2048 L2-normalised target
embeddings from recent batches. For each training step, concatenate the queue
with the current batch's z_tgt as `all_keys = [queue | z_tgt_batch]`.
Positives are the current batch positions (index QUEUE_SIZE+i), all queue
entries are pure negatives.

The target encoder we already maintain via EMA **is** the momentum encoder from
MoCo — we are completing the MoCo recipe by adding the queue. Memory cost:
2048 × 168 × 4 bytes ≈ 1.4 MB (negligible).

## 2. Not Low-Hanging Fruit — Justify This

- **Not a hyperparameter sweep.** Changes the fundamental structure of the
  InfoNCE loss: qualitative shift from 25 to 2048 negatives.
- **Not scaling.** Zero additional model parameters; the queue is a 1.4 MB
  numpy array, not a network layer.
- **Not tried before.** All prior runs use in-batch negatives only.
- **Mechanistic novelty.** The previous plan.md explicitly identified the
  bottleneck: *"with only 26 in-batch negatives, gradient signal is noisier
  than large-batch SimCLR."* This directly attacks that weakness. With 25
  negatives, InfoNCE entropy ≈ log(26) ≈ 3.3 bits; with 2074 negatives it
  rises to ≈ log(2074) ≈ 11.0 bits — the model must produce far more
  discriminative representations to minimise the loss.

## 3. Evidence / References

- **MoCo (He et al. 2019, 2020)**: introduced the momentum encoder + queue
  to decouple the number of negatives from batch size. QUEUE_SIZE=65536 in
  MoCo v1; much smaller queues still outperform in-batch baselines at small
  batch sizes.
- **SimCLR (Chen et al. 2020)**: Table 1 shows Recall@1 rising monotonically
  with batch size (= negatives). Going from 256 → 4096 negatives improved
  Top-1 by ~10 points. MoCo's queue is the efficient solution when large
  batches are infeasible.
- **MoCo v2 (Chen et al. 2020)**: Combined MoCo queue with improved projector
  and outperformed SimCLR with 256-step batch (vs SimCLR's 4096).
- **Our EMA target encoder is already the MoCo momentum encoder.** We are
  one data structure away from the full recipe.
- Known failure mode: very stale queue entries (target encoder drifts fast).
  Mitigated here by our slow EMA (τ = 0.996 → 0.9999) which keeps the target
  encoder stable, so embeddings from 78 steps ago (queue_size/batch_size) are
  still approximately on-distribution.
