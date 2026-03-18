# Plan: Hard Negative Mining in MoCo

## 1. Idea
Improve the MoCo-style InfoNCE contrastive loss by implementing **Hard Negative Mining**. Currently, the `infonce_loss` treats all 2048 embeddings in the queue as negatives with equal importance. I will modify the loss to select the Top-512 most similar negatives (hard negatives) from the queue for each batch item. This forces the model to focus its gradients on distinguishing between the most confusingly similar expressions in latent space.

## 2. Not low-hanging fruit — justify this
This is a sophisticated refinement of the contrastive objective. It moves from a "passive" contrastive loss to an "active" one that dynamically adapts to the model's current failures. Mechanistically, hard negative mining increases the signal-to-noise ratio of the gradients by ignoring easy negatives that have already been well-separated. This is particularly relevant for Clojure code, where many expressions may have similar syntactic structures but different execution results (e.g., changing a `+` to a `-`).

## 3. Evidence / references
- **Contrastive Learning with Hard Negative Samples (Robinson et al., 2020)**: Shows that focusing on hard negatives can significantly improve representation quality.
- **FaceNet (Schroff et al., 2015)**: Famously used triplet loss with hard negative mining to achieve state-of-the-art face recognition.
- **DINO / DINOv2**: Use various forms of local-to-global matching and negative management to improve discriminative power.
- **Metric Learning**: Hard negative mining is a staple in training retrieval-based models where Recall@K is the primary metric.

I will use `mx.topk` to select the Top-512 negatives from the MoCo queue for each element in the batch.
