# Plan: Pure Cross-Attention Predictor

## 1. Idea
Refine the Query-based Transformer Predictor to use **Pure Cross-Attention**. Currently, the predictor prepends a `[RESULT_QUERY]` to the `hidden_ctx` and runs full bidirectional self-attention over the combined sequence. This causes the context tokens to attend to each other again (redundant) and to the query token (potentially harmful). I will replace this with layers where only the `[RESULT_QUERY]` attends to the `hidden_ctx` via cross-attention. The context tokens will remain fixed throughout the predictor layers.

## 2. Not low-hanging fruit — justify this
This is a structural refinement of the attention mechanism. It moves from a "global workspace" model (where all tokens interact) to a "query-retrieval" model. Mechanistically, this ensures that the `hidden_ctx` acts as a static "key-value" memory that the predictor can probe multiple times without altering the memory itself. It focuses the predictor's capacity entirely on the query's transformation, which should lead to more efficient and better-grounded predictions. It also significantly reduces the computational complexity of the predictor from $O((1+T)^2)$ to $O(T)$ per layer.

## 3. Evidence / references
- **Perceiver IO (Jaegle et al., 2021)**: Successfully uses cross-attention to map large inputs to latents and back to outputs, proving that self-attention on the input is not always necessary if a powerful encoder is used.
- **Standard Transformer Decoders**: The cross-attention layers are what allow the decoder to focus on specific encoder outputs; our predictor is essentially a 1-token decoder.
- **Stable Diffusion / ControlNet**: Use cross-attention to inject conditioning information (context) into a latent process (query).

I will use 3 layers of Cross-Attention + MLP for the predictor to utilize the saved computation.
