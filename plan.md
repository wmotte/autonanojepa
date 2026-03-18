# Plan: Query-based Transformer Predictor

## 1. Idea
The current model pools the context expression into a single vector `z_ctx` using an `AttentionPool`, then passes this vector through a 3-layer MLP to predict the result embedding `z_tgt`. This "bottleneck" mean-pooling may lose structural information critical for Clojure execution (e.g., the specific pairing of operators and operands in nested forms).

I propose replacing the MLP predictor with a **Query-based Transformer Predictor**. Instead of pooling, the predictor will use a learned `[RESULT_QUERY]` token to attend to the full sequence of hidden states (`hidden_ctx`) from the context encoder. This allows the model to dynamically aggregate information from relevant parts of the expression to form its prediction in latent space.

## 2. Not low-hanging fruit — justify this
This is not a simple hyperparameter sweep or scaling move. It changes the fundamental information flow of the JEPA architecture from a "collapse-then-predict" (bottleneck) approach to a "query-then-predict" (attention) approach. It introduces a structural bias that the result of an expression depends on specific components that should be retrieved via attention, rather than a uniform average of all tokens. Mechanistically, this allows the predictor to maintain higher-order structural relationships that are usually lost during mean-pooling.

## 3. Evidence / references
- **I-JEPA (Assran et al., 2023)**: Uses a Transformer-based predictor to predict masked target patches from context patches, showing that a powerful predictor is essential for latent-space prediction.
- **Cross-Attention in Seq2Seq**: Standard practice in NMT and code generation (e.g., Transformer decoders) where a query attends to encoder states. We are applying this principle purely in latent space for retrieval.
- **Perceiver (Jaegle et al., 2021)**: Uses a small set of latent queries to attend to large inputs, which is efficient and helps in extracting relevant features.
- **CodeBERT/GraphCodeBERT**: Show that structural awareness (like attention to specific nodes or tokens) is crucial for understanding code semantics.

I will also reduce `N_EMBD` to `128` (from `168`) and `DEVICE_BATCH_SIZE` slightly to try and bring `peak_vram_mb` closer to the 1000MB target, as the baseline is currently at ~1.3GB.
