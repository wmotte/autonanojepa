# nanoJEPA Roadmap

## Current State Assessment

### What is working

The project has a solid technical foundation. The JEPA framework is correctly implemented with a proper set of anti-collapse mechanisms (EMA target encoder, VICReg variance and covariance penalties, L2 normalization). The pivot from masked-JEPA to actual execution prediction was the right call — the masked proxy saturated at ~0.98 cosine similarity and provided no useful autoresearch gradient. The current task (predict the embedding of a computed result given the expression) is genuinely harder and provides better learning signal.

Best current result: **94.08% Recall@1** on 2,000 held-out (expression, result) pairs.

Architecture highlights that contributed positively:
- EMA tau cosine schedule (0.996 → 0.9999, I-JEPA style): fast bootstrap early, slow convergence late
- Learned attention pooling (D→1 learned query) replacing mean-pool
- Sub-expression auxiliary loss at bracket positions (sub-expression cosine: 0.9674)
- Bidirectional transformer with depth embeddings encoding syntactic nesting level
- In-batch InfoNCE contrastive loss (T=0.1, weight=0.5)

### Honest limitations

**The task is too narrow to make strong claims about code understanding.** The 25 expression families are hand-crafted synthetic Python-generated data with no actual Clojure runtime. The result space is small (1–8 tokens, mostly single integers), and 94% Recall@1 likely reflects the model learning the 25 family patterns rather than general program semantics.

**The 60-second training budget means the task must stay simple.** A 2.84M parameter model trained for one minute cannot learn a large, diverse distribution. The experimental loop is well-suited for architectural ablations, but any result achieved at this scale needs to be verified at larger scale before claiming it would transfer.

**The gap to real Clojure is large.** Real programs have macros, lazy sequences, namespaces, Java interop, side effects, complex data structures, and multi-file dependencies. None of this is represented. There is no demonstrated transfer path from the current setup to a claim about program comprehension in the wild.

---

## Near-term priority: Expand synthetic complexity

Before reaching for real Clojure data, the synthetic task should be pushed until the model meaningfully struggles. If the current architecture plateaus on harder synthetic data, that is the signal that the bottleneck is actually distribution — which is when real data becomes the right lever.

### Recommended expansions to the synthetic data generator

**Increase nesting depth.** Current expressions reach depth 3–4. Push to depth 5–7 with nested let inside cond inside threading macros, etc. The model should have to track multiple simultaneous variable bindings and multi-step data flow.

**Expand the result type space.** Current results are mostly small integers. Add:
- Vectors and sequences as results: `(map inc [1 2 3])` → `[2 3 4]`
- Boolean results beyond simple `pos?` checks
- `nil` from `when` forms (already present but underweighted)
- Nested maps: `(assoc {} :a 1)` → `{:a 1}`

**Larger integer range.** The vocabulary currently encodes integers −10 to 30 as discrete tokens. Intermediate computation values that overflow this range are clipped. Extending the integer range — or switching to a positional integer encoding — would prevent the model from shortcutting via vocabulary lookup.

**More variable shadowing and sequential binding.** Family J (sequential let) is hard because the result never appears as a literal token. This is the most semantically rich family. Expand it: deeper sequential bindings, multi-level variable shadowing, inner bindings that modify outer variable names.

**Cross-family compositions.** Currently families are sampled independently. Explicitly compose: a threading macro whose intermediate value feeds a HOF that feeds a nested let. This directly tests compositional generalization, which is what the sub-expression auxiliary loss is designed to encourage.

**Increase training data diversity over time budget.** With 60 seconds of training, the generator runs through a finite number of steps. Ensure the family sampling distribution is not dominated by the easier families so the model sees adequate signal from the hard families (I, J, K, U).

---

## On real Clojure data: not yet, and not as training data

### Why not yet

Adding real Clojure programs as training data faces three compounding problems:

**1. The execution problem.** To get `(expression, result)` pairs, you must execute the code. This requires a JVM and a Clojure runtime. Real Clojure code has:
- External library dependencies (`require`, `import`)
- Side effects (IO, network, database calls, random state)
- Macros that need to be expanded before evaluation
- Expressions that throw exceptions
- Non-terminating computations

Isolating pure, safely-executable sub-expressions from real Clojure source files is a substantial engineering task. A naive approach (extract s-expressions, run them) will fail on the vast majority of real code.

**2. The scale mismatch.** Even if the execution pipeline were built, a 2.84M parameter model trained for 60 seconds cannot learn from a large, diverse real-world distribution. You would likely see a sharp drop in Recall@1 and be unable to distinguish "model cannot generalize" from "model has no capacity." Real data requires solving the scale problem first.

**3. The result space explosion.** Real Clojure functions return complex data structures, custom types, lazy sequences, and exceptions. The current evaluation metric (Recall@1 over single-token result prototypes) would not work. The metric, evaluation protocol, and vocabulary encoding would all need to be redesigned.

### The right way to use real data: validation only

A productive intermediate step is to build a small, curated **hard validation set** from real Clojure code — without changing training at all. This tests whether synthetic training generalizes out of distribution, which is the most important scientific question.

**Concretely:**

1. Mine Clojure test files from well-known open-source libraries (clojure.core, clojure/math, etc.). Test files often contain pure assertions of the form `(is (= (f x) expected))`, which are already (expression, result) pairs.

2. Use [Babashka](https://babashka.org/) — a fast-starting Clojure scripting runtime — to execute these expressions and verify the expected values. Babashka starts in ~15ms and supports a large subset of clojure.core with no JVM startup overhead.

3. Filter for expressions whose results fall within the current vocabulary (integers −10 to 30, booleans, nil) to keep the metric compatible.

4. Use this as a fixed **out-of-distribution validation set** — report both in-distribution recall (current val_pairs.txt) and out-of-distribution recall (real Clojure). If the gap is large, the model is pattern-matching on synthetic families. If the gap is small, synthetic pretraining genuinely generalizes.

This is low engineering overhead (no training pipeline changes), directly answers the generalization question, and produces a clear signal about whether the current approach is worth scaling.

---

## Medium-term: address the scale constraints

The autoresearch loop at 60-second experiments is well-suited for architectural ablations, but fundamental questions about the approach (does JEPA-based execution prediction work, period?) require more training compute.

**Graduated training budgets.** Once the architecture stabilizes from autoresearch, evaluate the best configuration at 5min, 30min, and 2hr budgets. If Recall@1 keeps improving with compute, the task is not saturated and scale is the bottleneck.

**Model size scaling.** The current 2.84M parameter model is deliberately tiny for fast iteration. The key question is whether doubling or quadrupling model size (while holding training budget constant) changes which architectural choices matter. Some design decisions that look neutral at 2.84M parameters may become important or harmful at 10M+.

**Batch size scaling.** The current device batch size is 26, constrained by the 1GB memory target. With larger batches, the in-batch InfoNCE has more negatives per step without needing the MoCo queue. The MoCo queue experiment (2048 negatives) did not improve over in-batch InfoNCE — this is worth re-examining at larger batch sizes to separate "more negatives" from "stale negatives."

---

## Long-term: toward real program comprehension

If the approach validates at larger scale and out-of-distribution generalization holds, the natural next steps are:

1. **Build the Babashka execution pipeline** for training on real Clojure expressions. Focus on pure functions from clojure.core and popular utility libraries (medley, specter, etc.) where dependencies are controlled.

2. **Redesign the result encoding** for complex output types (vectors, maps, sets) using a hierarchical or recursive encoder rather than the flat token-sequence encoding used now.

3. **Cross-file context.** The context encoder currently sees one expression at a time. Real program comprehension requires understanding the function definitions and let bindings that surround an expression. Extending to file-level context (with selective attention over the surrounding ns form) is the key architectural challenge.

4. **Evaluate on a real downstream task.** The ultimate test of whether the latent-space execution predictor captures program semantics is whether its embeddings are useful for something: bug detection, code search, refactoring suggestion, or test generation. Pick one concrete downstream task and evaluate there, rather than continuing to optimize Recall@1 on a synthetic benchmark indefinitely.

---

## Summary of priorities

| Priority | Action | When |
|---|---|---|
| 1 | Expand synthetic complexity (deeper nesting, richer result types, cross-family compositions) | Now |
| 2 | Build small real-Clojure validation set via Babashka (no training pipeline changes) | After synthetic expansion plateaus |
| 3 | Graduated training budget experiments (5min, 30min, 2hr) | After architecture stabilizes |
| 4 | Model size scaling experiments | After training budget scaling |
| 5 | Full real-data training pipeline with Babashka execution | If out-of-distribution generalization validates |
| 6 | Redesign result encoding for complex types | Alongside full real-data pipeline |
| 7 | Downstream task evaluation | Final validation |
