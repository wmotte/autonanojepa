# Plan: Expand Synthetic Complexity

## Idea
The goal is to move from 25 relatively shallow, flat expression families to a unified recursive generation system that produces deeper nesting (4–6 levels) and cross-family compositions. We will also introduce more return types (keywords, strings, maps, vectors, nil, booleans) and expand the result vocabulary. This will be achieved by implementing a recursive `generate_expr(depth)` function that can compose arithmetic, logic, higher-order functions, threading macros, and let-bindings at any level of the tree.

## Not low-hanging fruit — justify this
This is not a simple parameter tweak. It requires:
1.  **Refactoring the data generation core**: Moving from a flat list of templates to a recursive grammar with a consistent internal evaluator.
2.  **Expanding the type system**: The current model mostly handles integers. Adding keywords, strings, and collections requires the model to learn much richer latent representations and polymorphic operations (e.g., `count` working on vectors vs maps).
3.  **Increasing structural entropy**: Depth 4–6 nesting with cross-family composition (e.g., `(-> (let [x 5] (if (pos? x) [x] [])) (conj 2) count)`) creates a much larger and more diverse state space than the current template-based approach.

## Evidence / references
- **Recursive Task Complexity**: Papers like "Deep Learning for Symbolic Mathematics" (Lample & Charton, 2019) show that transformer-based models can solve complex symbolic tasks if trained on sufficiently diverse recursive data.
- **JEPA for Program Semantics**: Joint-Embedding architectures benefit from tasks that require understanding "local transitions" in semantics. Deeply nested expressions force the context encoder to maintain hierarchical state.
- **Compositional Generalization**: Research suggests that models trained on composed primitives generalize better to unseen combinations. Cross-family composition is the primary way to test and enforce this.
