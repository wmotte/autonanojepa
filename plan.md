# Plan: EMA Tau Cosine Schedule (I-JEPA style)

## 1. Idea

Replace the fixed `EMA_TAU = 0.999` with a cosine ramp from `TAU_START = 0.996` to
`TAU_END = 0.9999` over the training budget:

```
tau(t) = TAU_END - (TAU_END - TAU_START) * (cos(π * progress) + 1) / 2
```

At `progress=0`: `tau = 0.996` (faster EMA, target tracks context encoder quickly).
At `progress=1`: `tau = 0.9999` (slow EMA, stable target for precise alignment).

## 2. Not Low-Hanging Fruit — Justification

- **Not a hyperparameter sweep.** Changes the *shape* of the EMA schedule (constant → cosine
  curve), not just its final value. A sweep over fixed tau values would not discover that
  a schedule outperforms any constant.
- **Not scaling.** Zero additional parameters or compute.
- **Not a copy of prior experiments.** All previous runs used fixed tau.
- **Mechanistic novelty.** Our target encoder is intentionally initialised independently
  (different random state from the context encoder). Early in training, the target encodes
  random noise — a fixed high tau (0.999) means it takes ~1000 steps to move appreciably,
  wasting early training budget on a meaningless prediction target. A lower starting tau
  (0.996) makes the target converge ~4× faster in expectation, compressing the noisy
  bootstrap phase. As training stabilises, a higher tau is needed to prevent the target
  from chasing the still-learning context encoder too aggressively.

## 3. Evidence / References

- **I-JEPA** (Assran et al., 2023): explicitly uses cosine tau schedule from 0.996 → 0.9999.
  Ablations show fixed tau performs worse; the schedule is highlighted as a key design choice.
- **BYOL** (Grill et al., 2020): uses a cosine schedule from 0.996 → 1.0. Authors note
  that the schedule "provides implicit curriculum" and that fixed tau at 0.996 causes
  representation collapse in some configurations.
- **MoCo v3** (Chen et al., 2021): uses fixed tau=0.99 but notes that lower tau values
  help early convergence; later work moved to schedules.
- Known failure mode: if TAU_START is too low (e.g. 0.9), the target encoder tracks the
  context encoder so closely it collapses to a trivial solution. 0.996 is the value used
  in I-JEPA, where it was empirically validated.
