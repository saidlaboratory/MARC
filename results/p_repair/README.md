# MARC v0.3 — selected structural-repair results

This directory contains both development diagnostics and the selected Data Version 6
results.  **Only the files listed under “Citable” use the final anti-shortcut protocol.**

## Method

The candidate-conditioned repair ranker applies every proposed augmentation to the
fixed graph, encodes the resulting polynomial factor graph with operator-aware node
and edge features, and assigns a listwise compatibility score.  The highest-scoring
repair receives one classical solve; the exact checker remains the acceptance gate.

The matched `control` model sees the candidate recipe but no problem graph.  `random`
chooses uniformly from the same K candidates.  Linear menus have exact rank
certificates.  Nonlinear menus have empirical, budget-literal certificates and use
the same refinement family for end-to-end evaluation.

## Citable results

| Result | Full | Candidate-only | Random | Evidence |
|---|---:|---:|---:|---|
| Linear, `shared` held out (N=400) | **0.565** [0.516, 0.613] | 0.343 | 0.283 | `random_support_holdout_shared_paired.json` |
| Linear, all test patterns (N=1,200) | **0.552** [0.523, 0.580] | 0.351 | 0.253 | same; full vs control paired p=5.4e-26 |
| Balanced nonlinear (N=360) | **0.889** [0.852, 0.917] | 0.422 | 0.253 | `nonlinear_balanced_full_paired.json`; paired p=4.1e-46 |
| Nonlinear `quad_link` (N=180) | **0.928** [0.880, 0.957] | 0.222 | 0.228 | same |
| Nonlinear `vieta` (N=180) | **0.850** [0.791, 0.895] | 0.622 | 0.278 | same; report the stronger candidate prior |
| Vieta→`quad_link` relation holdout (N=150) | **0.367** [0.294, 0.446] | 0.240 | 0.213 | `nonlinear_holdout_quad.json`; partial transfer |

Three independent initialization/training-order seeds on shared certified tests:

- Linear: **0.430 / 0.458 / 0.471**, mean 0.453, population SD 0.017;
  candidate-only mean 0.303, random 0.249 (`linear_multiseed.json`).
- Nonlinear: **0.889 / 0.875 / 0.881**, mean 0.881, population SD 0.006;
  candidate-only mean 0.417, random 0.253 (`nonlinear_multiseed.json`).

## End-to-end and amortization

The selected repair is actually applied and solved; these are not label-only scores.

| Suite | Full solve | Candidate-only | Random | Oracle | Enumeration | Calls: full / enum |
|---|---:|---:|---:|---:|---:|---:|
| Linear K=4 (N=300) | **0.540** | 0.333 | 0.253 | 1.000 | 1.000 | 1 / 2.53 |
| Nonlinear K=4 (N=60) | **0.883** | 0.433 | 0.250 | 0.950 | 0.950 | 1 / 2.62 |

For linear menus, a K=4-trained ranker is evaluated without retraining at larger
budgets.  Accuracy remains above controls while enumeration cost grows:

| K | Full | Candidate-only | Random | Enum calls | Policy+solve | Enumeration | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.540 | 0.333 | 0.253 | 2.53 | 3.95 ms | 4.78 ms | 1.21x |
| 8 | 0.347 | 0.213 | 0.140 | 4.56 | 4.48 ms | 10.52 ms | 2.35x |
| 16 | 0.247 | 0.140 | 0.020 | 8.56 | 5.06 ms | 19.80 ms | 3.91x |

Direct K=16 training reached 0.192 and is not selected; the K=4 checkpoint transfers
better (0.247).  This negative stays part of the record.

## Anti-shortcut protocol (Data Version 6)

- Linear gold insertion support is randomized per instance.
- Expression-defined candidates never receive the gold auxiliary value as an input.
- Every nonlinear menu option uses the same defining-expression family, insertion
  support, and coefficients; only the defining offset differs.
- Gold nonlinear offsets cycle uniformly by seed over the family-feasible support;
  distractors use the identical support.
- Menus, order, train/validation/test seeds, and solver restart streams are
  deterministic and disjoint.  End-to-end arms use common random numbers.
- Full-vs-control significance uses exact paired McNemar tests; Wilson intervals
  report instance uncertainty.

## Non-citable diagnostics

Files containing `quick`, `pilot`, `medium`, `linear_full.json`,
`linear_holdout_shared.json`, or pre-`balanced` nonlinear names were used to find and
remove support/value/offset leakage.  They are kept locally as an audit trail but are
gitignored — only the selected files above are committed (see `.gitignore`).
`linear_K16_trained.json` is a valid negative control but not a selected checkpoint.

