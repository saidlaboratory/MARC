# MARC v0.3 — selected structural-repair results

This directory contains both development diagnostics and the selected Data Version 8
results.  **Only the files listed under “Citable” use the final protocol.**  Every
earlier data version is withdrawn: v6 leaked the gold pin prior (its candidate-only
control scored 0.343 against 0.25 chance), and a CAS audit showed most v6/v7 nonlinear
“certified unsolvable” distractors actually have real roots that the weak refine probe
could not find.  Do not cite 0.565 or 0.889.

## Method

The candidate-conditioned repair ranker applies every proposed augmentation to the
fixed graph, encodes the resulting polynomial factor graph with operator-aware node
and edge features, and assigns a listwise compatibility score.  The highest-scoring
repair receives one classical solve; the exact checker remains the acceptance gate.

The matched `control` model sees the candidate recipe but no problem graph.  `random`
chooses uniformly from the same K candidates.  Linear menus have exact rank
certificates.  Nonlinear menus are certified by an exact CAS real-root decision
(a proven-rootless distractor is unsolvable at any budget and any seed); the few
CAS-undecided instances carry the disclosed budget-relative probe certificate and are
counted in each JSON's `certificates` block.  One reference solver (scipy LM, k=4,
owned by `marc/structure/invention_data.py`) certifies golds, grades every arm, and
runs the end-to-end solves.

## Citable results (Data Version 8)

| Result | Full | Candidate-only | Random | Evidence |
|---|---:|---:|---:|---|
| Balanced nonlinear (N=360) | **0.997** [0.984, 1.000] | 0.333 | 0.236 | `nonlinear_balanced_full.json`; paired p=1.1e-72 (239 vs 0) |
| Nonlinear `vieta` (N=180) | **1.000** | 0.311 | 0.211 | same |
| Nonlinear `quad_link` (N=180) | **0.994** | 0.356 | 0.261 | same |
| Vieta→`quad_link` relation holdout (N=150) | **0.420** [0.344, 0.500] | 0.120 | 0.253 | `nonlinear_holdout_quad.json`; partial transfer |
| Linear, `shared` held out (N=400) | **0.380** [0.334, 0.428] | 0.195 | 0.287 | `random_support_holdout_shared.json` |
| Linear, all test patterns (N=1,200) | **0.339** [0.313, 0.366] | 0.207 | 0.249 | same; full>random p=7.8e-07, full>control p=3.5e-13 |

Checkpoint-only replays (`*_paired.json`) reproduce both headline evals exactly.

Closing each shortcut lowered the linear headline (0.565 v6 → 0.445 v7 → 0.380 v8)
and pushed the candidate-only control to chance: the remaining linear signal comes
entirely from reading the problem graph, and it is modest.  The nonlinear numbers
rose (0.889 → 0.997) because v8 menus finally carry theorem-grade “exactly one
solvable option” semantics, which is exactly the discrimination the operator-aware
encoder was built for.

Optimization-seed repeats (v8, train seeds 11/29/47; issue #103 fixed the earlier
artifact where one random-arm draw was reused across seeds): nonlinear full
0.982 ± 0.006 (0.975/0.983/0.989), linear full 0.317 ± 0.069 (0.333/0.392/0.227,
one seed below random 0.248 ± 0.002) — the nonlinear headline is seed-robust, the
linear edge is not (`nonlinear_multiseed.json`, `linear_multiseed.json`).

## End-to-end and amortization

The selected repair is actually applied and solved; these are not label-only scores.
Common restart seeds across arms.

| Suite | Full solve | Candidate-only | Random | Oracle | Enumeration | Calls: full / enum |
|---|---:|---:|---:|---:|---:|---:|
| Nonlinear K=4 (N=60) | **0.933** | 0.200 | 0.250 | 0.933 | 0.933 | 1 / 2.62 |
| Linear K=4 (N=300) | **0.300** | 0.217 | 0.227 | 1.000 | 1.000 | 1 / 2.54 |

The nonlinear ranker's single call sits at the oracle/enumeration ceiling.

For linear menus, the K=4-trained ranker is evaluated without retraining at larger
budgets:

| K | Full | Candidate-only | Random | Enum calls | Policy+solve | Enumeration | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.300 | 0.217 | 0.227 | 2.54 | 3.83 ms | 4.82 ms | 1.26x |
| 8 | 0.187 | 0.100 | 0.120 | 4.40 | 4.15 ms | 10.11 ms | 2.44x |
| 16 | 0.113 | 0.053 | 0.107 | 9.05 | 4.85 ms | 22.54 ms | 4.65x |

The accuracy advantage over random shrinks with K and is gone at K=16 — at that
budget the win is wall-clock only.  This limitation is reported, not smoothed over.
Direct K=16 training performs at chance and is not selected
(`linear_K16_trained.json`); the negative stays part of the record.

## Anti-shortcut protocol (Data Version 8)

- One `REFERENCE_SOLVER` (scipy LM, k=4) for certification, arm grading, training
  reward, and e2e solves; identity across modules is test-enforced.
- Nonlinear distractors are CAS-certified to have no real solution wherever sympy
  can decide (356/360 on the balanced test); anything mathematically equal to the
  gold is excluded from the menu.
- Both nonlinear families use one-sided defining templates over per-family (a, δ)
  supports shared by golds and distractors: `u = a·x² + δ` (quad_link) and
  `u = a·(x²+y²) + δ` (vieta), a = ±1.  A linear defining relation cannot yield
  rootless corruptions (line ∩ hyperbola), which is why the old vieta menus were
  probe artifacts.
- Golds must solve at eval grade under two independent seeds.
- Linear gold insertion support and menu-filler support use the same size-uniform
  sampler; gold and distractor pins draw from one shared prior.
- Menus, order, train/validation/test seeds, and solver restart streams are
  deterministic and disjoint; `seed_hygiene` blocks record real per-source seed
  ranges and a computed (not asserted) instance-id overlap.
- Full-vs-control significance uses exact paired McNemar tests; Wilson intervals
  report instance uncertainty.

## Non-citable diagnostics

All `data_version < 8` files are withdrawn (see above).  Files containing `quick`,
`pilot`, `medium`, `linear_full.json`, `linear_holdout_shared.json`, or
pre-`balanced` nonlinear names were used to find and remove support/value/offset
leakage.  They are kept locally as an audit trail but are gitignored — only the
selected files above are committed (see `.gitignore`).  `linear_K16_trained.json`
is a valid negative control but not a selected checkpoint.
