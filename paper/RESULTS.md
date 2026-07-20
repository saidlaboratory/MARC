# MARC — consolidated results (paper-ready, honest)

All numbers carry a solver label and, where applicable, N and 95% CI. Provenance in
`paper/PROVENANCE.md`. Reproduce commands are listed per result. `refine` is always a
classical baseline, never the headline.

---

## R1 · The learned solver converges (was 0%, diverging)
Fixing five bugs (denoiser never saw `x_t`; constants absent from graph tensors; `t` not
conditioning variables; guidance explosion; a 1-var `squeeze` bug) took the learned solver
from *diverging (≈1e4, 0% solve)* to **solve_rate 1.000** on convex linear systems
(in-distribution and held-out, generalization gap 0). Stage-A DSM loss drops from a flat
~1.0 to ~0.37. Method: diffusion proposal + energy-descent polish.
*(paper/learned_solver_fix.md)*

## R2 · Entrapment — noise escapes where deterministic descent cannot (RQ2)
On 200 non-convex problems, deterministic energy descent is **100% trapped**; annealed-noise
(Langevin) descent cuts entrapment to **0.475**. Reduction **0.525 ± 0.086** (95% CI excludes
0), N=200, 5 seeds. A pre-registered, falsifiable RQ answered with CIs.
`python -m marc.eval.ablations.noise_ablation --graphs 200`

## R3 · A8.1 — the learned proposal beats classical refinement on non-convex problems
The central mechanism ablation (best-of-8, 60 held-out/family, 95% Wilson CIs). Convex linear
systems saturate every solver at 1.000, so we use non-convex families where deterministic
descent is trapped. **p** = one-sided two-proportion z-test (learned_hybrid > refine+Langevin).

| Family | refine (cold) | refine + Langevin | **learned hybrid (ours)** | p |
|---|---|---|---|---|
| BilinearSystem | 0.000 | 0.300 [0.20,0.43] | **0.550 [0.42,0.67]** | 0.003 |
| BilinearProduct | 0.000 | 0.100 [0.05,0.20] | **0.683 [0.56,0.79]** | <1e-4 |
| QuadraticSystem | 0.000 | 0.300 [0.20,0.43] | **0.683 [0.56,0.79]** | <1e-4 |
| CircleLine | 0.000 | 0.033 [0.01,0.11] | **0.000** | — (fails) |

**Significant win on 3/4 families (p<0.01).** Directly answers "what does the learned denoiser
add over the classical solver?". **Honest failure:** on CircleLine (`x²+y²=r, x+y=s`) the
learned model does not help — the x↔y/sign symmetry makes the root un-inferable from the
constraint constants, and the polish itself is weak there (Langevin 0.033).
`python scripts/run_hard_eval.py`

## R4 · Cross-family generalization (H1 transfer) — partial, honest
Leave-one-family-out: train on 3 non-convex families, test the hybrid on a **held-out** 4th
(best-of-8, 60 test/family, 200 epochs). **p** = z-test, learned(cross) > refine+Langevin.

| Held-out family | trained on | refine+Langevin | **learned (cross)** | p |
|---|---|---|---|---|
| BilinearProduct | Sys, Quad, Circle | 0.100 [0.05,0.20] | **0.683 [0.56,0.79]** | <1e-4 ✓ |
| QuadraticSystem | Sys, Prod, Circle | 0.300 [0.20,0.43] | **0.683 [0.56,0.79]** | <1e-4 ✓ |
| BilinearSystem | Prod, Quad, Circle | 0.300 [0.20,0.43] | **0.000** | fails |
| CircleLine | Sys, Prod, Quad | 0.033 [0.01,0.11] | **0.000** | fails |

**Transfer is partial: strong structural generalization on 2/4 held-out families (0.683,
p<1e-4 — the model solves a family it never trained on), failure on 2/4.** The BilinearSystem
failure is notable because that family is solvable *in-distribution* (R3: 0.55); a smoke run
that trained on only the two *similar* families {Product, Quadratic} recovered it (0.70),
whereas adding the pathological CircleLine family to the training mix collapsed it to 0.00 —
so dissimilar training families can disrupt transfer. Honest read: MARC's learned proposal
transfers across *related* non-convex structure but is not a universal solver, and a bad
training family hurts. `python scripts/run_crossfamily_eval.py`

## R5 · Dimension scaling — amortized inference beats classical + prior
Bundled non-convex traps with per-instance-varying (wide, signed) roots. Learned x0-proposal
beats deterministic (0), Langevin (→0 by n=3), and a mean-prior (0) at every dimension:

| n | deterministic | Langevin | mean-prior | learned |
|---|---|---|---|---|
| 1 | 0.000 | 0.225 | 0.000 | 0.675 |
| 2 | 0.000 | 0.025 | 0.000 | 0.425 |
| 4 | 0.000 | 0.000 | 0.000 | 0.650 |
| 6 | 0.000 | 0.000 | 0.000 | 0.100 |

Langevin decays geometrically (~p^n); the learned proposal beats it throughout. **Honest
caveat:** the learned model also degrades at n=6 (0.10) — decays slower than baselines and
always beats the prior, but is not dimension-immune. Architectural finding: variables must be
conditioned directly on incident constraint constants (message-passing LayerNorm washes the
magnitude out); a direct constant→output skip recovers roots (mean|err| 5.4 → 0.9).
`python scripts/run_dimension_scaling.py`

## R6 · MATH benchmark — coverage / reality check (not a solve claim)
On a 48-problem MATH-500 sample, the template formalizer covers **0/48**; ~20% of the sample is
constraint-shaped (MARC's paradigm), ~31% is CAS computation, ~48% is reasoning/proof (out of
architectural scope). The bottleneck is autoformalization (NL→graph), not the solver. This
quantifies scope honestly. `python scripts/run_math_coverage.py`

---

## Framing rules (house style)
- `refine` is a classical baseline; the learned hybrid is the system.
- Every solve rate has N + a CI or z-test.
- Report CircleLine failure and the n=6 dropoff openly.
- No claim to beat combinatorial-optimization SOTA (DIFUSCO etc.) — different problem class
  (`paper/related_work.md`).
- MARC targets continuous algebraic constraint solving; it does **not** do general MATH/olympiad
  reasoning.
