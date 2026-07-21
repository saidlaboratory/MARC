# MARC — consolidated results (paper-ready, honest)

All numbers carry a solver label and, where applicable, N and 95% CI. Provenance in
`paper/PROVENANCE.md`. Reproduce commands are listed per result. `refine` is always a
classical baseline, never the headline.

## Headline (corrected after adding the random-restart control)
Running the proper **random multi-start + polish** control forced two honest corrections —
and produced a sharper, more defensible story:
1. **The hybrid recipe (any good proposal + energy-descent polish) beats cold-start Langevin**
   on trapped non-convex problems (R3). This is a real, useful result.
2. **The *learned* proposal beats *random* multi-start only in high dimension** (R5 crossover:
   random wins at n≤2, learned wins at n≥4 where random collapses to ~0). This is the genuine
   amortized-inference contribution — learning earns its keep when the search space is too large
   to brute-force. On small-solution families (R3) the learned proposal shows **no** advantage
   over random restart, and we say so.
Net: claim *"amortized learned proposals beat random search in high-dimensional non-convex
constraint solving"*, not *"the learned solver beats classical refinement everywhere."*

---

## R1 · The learned solver converges (was 0%, diverging)
Fixing five bugs (denoiser never saw `x_t`; constants absent from graph tensors; `t` not
conditioning variables; guidance explosion; a 1-var `squeeze` bug) took the learned solver
from *diverging (≈1e4, 0% solve)* to **solve_rate 1.000** on convex linear systems
(in-distribution and held-out, generalization gap 0). Stage-A DSM loss drops from a flat
~1.0 to ~0.37. Method: diffusion proposal + energy-descent polish.
*(paper/notes/learned_solver_fix.md)*

## R2 · Entrapment — noise escapes where deterministic descent cannot (RQ2)
On 200 non-convex problems, deterministic energy descent is **100% trapped**; annealed-noise
(Langevin) descent cuts entrapment to **0.475**. Reduction **0.525 ± 0.086** (95% CI excludes
0), N=200, 5 seeds. A pre-registered, falsifiable RQ answered with CIs.
`python -m marc.eval.ablations.noise_ablation --graphs 200`

## R3 · A8.1 — a good proposal + polish beats cold-start refinement (but the *learned* proposal ≈ random here)
Mechanism ablation (best-of-8, 60 held-out/family, 95% Wilson CIs) on non-convex families where
deterministic descent is trapped. We add the **key control** — random-init + polish, best-of-K,
same budget, no learning — because without it the comparison is against a weaker baseline.

| Family | refine cold | refine+Langevin | **random-init+polish (control)** | learned hybrid | learned>random? |
|---|---|---|---|---|---|
| BilinearSystem | 0.000 | 0.300 | **0.550 [0.42,0.67]** | 0.550 | tie (p=0.50) |
| BilinearProduct | 0.000 | 0.100 | **0.717 [0.59,0.81]** | 0.683 | no (random wins) |
| QuadraticSystem | 0.000 | 0.300 | **0.683 [0.56,0.79]** | 0.683 | tie (p=0.50) |
| CircleLine | 0.000 | 0.033 | **0.200 [0.12,0.32]** | 0.000 | no (random wins) |

**Honest conclusion (this corrects an earlier over-claim):** on these families the learned
proposal provides **no advantage over random multi-start + polish** — it ties on 2 and loses on
2. The apparent "learned beats Langevin" gap is really *"diverse restart + deterministic polish
beats cold-start Langevin"*, i.e. the **hybrid recipe** is the contribution, not the learned
denoiser specifically. Why: solutions here are small integers in [-3,3], which random restart in
[-5,5] hits readily, so there is nothing for learning to amortize. The learned proposal's genuine
value appears only when the solution manifold is wide/structured enough that random restart fails
— that is exactly the R5 regime (±[3,8] roots, where the mean-prior and random both score ~0).
`python scripts/run_hard_eval.py` (+ random-init control, PROVENANCE R11c)

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

## R5 · Dimension scaling — the learned proposal beats random search *only in high dimension* (the real amortization result)
Bundled non-convex traps with per-instance-varying (wide, signed ±[3,8]) roots. Including the
**random multi-start + polish control** (best-of-8, same budget) — the baseline that matters:

| n | deterministic | Langevin | mean-prior | **random restart** | **learned** |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.225 | 0.000 | **0.875** | 0.675 |
| 2 | 0.000 | 0.025 | 0.000 | **0.700** | 0.425 |
| 3 | 0.000 | 0.000 | 0.000 | 0.050 | **0.550** |
| 4 | 0.000 | 0.000 | 0.000 | 0.000 | **0.650** |
| 6 | 0.000 | 0.000 | 0.000 | 0.025 | **0.100** |

**The honest, sharper claim: there is a clean crossover at n=3.** At low dimension (n≤2),
*random restart wins* (0.875 vs 0.675) — the solution space is small enough to brute-force, and
learning buys nothing. At **n≥3 random restart collapses** (0.050 → 0.000; it must hit all n
basins simultaneously by chance, cost ~p^n), while the **learned proposal holds** (0.550 at n=3,
0.650 at n=4). This is the genuine amortized-inference result: the learned model earns its keep
exactly when the search space is too large for random search. Langevin and the mean-prior are ~0
by n≥3.

**Honest caveats:** the learned model still degrades at n=6 (0.10); and it *loses* to random
restart at n≤2. The claim is specifically "amortized proposal > random search in high
dimension," not "learned is best everywhere." Architectural finding: variables must be
conditioned directly on incident constraint constants (message-passing LayerNorm washes the
magnitude out); a direct constant→output skip recovers roots (mean|err| 5.4 → 0.9).
`python scripts/run_dimension_scaling.py` (+ random-restart control, PROVENANCE R6b)

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
  (`paper/notes/related_work.md`).
- MARC targets continuous algebraic constraint solving; it does **not** do general MATH/olympiad
  reasoning.

## R7 · Coupled high-dim families — the learned advantage does NOT survive coupling (honest negative)
The main-track stress test. Coupled chained bilinear (`x_i+x_{i+1}=s_i, x_i·x_{i+1}=p_i`), where
the solution is a *joint* object (not per-variable marginals) and random restart does **not**
collapse in dimension (the chain lets the polish propagate). Best-of-8, 60 test/n.

| n | langevin | random restart | learned | learned > random? |
|---|---|---|---|---|
| 2 | 0.167 | 0.483 | 0.233 | no (loses) |
| 3 | 0.100 | 0.600 | 0.533 | no |
| 4 | 0.050 | 0.517 | 0.517 | no (tie) |
| 6 | 0.000 | 0.367 | 0.333 | no |
| 8 | 0.000 | 0.467 | 0.483 | no |

**Conclusion (decisive, honest):** on coupled systems the learned proposal **ties or loses to
random restart at every dimension (0/5 significant wins).** The R5 high-dim advantage was largely
an **independence artifact** — it needs (i) per-variable-separable solutions the model can
memorize as marginals, and (ii) random restart collapsing because it must hit all n basins by
chance. Neither holds under coupling. So on realistic (coupled) constraint systems, the learned
diffusion model provides **no advantage over random search + refinement**. This closes the
"amortized learned proposal beats classical search" route to a main-track claim. Reported, not
hidden. `python scripts/run_coupled_eval.py`

## R8 · Menu-based structure selection (H2) — **preliminary numbers withdrawn — contaminated eval (test==val seeds); regeneration pending under seed-space v1 protocol**
The trained structure policy (menu-based structure selection with predicted defining value —
the honest name for what the code's "invention" identifiers do: pick one of K candidate
auxiliary structures and predict its defining value) has been trained and evaluated end-to-end,
but the preliminary run's eval seeds overlapped the validation seeds used for checkpoint
selection, **and** the harness evaluated a different data source (`toys`) than training
(`aux_required`). Those numbers are void and are deliberately not recorded here — do not cite
them. Clean numbers regenerate under the seed-space v1 protocol (disjoint seed ranges, a
`seed_hygiene` block with `overlap_instances: 0` in the results JSON) via the fixed overnight
harness (`eval_invention` + the held-out-pattern `eval_invention_heldout`).
`python3 scripts/run_invention_eval.py --ckpt checkpoints/structure_policy.pt --out results/p5_invention/invention.json --data aux_required`
