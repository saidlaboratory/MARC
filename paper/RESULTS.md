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
**History only — cite R15 (methodology unified-v2, `results/p_scaling/scaling.json`) for
current numbers: learned 0.950/0.950/0.975/0.925/0.250 vs random 1.000/0.725/0.075/0.000/0.000
at n=1/2/3/4/6. The rows below predate the stats unification (bespoke descent + solution-space
acceptance) and are NOT comparable; the paper cites unified-v2.**
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
  (Related Work section of the paper).
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

## R8 · Menu-based structure selection (H2) — **clean regeneration (seed-space v1, `overlap_instances: 0`); honest null vs random selection**
The trained structure policy (menu-based structure selection with predicted defining value —
pick one of K candidate auxiliary structures and predict its defining value) is now regenerated
cleanly under the seed-space v1 protocol (train seeds [0,500), val [500000,500050), test
[900000,…); both output JSONs carry a `seed_hygiene` block with **`overlap_instances: 0`**).
The earlier 0.45/0.53 numbers stay withdrawn (contaminated). Trained on `aux_required`
(offset/coupled), 200 epochs, `shared` held out; K=4 menu, 500 instances/arm, Wilson CIs,
Holm-corrected within the 7-comparison family.

Reference solver = **scipy Levenberg–Marquardt** (matches aux_required's exact solvability
certificate; the `positive_control_ok=True`, gold-oracle solve **1.000**). Solve rate and
invention accuracy coincide here because LM solves any *valid* augmentation and the K−1
distractors are certified unsolvable, so "solve" ≡ "picked the gold structure".

| Split | policy | random-slot | no-context | fixed | gold-oracle | policy vs random |
|---|---|---|---|---|---|---|
| in-pattern (offset/coupled) | **0.410 [0.37,0.45]** | 0.200 | 0.294 | 0.000 | 1.000 | **p_holm<0.001 (sig)** |
| held-out (`shared`, untrained) | **0.234 [0.20,0.27]** | 0.238 | 0.170 | 0.000 | 1.000 | p_holm=1.0 (ties) |

(single-shot sampler, 500 instances/arm, Holm-corrected over the 7-comparison family.)

**Honest conclusion:** **in-pattern, the structure policy significantly beats random selection**
(0.410 vs 0.200, p_holm<0.001) and the no-context and always-fixed controls — the learned
context genuinely helps pick the right augmentation, and its invention rate (0.41) sits above
chance (CI excludes 0.25). **But the advantage does not generalize:** on the held-out `shared`
pattern the policy (0.234) **ties random** (0.238) while still beating no-context (p_holm=0.023)
and fixed. So MARC's structure-selection policy is a real in-distribution win over its controls
that degrades to chance-vs-random on an unseen pattern — reported openly, and now with clean,
citable numbers (`overlap_instances: 0`, positive control passing).

*Data-version note:* R8 was generated under the pre-v8 linear menu protocol (Bernoulli filler
support). Its arms compare against each other on one internally consistent dataset, and its
role here is motivational (v0.2's holdout null is what prompted v0.3); it is not numerically
comparable to the R10 v8 tables.

Reproduce:
`python3 scripts/train_structure_policy.py --data aux_required --epochs 200 --exclude-family shared --out checkpoints/structure_policy.pt`
then `run_invention_eval.py … --data aux_required [--families shared]` (RUNBOOK §7; PROVENANCE R16/R16h).

## R9 · The factorization law — *why* learning helps in R5 and not in R7 (the unifying result)
The central scientific contribution: one falsifiable, parameter-free law that predicts both
the R5 positive and the R7 negative. All methods share one polish operator and one checker;
let `q(n)` = single-start reachability (prob. one random start + polish is accepted). Then
best-of-K random restart is exactly `P_random(n;K) = 1 − (1 − q(n))^K`. The scaling of `q(n)`
is set by whether acceptance basins **factorize across variables**:

- **Separable (independent traps):** the polish is coordinate-decoupled, so `q(n) = v^n` with
  `v := q(1)`. Measured (600 fresh instances/n, Wilson CIs): `log q(n)` is linear with slope
  **−1.03 (R²=0.98)**; `v = 0.27`. Substituting into the best-of-K identity reproduces the
  **entire** measured random-restart curve (0.90, 0.47, 0.16, 0.03, 0.00) with **no free
  parameters, MAE 0.012**. Expected restarts `1/q(n)` explode **3.7 → 13 → 32 → 150 → 600**,
  so any fixed budget is exhausted — random search *must* lose to a flat learned proposal
  (R5: learned ≈ 0.95 across n=1–4). This is the mechanism behind R5.
- **Coupled (chained bilinear):** the polish propagates along the chain, basins do **not**
  factorize, `log q(n)` slope is only **−0.13 (R²=0.96)** — nearly flat. Expected restarts stay
  **2.0 → 4.3**; random never collapses, so the learned proposal has nothing to amortize (R7:
  ties random) and the classical LM solver dominates (0.77 → 1.0). This is the mechanism
  behind R7 — a *predicted* consequence of broken factorization, not an unexplained negative.

- **Geometry (real domain) — reachability collapses but learning does NOT help:** a coupled
  point-chain family (`marc/data/geometry.py` `make_point_chain`, 2k vars, quartic energy).
  Reachability **collapses**: slope **−0.77 (R²=0.999)**, `q(n)` = 0.653, 0.147, 0.027, 0.007.
  We then trained a denoiser here with the identical R5 methodology
  (`scripts/run_pointchain_learned.py`): the learned proposal **exactly ties random restart at
  every chain length and collapses with it** (both 0.625, 0.175, 0.025, 0.000 at n=2,4,6,8; 0/4
  significant wins, p=0.50). Reachability collapse alone was **not** enough — geometry's
  coupling means there is no per-variable marginal for the denoiser to amortize.

**Why this matters (sharper, two-condition law):** collecting all three families, a learned
proposal beats classical search iff **(1)** single-start reachability collapses with dimension
(random fails) **AND (2)** the solution is per-variable separable (the denoiser can amortize it
as marginals). Independent traps have both → learning wins (R5). Coupled bilinear fails (1),
random survives → ties (R7). Geometry satisfies (1) but fails (2) → learning ties random and
collapses with it. The measured slope predicts *random's* collapse; separability predicts
whether *learning* can exploit it. This corrects the earlier "slope alone is the diagnostic"
reading. Full derivation + limitations in `paper/notes/crossover_law.md`; figure
`paper/figures/fig_crossover_theory.pdf`.
`python3 scripts/run_crossover_theory.py --trials 600 --K 8 --seed 20260721` (PROVENANCE
R17–R19); `python3 scripts/run_pointchain_learned.py` (geometry learned arm, PROVENANCE R25).

## R10 · Candidate-conditioned structural repair (v0.3) — **new primary result (Data Version 8)**

The v0.2 slot policy encoded the fixed graph once and classified candidate slots; it
could not see nonlinear operator identity and fell to chance on the held-out `shared`
pattern (R8).  v0.3 instead applies every candidate augmentation, encodes the resulting
polynomial graph with operator-aware factor/edge features, and ranks repairs listwise.
The candidate-only control receives the same augmentation recipe but no problem graph.

**Earlier R10 numbers (Data Versions 6/7) are withdrawn — do not cite 0.565 or 0.889.**
v6 drew gold pins from a narrower prior than distractor pins (candidate-only control
0.343 ≫ 0.25 chance = a measured leak), and a CAS audit showed most v6/v7 nonlinear
"certified unsolvable" distractors actually have real roots: the weak refine probe just
could not find them, and the vieta defining relation (u = x−y+δ) *cannot* produce
rootless corruptions at all (eliminating u leaves a line meeting a hyperbola).  Data
Version 8 (issue #100) fixes generation, certification, and grading together:

- one `REFERENCE_SOLVER` (scipy LM, k=4), owned by `invention_data` and imported by
  certification, the eval arms, the training reward, and e2e grading — identity is
  test-enforced;
- nonlinear distractor unsolvability is an **exact CAS no-real-roots proof** for
  356/360 balanced-test instances (the rest keep the disclosed empirical probe);
  a proven-rootless distractor is unsolvable at *any* budget and seed, closing the
  seed-variance loophole entirely (0/72 certified distractors solvable at grading
  budget in the audit);
- both nonlinear families use one-sided defining templates over per-family (a, δ)
  supports (`u = a·x²+δ`, `u = a·(x²+y²)+δ`, a = ±1) that golds and distractors share;
  anything sympy-equal to the gold is excluded (no zero-shift duplicate golds);
- golds must solve at eval grade under two independent seeds (stable oracle ceiling);
- linear menu fillers use the gold's size-uniform support sampler and the single
  shared pin prior (no support-size or pin-frequency shortcut).

| Evaluation (v8) | full ranker | candidate-only | random |
|---|---:|---:|---:|
| Balanced nonlinear (N=360) | **0.997 [0.984,1.000]** | 0.333 | 0.236 |
| Nonlinear `vieta` (N=180) | **1.000** | 0.311 | 0.211 |
| Nonlinear `quad_link` (N=180) | **0.994** | 0.356 | 0.261 |
| Vieta→unseen `quad_link` (N=150) | **0.420 [0.344,0.500]** | 0.120 | 0.253 |
| Linear `shared` held out (N=400) | **0.380 [0.334,0.428]** | 0.195 | 0.287 |
| Linear all patterns (N=1,200) | **0.339 [0.313,0.366]** | 0.207 | 0.249 |

Paired comparisons: nonlinear full-only correct 239 vs control-only 0 (exact McNemar
p=1.1e-72); linear full>random 304 vs 196 (p=7.8e-07) and full>control 326 vs 167
(p=3.5e-13).  Checkpoint-only replays reproduce both headline evals exactly.

**Honest movement of the numbers:** each closed shortcut lowered the linear headline
(0.565 v6 → 0.445 v7 → 0.380 v8 held-out) while pushing the candidate-only control to
chance — under v8 the *entire* linear signal comes from reading the problem graph, and
it is modest.  The nonlinear results moved the other way (0.889 → 0.997) because v8
menus finally carry theorem-grade "exactly one solvable option" semantics and the
operator-aware encoder is built precisely to read operator/parameter identity from the
augmented graph.  Learning is decisive exactly where operator identity matters; that is
the claim, and the controls now support it cleanly.

End-to-end, after actually applying the repair and invoking the matched solver
(common restart seeds across arms): nonlinear K=4 solves **0.933 = oracle =
enumeration ceiling** with one solver call vs enumeration's 2.62 (N=60); control 0.200,
random 0.250.  Linear K=4 solves 0.300 vs 0.217/0.227, oracle and enumeration 1.000
(N=300), one call vs 2.54.

**Transfer breadth (rotations; `nonlinear_holdout_vieta.json`,
`random_support_holdout_offset.json`, `random_support_holdout_coupled.json`):** the
two original transfer cells generalize.  Reverse nonlinear direction
(quad_link-trained → unseen vieta): 0.393 vs 0.180 random (N=150; forward was 0.420
vs 0.253) — partial transfer is bidirectional.  Linear held-out-pattern rotations:
offset 0.407, coupled 0.450, shared 0.380, each vs ~0.23–0.29 random (N=400 per
cell) — the linear pattern-transfer effect is not specific to one held-out choice.

**Operator-feature ablation (attribution check; `nonlinear_opmask_ablation.json`,
`linear_opmask_ablation.json`):** masking the operator-identity features (factor
degree/has_cross/has_square; edge diag-quadratic/max-exponent/cross) and retraining
leaves the ranker essentially intact — nonlinear 0.981 [0.960,0.991] vs 0.997, linear
0.379 [0.352,0.407] vs 0.339.  The v0.2→v0.3 gain is therefore attributed to
**candidate conditioning** (encoding each candidate-augmented graph), with the
compatibility signal carried by constants/coefficients/incidence jointly with the
candidate — NOT to the operator flags per se.  The earlier "operator-aware encoding is
what distinguishes x+y−3 from x·y−3" attribution is corrected in the paper.

**Cheap-probe control (the "why not just probe?" answer; `probe_nonlinear.json`,
`probe_linear_holdout.json`):** spend a short-budget LM solve on every candidate, pick
first-accept else lowest residual, grade the pick at full budget.  Nonlinear: the
strongest probe (300-step budget) solves **0.881 at 4.73 calls/instance (33.9 ms)**;
the ranker solves **0.939 with one call (3.1 ms)** — the learned component beats
probing on accuracy and cost simultaneously (rootless distractors are unsolvable at
any budget, so the probe's errors come from short-budget misses on the gold, which
graph-reading avoids).  Linear: the probe saturates (0.978–0.989 at ~4.4–4.9 calls)
and plain enumeration is already perfect at 2.49 calls, so the linear rows are a
controlled mechanism result (the ranker reads the problem graph far better than its
controls), not a deployment case — stated plainly.

Cross-budget K scaling (K=4 linear checkpoint, zero-shot, N=300/150/150): full
0.300/0.187/0.113 vs random 0.227/0.120/0.107 — the accuracy advantage shrinks with K
and is gone at K=16, reported as a limitation.  Enumeration calls grow 2.54/4.40/9.05
and measured policy+solve wall-clock stays 3.8–4.9 ms while enumeration grows
4.8→22.5 ms (1.26×→4.65×); at K=16 that is a cost win only, not an accuracy win.
Direct K=16 training performs at chance (val ≈ 0.04–0.06 throughout), so the
cross-budget checkpoint is selected and the direct-training negative remains reported.

Optimization-seed repeats (v8, seeds 11/29/47, shared certified splits, per-seed eval
draws — issue #103 fixed; the prior files reused one random-arm draw across seeds):
**nonlinear full 0.982 ± 0.006** (0.975/0.983/0.989; control 0.319 ± 0.013, random
0.263 ± 0.019) — the nonlinear headline is optimization-robust. **Linear full
0.317 ± 0.069** (0.333/0.392/0.227; random 0.248 ± 0.002) — real seed-to-seed
variance on linear, reported as a limitation alongside the K-scaling one: the linear
ranker's small edge over random is not stable across optimization seeds.
`python3 scripts/run_repair_multiseed.py --data {nonlinear,aux_required} ... --jobs 3`
(`nonlinear_multiseed.json`, `linear_multiseed.json`).

**Conclusion:** the failed value-denoising project now has a positive, controlled learned
component in the correct division of labor: learn which structural repair deserves a
solver call, delegate values, and verify exactly.  Scope remains menu-based repair on
synthetic factor graphs; the "exactly one solvable option" claim is now a CAS theorem
for 99% of nonlinear test menus and an exact rank theorem for all linear menus, with
the remaining 1% carrying the disclosed budget-relative probe certificate.

## R26 · External validity — named real systems (answers the synthetic-only critique)
Eight recognized test problems (geometry circle/conic intersection, GPS-style trilateration,
Rosenbrock & Himmelblau stationary points, 2R and 3R inverse kinematics, the cyclic-4 algebra
benchmark), encoded once as factor graphs; same solver battery, numeric residual acceptance
(`max|r|<1e-6`, since real roots are irrational). `scripts/run_real_systems.py --K 8 --trials 200`.

| arm (best-of-8) | solved / 8 |
|---|---|
| deterministic | 0 |
| Langevin | 1 |
| random restart + gradient polish | 4 |
| **Levenberg–Marquardt** | **8** |

**Honest reading (external validity, not a new positive):** (1) classical LM solves **8/8** — the
paper's central thesis, now on real problems from robotics/positioning/optimization/algebra, not a
constructed family. (2) Where MARC's gradient polish fails (Rosenbrock valley, Himmelblau saddles,
overdetermined trilateration, 6-var 3R chain; single-start q=0), the bottleneck is the **polish**
(LM fixes each), not the proposal — a learned proposal inherits the same weak polish. (3) **No
learning-favorable regime appears**: these systems are low-dim + coupled, so classical search
suffices, exactly as the factorization law (R9) predicts for real coupled systems. Converts
"synthetic-only" into "tested the characterization on eight standard real systems, and it held."
Full writeup: `paper/notes/real_systems.md` (PROVENANCE R26).

## R27 · The amortization crossover replicates, and beats LM too (strengthens R5)
Two attacks on R5 ("learned beats random restart at high-dim separable") are: it was one designed
family, and you never compared to a strong classical solver. R27 answers both. Same experiment
(per-n inline x0 training, one-shot proposal + shared polish, best-of-8, Wilson CIs) on three
structurally different separable families, with a **Levenberg–Marquardt arm** (scipy, analytic
Jacobian, 8 Gaussian multistarts). `scripts/run_crossover_families.py`.

**LM also collapses ~p^n** (it must hit all n independent basins from its multistart), so it is
not a way out: on the baseline family LM = 0.825 / 0.575 / 0.200 / 0.100 / 0.000 at n=1/2/3/4/6,
random = 1.000 / 0.725 / 0.075 / 0.000 / 0.000.

| family | n=4: random | n=4: **LM** | n=4: **learned** | learned > both (sig) at n |
|---|---|---|---|---|
| baseline | 0.000 | 0.100 | **1.000** | 2, 3, 4, 6 |
| wide_roots | 0.000 | 0.000 | **0.225** | 3, 4, 6 |
| double_well | 0.000 | 0.000 | 0.000 | none (learned failed to amortize) |

**Honest conclusion:** the crossover is a **general phenomenon**, not a single construction — on
2/3 families the learned proposal significantly beats **both** random restart and LM at high
dimension (baseline: learned 1.000 vs 0.000/0.000 at n=6). This upgrades R5's claim from "beats
random restart" to "beats every classical baseline including Levenberg–Marquardt, because all
classical methods collapse ~p^n while an amortized proposal that memorizes per-variable marginals
holds." The honest boundary: on double_well the small denoiser did not learn the harder two-well
marginals and collapsed with the classical methods — learning wins only where it can actually
amortize the marginals. `scripts/run_crossover_families.py` (PROVENANCE R27).

## R29 · Oracle-marginal control — true marginals also tie random under coupling (the R9 mechanism, made causal)
The law's explanation of the R7 coupled null is that a value proposal can only amortize
per-variable marginals, which carry no joint information under coupling. As stated that was
correlational — maybe our denoiser was just a bad marginal learner. R29 removes the learner:
sample each coordinate independently from the family's **true** per-variable marginal (pooled
from 200 training-side instances at the same n, seed-disjoint from test, `overlap_instances: 0`
— the same range the learned arm trained on), then the identical best-of-8 polish + checker with
the same restart seeds as the random arm (CRN). Since the family draws each coordinate i.i.d.
from `{-3..3}\{0}`, the pooled marginal is ≈ uniform on the support: this arm is the exact
product-of-marginals proposal, a ceiling for *any* factorized learner. The recomputed random
column reproduces R7 digit-for-digit at every n. Best-of-8, 60 test/n.
`scripts/run_oracle_marginal.py` (`results/p_scaling/oracle_marginal.json`).

| n | random restart | **oracle marginal** | learned (R7) | oracle > random? |
|---|---|---|---|---|
| 2 | 0.483 | **0.733** | 0.233 | **yes** (p=0.0025) |
| 3 | 0.600 | 0.600 | 0.533 | no (exact tie) |
| 4 | 0.517 | 0.483 | 0.517 | no |
| 6 | 0.367 | 0.333 | 0.333 | no |
| 8 | 0.467 | 0.483 | 0.483 | no |

**Honest conclusion:** at n≥3 a *perfect* marginal proposal ties random restart (0/4 significant,
all gaps within noise) — the R7 null is not a training failure but a ceiling: no better marginal
learner could have closed it, so the mechanism (nothing to amortize in the marginals under
coupling) is confirmed causally and architecture-free. The n=2 exception is instructive, not
damaging: with only two coupled coordinates the 8 support draws still land near joint solutions,
so marginal information has bite while the joint grid (6^n points) is tiny — and the learned
model (0.233) sat far *below* its own marginal ceiling there, a separate small-n training miss.
The boundary reads: marginals help at trivially low n and are measurably insufficient from n=3
on, exactly where the law needs them to be.
