# Making the learned diffusion solver converge

## TL;DR
The learned solver previously **diverged** (inference produced values ~1e4; solve
rate 0%). It now **converges to solve rate 1.0** on the in-distribution and
held-out-structure splits (generalization gap 0). Stage-A DSM loss descends
(0.94 → 0.37 over 60 epochs) instead of the old flat ~1.0 plateau.

## Root causes found (in order of impact)

1. **The denoiser never saw the corrupted input `x_t`.** `train_step_A` computed
   `x_t = corrupt(x0, t, eps)` but never assigned it into `graph["variable"].x`,
   so the model trained on the graph's static values. Fixed (also in the scale
   experiment's epoch loop).

2. **The equation constants (RHS `b`) were absent from every graph tensor.**
   `build_heterodata` set `data["factor"].x` to all-zeros, and edges only carry
   coefficients `a`. So `x+y=3` and `x+y=100` were *identical* inputs — the model
   could not learn to produce different solutions and collapsed to predicting the
   mean (loss = variance ≈ 1.0). Fixed: factor node feature = the constraint's
   constant term (value of the expression at x=0).

3. **The timestep `t` only conditioned factor nodes.** Variable nodes and the
   output head had no noise-level signal, so the model could not scale its
   prediction. Fixed: t-embedding added to the variable encoder and the output MLP.
   Factors additionally receive the current residual (analytic for linear systems,
   exact via CAS when a CAS engine is passed).

4. **Inference guidance exploded.** In `solve()` the CAS energy gradient grows
   without bound as `x` drifts, feeding back into an exploding trajectory. Fixed:
   clip the guidance-gradient norm and clamp the state each DDIM step.

5. **Pure diffusion can't reach the strict checker tolerance.** DDIM guidance
   scales by √(1−ᾱ), which vanishes as t→0, so the final steps do no correction.
   Fixed by a **diffusion-proposes / energy-descent-disposes** hybrid: the
   `LearnedSolver` polishes each diffusion candidate with the existing Langevin
   `refine()`, seeded at the diffusion output.

## What the controlled experiments showed
- The GNN can learn **local denoising** (low-noise loss → 0.008) but **cannot learn
  to solve `Ax=b` from scratch** — it fails to overfit even 4 systems from raw
  `(a,b)` features (blind matrix inversion is too hard to learn by regression).
- On **convex** (linear) problems the learned model therefore cannot beat plain
  refinement, which already solves 100%. The guidance and purist-reward ablations
  are correspondingly **saturated at 1.0** on this suite — they will only separate
  on harder / non-convex problem families.
- The paper's real convergence win is the **noise / entrapment** mechanism, which
  is solid and now measured on 200 graphs: deterministic descent is **100%
  trapped**, annealed-noise (Langevin) descent drops entrapment to **0.475**
  (reduction **0.525 ± 0.086**, 95% CI excludes 0).

## Honest caveats
- Checkpoints are toy-scale CPU runs (D=128, L=4, minutes) — enough to unblock the
  eval pipeline end-to-end, not to demonstrate scaling. See
  `results/p4_scale/scaling_notes.md` / `roadmap.md`.
- On these convex suites the learned solver's success is carried by the polish
  step; the diffusion front-end's value is on non-convex problems (entrapment).
  Demonstrating a learned-model win *over* Langevin refine on a non-convex family
  is the natural next experiment.
