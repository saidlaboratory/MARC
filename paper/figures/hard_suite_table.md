# Hard-suite results (non-convex families)

_Best-of-8, 40 held-out problems/family, trained 250 epochs. `refine` variants are classical baselines._

| Family | refine (cold) [baseline] | refine + Langevin [baseline] | **learned hybrid (ours)** |
|---|---|---|---|
| BilinearSystem | 0.000 | 0.350 | **0.625** |
| BilinearProduct | 0.000 | 0.125 | **0.725** |

**Reading:** convex linear systems saturate every solver at 1.000 (no signal); these non-convex bilinear families trap deterministic descent (0.000) and pull solvers off the ceiling. The learned proposal + refine polish beats the best classical method on every family — isolating the denoiser's contribution (A8.1).