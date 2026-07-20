# Hard-suite results (non-convex families)

_Best-of-8, 60 held-out problems/family, trained 250 epochs. 95% Wilson CIs in brackets. `refine` variants are classical baselines. **p** = one-sided two-proportion z-test, learned_hybrid > refine+Langevin._

| Family | refine (cold) [baseline] | refine + Langevin [baseline] | **learned hybrid (ours)** | p (hybrid>langevin) |
|---|---|---|---|---|
| BilinearSystem | 0.000 [0.00, 0.06] | 0.300 [0.20, 0.43] | **0.550 [0.42, 0.67]** | **0.0028** |
| BilinearProduct | 0.000 [0.00, 0.06] | 0.100 [0.05, 0.20] | **0.683 [0.56, 0.79]** | **0.0000** |
| QuadraticSystem | 0.000 [0.00, 0.06] | 0.300 [0.20, 0.43] | **0.683 [0.56, 0.79]** | **0.0000** |
| CircleLine | 0.000 [0.00, 0.06] | 0.033 [0.01, 0.11] | **0.000 [0.00, 0.06]** | 0.923 |

**learned_hybrid significantly beats refine+Langevin (p<0.05) on 3/4 families.**

**Reading:** convex linear systems saturate every solver at 1.000 (no signal); these non-convex families trap deterministic descent (0.000) and pull solvers off the ceiling. The learned proposal + refine polish beats the best classical method on 3/4 families (p<0.01), isolating the denoiser's contribution (A8.1). **Honest failure case:** on CircleLine (x^2+y^2=r, x+y=s) the learned model does not help (0.000) — the x<->y and sign symmetry makes the root un-inferable from the constraint constants, and the polish is itself weak there (Langevin 0.033). Reported, not hidden.