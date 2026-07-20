# Related work & positioning (draft notes)

Honest positioning for the paper. MARC's novelty is **moderate** — diffusion solvers,
Langevin escape, and diffusion+refinement hybrids all exist. The differentiator is the
**niche**: symbolic-CAS-integrated diffusion for *continuous algebraic constraint systems*,
framed around **entrapment** as a falsifiable research question, plus the amortized-proposal
scaling result. State the prior art plainly and claim only the niche.

## 1. Graph-based diffusion solvers for combinatorial optimization
- **DIFUSCO** (Sun & Yang, NeurIPS 2023) — graph denoising diffusion over discrete {0,1}
  vectors for NP-complete problems (TSP, MIS); SOTA neural CO solver.
  *Difference:* MARC solves **continuous algebraic** constraint systems (real-valued variable
  assignments satisfying symbolic residuals), not discrete combinatorial selection. We use a
  **symbolic CAS** (exact residuals, energy gradient, exact checker) as an oracle; DIFUSCO is
  unsupervised/heuristic with no symbolic verifier.
- Unsupervised neural CO diffusion (arXiv:2406.01661); constraint-aware diffusion for
  trajectory optimization (arXiv:2406.00990); constraint-matrix diffusion for VRP
  (arXiv:2603.07568). All discrete/continuous-control CO; none integrate a symbolic algebra
  checker or study entrapment as a hypothesis.

## 2. Diffusion + iterative refinement hybrids
- **Diffusion-based learning for constrained non-convex optimization with weighted
  bootstrapped refinement** (arXiv:2502.10330) — a diffusion-proposes / refine cousin of our
  hybrid.
  *Difference:* our polish is the **same CAS energy descent** that defines the problem and the
  checker; the ablation (A8.1) isolates the denoiser's marginal contribution over cold-start
  refinement on non-convex algebraic systems, with CIs.

## 3. Langevin dynamics, local-minima escape, amortized inference
- **Regularized Langevin Dynamics for CO** (arXiv:2502.00277) — Langevin avoiding local minima;
  compares SA vs. NN solvers. Directly supports our **entrapment (RQ2)** result: deterministic
  descent is trapped; annealed noise escapes.
- **Amortized Langevin dynamics** (Taniguchi et al., NeurIPS 2022) — replaces per-datapoint MCMC
  with a learned encoder. Our **dimension-scaling** result is an instance: a learned proposal
  amortizes the exploration cost and beats blind Langevin, whose joint-barrier-crossing cost
  grows ~p^n in dimension.
- Simulated-annealing / tempering SGLD; convergence of Langevin-simulated-annealing
  (arXiv:2109.11669) — theoretical grounding for the noise schedule.

## What MARC adds (claim only this)
1. **CAS-in-the-loop diffusion** for algebraic constraint solving: symbolic residuals as node
   features, exact energy-gradient guidance, and an exact symbolic checker as the accept gate —
   a neuro-symbolic combination the CO-diffusion line does not use.
2. **Entrapment as a pre-registered, falsifiable RQ** (falsified if noise does not reduce
   entrapment), answered with 95% CIs — honest science rather than a novelty claim.
3. **A8.1 hybrid ablation** isolating the learned proposal's contribution over cold-start
   refinement on non-convex families (with CIs), directly addressing "what does the denoiser
   add over the classical solver?".
4. **Amortized per-instance inference** that beats both classical Langevin and a mean-prior,
   with the advantage scaling in dimension — plus the architectural finding that variables must
   be conditioned directly on incident constraint constants (message-passing LayerNorm washes
   the magnitude out).

## Framing rule
Never claim to beat DIFUSCO/CO SOTA (different problem class). Position as: *a neuro-symbolic
diffusion solver for continuous algebraic constraint systems, with an entrapment analysis and a
learned-proposal-vs-classical-refinement study.*

_Sources: DIFUSCO (arXiv:2302.08224); unsupervised neural CO diffusion (arXiv:2406.01661);
constraint-aware diffusion (arXiv:2406.00990); bootstrapped-refinement diffusion
(arXiv:2502.10330); Regularized Langevin Dynamics for CO (arXiv:2502.00277); Amortized Langevin
(NeurIPS 2022); Langevin-SA convergence (arXiv:2109.11669). Verify each citation before it
enters the .tex._
