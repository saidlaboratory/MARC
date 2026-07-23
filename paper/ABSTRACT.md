# AAAI-26 abstract submission (deadline July 27; full paper ~7 days later)

Paste-ready text for the submission form. Source of truth: `paper/tex/marc_aaai.tex`
(the abstract below is the de-TeX'd copy; update BOTH if either changes). The R28
geometry sentence is in (the trap + budget-matched screen clause, second-to-last
sentence), backed by `results/p_geo_repair/` R28b/R28c artifacts.

## Title

Learn the Structure, Not the Values: A Controlled Characterization of Learning in
Exact Constraint Solving

## Abstract

Evaluations of diffusion proposal models for constraint solving rarely include the one
decisive control: random multi-start at the same refinement budget. Applied to our own
system, it sharply curtails our headline claim. MARC turns a continuous algebraic constraint
system into a factor graph, over which a graph-neural diffusion denoiser proposes
assignments, descent on an exact computer-algebra energy polishes them, and an exact symbolic
checker certifies solutions. Does the learned proposal beat random multi-start at choosing
values? Only narrowly, in a predictable regime. It ties or loses to random restart on trapped
low-dimensional families, wins where dimension collapses random search, and loses once
variables couple. All methods share one polish and checker, so best-of-K random multi-start
succeeds with probability exactly 1-(1-q(n))^K in the single-start reachability q(n); on the
separable family one measured constant reproduces the best-of-8 curve with no free parameters
(mean absolute error 0.012). Nor is the favorable regime an artifact of our synthetic
families: classical multi-start solves all eight standard test systems we encode, and none
falls in it. What classical solvers make only by enumeration is the discrete choice: which
structural augmentation renders an unsolvable system solvable. There a candidate-conditioned
repair ranker matches the exhaustive-enumeration ceiling at far fewer calls, beating random
and a stronger per-candidate probe on accuracy and cost (0.997 vs 0.236 balanced nonlinear
accuracy; p < 10^-70; 0.982 +/- 0.006 across seeds). A closing geometry study prices the trap
such claims invite: single-stream failure selection makes repairs look decisive; two-stream
selection and a budget-matched screen show the residual probe advantage is portfolio breadth,
not learnable signal. We map where learned proposals improve on classical search, and where
the residual win is cost over an exhaustive classical recourse, not reach beyond it.

## Keywords

constraint solving; diffusion models; graph neural networks; neuro-symbolic;
amortized inference; algorithm selection; distance geometry

---

## AAAI 150-word form version (149 words)

Short version for the submission form's 150-word abstract field (the full-length abstract
above is the paper/PDF version).

Learned diffusion proposals for constraint solving are rarely evaluated against the control that matters: random multi-start under the same refinement budget. Run on our own system, it shrinks our headline claim. MARC encodes algebraic constraint systems as factor graphs, proposes assignments with a graph-neural diffusion denoiser, polishes them on exact computer-algebra energies, and accepts only symbolically verified solutions. The learned proposal helps only narrowly: it ties or loses to random restart on trapped low-dimensional families, wins where dimension collapses random search, and loses once variables couple. One measured constant reproduces the full restart curve with no free parameters (mean absolute error 0.012); across eight standard test systems, classical multi-start solves all eight, none in the learning-favorable regime. The learnable edge is the discrete structural choice classical solvers make only by enumeration: a repair ranker matches the enumeration ceiling at far fewer calls, scoring 0.997 versus 0.236 random ($p<10^{-70}$, seed-robust).
