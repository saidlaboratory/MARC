# AAAI-26 abstract submission (deadline July 27; full paper ~7 days later)

Paste-ready text for the submission form. Source of truth: `paper/tex/marc_aaai.tex`
(the abstract below is the de-TeX'd copy; update BOTH if either changes). The R28
geometry sentence is in (the trap + cross-fitted ceiling clause, second-to-last
sentence), backed by `results/p_geo_repair/` R28b/R28c artifacts.

## Title

Learn the Structure, Not the Values: A Controlled Characterization of Learning in
Exact Constraint Solving

## Abstract

Evaluations of diffusion proposal models for constraint solving rarely include the one
decisive control: random multi-start at the same refinement budget. Applied to our own
system, it sharply curtails our headline claim. MARC turns a continuous
algebraic constraint system into a factor graph, over which a graph-neural diffusion
denoiser proposes assignments, descent on an exact computer-algebra energy polishes them,
and an exact symbolic checker certifies solutions. Does the learned proposal beat random
multi-start at choosing values? Only narrowly, in a predictable regime. It ties random restart on trapped low-dimensional families, dominates in high dimension
where random search fails, and loses the advantage once variables couple. Since all
methods share one polish and checker, best-of-K random
multi-start succeeds with probability exactly 1-(1-q(n))^K, where q(n) is single-start
reachability; one measured constant reproduces the entire curve with no free parameters
(mean absolute error 0.012). Nor is the regime an artifact of our synthetic families: classical multi-start solves
all eight real systems we encode, and none falls in the learning-favorable regime. What
classical solvers lack is discrete choice: which structural augmentation renders an
unsolvable system solvable. There a candidate-conditioned repair ranker beats random
on certificate-grade menus (0.997 vs 0.236 balanced nonlinear accuracy;
p < 10^-70; 0.982 +/- 0.006 across seeds) and a budget-matched per-candidate probe
on accuracy and cost. A closing geometry study prices the trap such claims invite:
single-stream failure selection makes repairs look decisive; two-stream selection and a
cross-fitted ceiling show the residual probe advantage is portfolio diversity, not
learnable signal. We delineate where learned proposals improve solvers, and
show learning succeeding where no classical algorithm applies.

## Keywords

constraint solving; diffusion models; graph neural networks; neuro-symbolic;
amortized inference; algorithm selection; distance geometry
