# AAAI-26 abstract submission (deadline July 27; full paper ~7 days later)

Paste-ready text for the submission form. Source of truth: `paper/tex/marc_aaai.tex`
(the abstract below is the de-TeX'd copy; update BOTH if either changes). The R28
geometry sentence is in (the trap + cross-fitted ceiling clause, second-to-last
sentence), backed by `results/p_geo_repair/` R28b/R28c artifacts.

## Title

Learn the Structure, Not the Values: A Controlled Characterization of Learning in
Exact Constraint Solving

## Abstract

Evaluations of diffusion-based proposal models for constraint solving rarely include one
key control: random multi-start under the same refinement budget. When applied to our
system, this control sharply curtails our headline claim. MARC turns a continuous
algebraic constraint system into a factor graph, over which a graph-neural diffusion
denoiser proposes assignments, descent on an exact computer-algebra energy polishes them,
and an exact symbolic checker certifies solutions. Does the learned proposal improve on
random multi-start at choosing satisfying assignments? Only narrowly, in a predictable
regime. Across trapped low-dimensional families it ties with random restart, but dominates
in high dimension, where random search fails. Once variables couple, the advantage is
gone. Since all methods share the same polishing and checker, best-of-K random
multi-start succeeds with probability exactly 1-(1-q(n))^K, where q(n) is single-start
reachability; a single measured constant, with no free parameters, reproduces the entire
curve (mean absolute error 0.012). The narrow favorable regime is not specific to our
synthetic families: across eight real-world systems, from robotics to algebra, classical
multi-start solved all eight, but none were in the learning-favorable regime. What
classical solvers lack is discrete choice: which structural augmentation renders an
unsolvable system solvable. There, a candidate-conditioned repair ranker outperforms
random on certificate-grade menus (0.997 vs 0.236 balanced nonlinear menu accuracy;
p < 10^-70; 0.982 +/- 0.006 across seeds), and beats a budget-matched per-candidate
probe on accuracy and cost. A closing geometry study prices the trap such claims invite:
failures selected on a single stochastic stream make repairs look decisive; two-stream
selection and a cross-fitted ceiling show the residual probe advantage there is portfolio
diversity, not learnable signal. We delineate the regimes where learned proposals improve
solvers, and show that learning can succeed where no classical algorithm applies.

## Keywords

constraint solving; diffusion models; graph neural networks; neuro-symbolic;
amortized inference; algorithm selection; distance geometry
