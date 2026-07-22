# AAAI-26 abstract submission (deadline July 27; full paper ~7 days later)

Paste-ready text for the submission form. Source of truth: `paper/tex/marc_aaai.tex`
(the abstract below is the de-TeX'd copy; update BOTH if either changes). One slot
is reserved for the R28 geometry construction-repair sentence — add it before
"Every negative result..." once `results/p_geo_repair/` lands, then refresh this file.

## Title

Learn the Structure, Not the Values: A Controlled Characterization of Learning in
Exact Constraint Solving

## Abstract

Learned proposals for constraint solving are usually evaluated against cold-start
refinement; the control that matters is random multi-start at the same budget. We run
that control on our own system and our headline claim shrinks. MARC poses continuous
algebraic systems as factor graphs, proposes values with a graph diffusion model,
polishes them on exact computer-algebra energies, and accepts only what an exact
symbolic checker verifies. Under this protocol, learned value proposals help only
where acceptance basins factorize across variables and dimension defeats random
search (0.975 versus 0.075 at n=3); a parameter-free law in the measured single-start
reachability reproduces the full restart curve (MAE 0.012), predicts the failure
under variable coupling, and holds on eight named real systems, where classical
multi-start leaves no learning-favorable regime. The decision that has no classical
baseline is discrete: which structural augmentation repairs an unsolvable system. A
candidate-conditioned ranker chooses repairs on certificate-grade menus — "exactly
one solvable option" is a computer-algebra theorem, not a budget-relative claim — at
0.997 versus 0.236 random (p < 10^-70, seed-robust), beating budget-matched
per-candidate probes on accuracy and cost together. Every negative result is reported
with confidence intervals; they are the boundary of the claim. Learning earns its
keep on the structural decision; everything else belongs to the solvers.

## Keywords

constraint solving; diffusion models; graph neural networks; neuro-symbolic;
amortized inference; algorithm selection; distance geometry
