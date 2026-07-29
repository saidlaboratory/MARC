# AAAI-26 abstract (submitted Jul 23; full paper due ~Jul 30)

Paste-ready text for the submission form. Source of truth: `paper/tex/main.tex`
(the abstract below is the de-TeX'd copy of the Jul 27 re-cut, which leads with the
structural positive; update BOTH if either changes). The R30 real-systems sentence and
the trap + budget-matched screen clause are backed by `results/p_real_repair/` and
`results/p_geo_repair/` (PROVENANCE R30, R28/R28b/R28c).

## Title

Learn the Structure, Not the Values: A Controlled Characterization of Learning in
Exact Constraint Solving

## Abstract

Learning pays in exact constraint solving at the discrete decision, not the continuous one.
We measure both under a single protocol. MARC encodes a system as a factor graph, proposes
assignments with a graph-neural diffusion denoiser, polishes them on an exact computer-algebra
energy, and certifies with an exact symbolic checker. The decision classical solvers make only
by enumeration is discrete — which structural augmentation renders an unsolvable system
solvable — and there a candidate-conditioned repair ranker matches the exhaustive-enumeration
ceiling at far fewer calls, beating random and a stronger per-candidate probe on accuracy and
cost (0.997 vs 0.236 balanced nonlinear accuracy; p < 10^-70; 0.982 +/- 0.006 across seeds).
The effect is not confined to menus we built: on hardened variants of named real systems, a
construction derived from the givens and selected on held-out failures repairs every remaining
failure of far-side trilateration and of a ghost-root conic intersection, against
0.433 +/- 0.049 and 0.114 +/- 0.011 for restart control matched to the full enumeration
budget. On the value decision the same discipline curtails our own earlier claim: random
multi-start at identical polish and budget — the control such evaluations rarely include —
erases most of the learned proposal's apparent advantage, leaving a predictable regime. It
ties or loses to random restart on trapped low-dimensional families, wins where dimension
collapses random search, and loses once variables couple. All methods share one polish and
checker, so best-of-K random multi-start succeeds with probability exactly 1-(1-q(n))^K in
the single-start reachability q(n); on the separable family one measured constant reproduces
the best-of-8 curve with no free parameters (mean absolute error 0.012), and classical
multi-start solves all eight standard test systems we encode, none of which falls in the
favorable regime. A closing geometry study prices the trap such claims invite: single-stream
failure selection makes repairs look decisive; two-stream selection and a budget-matched
screen show the residual probe advantage is portfolio breadth, not learnable signal. We map
where learning improves on classical search, and where the win is cost over enumeration, not
reach beyond it.

## Keywords

constraint solving; diffusion models; graph neural networks; neuro-symbolic;
amortized inference; algorithm selection; distance geometry

---

## AAAI 150-word form version

Short version for the submission form's 150-word abstract field (the full-length abstract
above is the paper/PDF version; both reflect the Jul 27 re-cut).

Learning pays in exact constraint solving at the discrete decision, not the continuous one. MARC encodes constraint systems as factor graphs, proposes assignments with a graph-neural diffusion denoiser, polishes them on exact computer-algebra energies, and accepts only symbolically verified solutions. At the discrete decision — which structural augmentation renders an unsolvable system solvable — a candidate-conditioned repair ranker matches the exhaustive-enumeration ceiling at far fewer calls (0.997 vs 0.236 random; p < 10^-70, seed-robust), and on hardened variants of named real systems a construction selected on held-out failures repairs every remaining trilateration and conic failure against 0.43 and 0.11 for budget-matched restarts. At the value decision, random multi-start under the same refinement budget — the control such evaluations rarely include — erases most of the learned advantage: it wins only where dimension collapses random search, never once variables couple. One measured constant reproduces the restart curve with no free parameters.
