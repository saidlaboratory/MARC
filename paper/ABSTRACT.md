# Abstract — MARC (AAAI 2026)

**Title:** When Do Learned Diffusion Proposals Help Constraint Solving?
A Controlled Study on Continuous Algebraic Systems

> Submission abstract (canonical copy lives in `paper/tex/marc.tex`). Numbers verified
> against `paper/RESULTS.md` + `paper/PROVENANCE.md` (Data Version 8). Two acts:
> (1) the value-diffusion characterization + factorization law (boundary of the claim);
> (2) structural repair, where relocating learning to the decision with no classical
> baseline yields a decisive positive. Keep both.
>
> **Length / trimming:** ~265 words. The validation sentence covers both the geometry
> test and the eight real systems; if the form caps tighter, cut the geometry clause
> first (a law nuance) and keep the real-systems clause (the strongest main-track
> signal, answering synthetic-only).

## Primary (v8 — restructured sentence rhythm, same claims and numbers as v7) — 247 words

Evaluations of diffusion-based proposal models for constraint solving rarely include the one control that matters: random multi-start under the same refinement budget. We ran that control on our own system, and it shrank our headline claim. MARC turns a continuous algebraic constraint system into a factor graph. A graph-neural diffusion denoiser proposes assignments, descent on an exact computer-algebra energy polishes them, and an exact symbolic checker decides what counts as solved. Does the learned proposal help decide values, i.e. which assignments satisfy the constraints? Only in a narrow, predictable regime. On trapped low-dimensional families it ties random restart. It wins in high dimension, where random search collapses. Once variables couple, the advantage is gone. Since every method shares the same polish operator and checker, best-of-$K$ random restart succeeds with probability exactly $1-(1-q(n))^K$, where $q(n)$ is single-start reachability; a single measured constant reproduces the entire restart curve with no free parameters (mean absolute error 0.012). The boundary is not an artifact of our synthetic families: across eight real systems from robotics, positioning, optimization, and algebra, classical multi-start solves all eight, and none sits in the learning-favorable regime. What classical solvers cannot do is discrete: choose the structural augmentation that turns an unsolvable system solvable. There, an operator-aware repair ranker beats random (0.997 vs 0.236 on balanced nonlinear menus, $p<10^{-70}$; $0.982\pm0.006$ across seeds) and beats a budget-matched per-candidate probe on accuracy and cost. We map where learned proposals help, and show learning wins where it has no classical baseline.

---

## Alternate (v1 — law-forward, no repair headline; the pre-repair version, kept for reference)

See git history / `paper/tex/marc.tex` prior abstract. Under-features the repair
positive; superseded by the primary above.
