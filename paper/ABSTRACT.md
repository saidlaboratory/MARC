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

## Primary (v9 — Sparsh's humanized draft with precision fixes; claims/numbers unchanged) — 241 words

Evaluations of diffusion-based proposal models for constraint solving rarely include one key control: random multi-start under the same refinement budget. When applied to our system, this control sharply curtails our headline claim. MARC turns a continuous algebraic constraint system into a factor graph, over which a graph-neural diffusion denoiser proposes assignments, descent on an exact computer-algebra energy polishes them, and an exact symbolic checker certifies solutions. Does the learned proposal improve on random multi-start at choosing satisfying assignments? Only narrowly, in a predictable regime. Across trapped low-dimensional families it ties with random restart, but dominates in high dimension, where random search fails. Once variables couple, the advantage is gone. Since all methods share the same polishing and checker, best-of-$K$ random multi-start succeeds with probability exactly $1-(1-q(n))^K$, where $q(n)$ is single-start reachability; a single measured constant, with no free parameters, reproduces the entire curve (mean absolute error 0.012). The narrow favorable regime is not specific to our synthetic families: across eight real-world systems, from robotics to algebra, classical multi-start solved all eight, but none were in the learning-favorable regime. What classical solvers lack is discrete choice: which structural augmentation renders an unsolvable system solvable. There, an operator-aware repair ranker outperforms random (0.997 vs 0.236 balanced nonlinear menu accuracy; $p<10^{-70}$; $0.982\pm0.006$ across seeds), and beats a budget-matched per-candidate probe on accuracy and cost. We delineate the regimes where learned proposals improve solvers, and show that learning can succeed where no classical algorithm applies.

---

## Alternate (v1 — law-forward, no repair headline; the pre-repair version, kept for reference)

See git history / `paper/tex/marc.tex` prior abstract. Under-features the repair
positive; superseded by the primary above.
