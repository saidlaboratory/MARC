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

## Primary (v6 — v5 claims, lightly rephrased; numbers re-verified) — ~270 words

Diffusion models are increasingly being applied to propose solutions to constraint and
optimization problems, but evaluations typically neglect the control that matters most:
random multi-start under the same refinement budget. We run that control, and our own
headline claim shrinks. MARC encodes a continuous algebraic constraint system as a
factor graph, proposes assignments with a graph-neural diffusion denoiser, polishes
each candidate by descent on an exact computer-algebra energy, and accepts only
assignments verified by an exact symbolic checker. We first ask whether the learned
proposal helps the value decision (which assignments satisfy the constraints). It does,
but only narrowly and predictably: it ties random restart on trapped low-dimensional
families, wins only in high dimension where random search collapses, and loses that
advantage once variables couple. Because every method shares one polish operator and
one checker, best-of-$K$ random restart is exactly $1-(1-q(n))^K$ in the single-start
reachability $q(n)$; one measured constant, the slope of $\log q(n)$, reproduces the
full random-restart curve with no free parameters (mean absolute error 0.012).
Factorized acceptance basins and high dimension are both required, and the boundary
holds beyond synthetic problems: a real geometric domain collapses in reachability and
a trained proposal ties random restart there, while across eight standard real-world
systems (robotics, positioning, optimization, algebra) classical multi-start solves all
eight, none falling in the learning-favorable regime. The decision classical solvers
cannot make is discrete: which structural augmentation turns an unsolvable system
solvable. Moving learning there, an operator-aware repair ranker clears its controls
(0.997 versus 0.236 against random on balanced nonlinear menus, $p<10^{-70}$;
$0.982\pm0.006$ across optimization seeds) and beats a budget-matched per-candidate
solver probe on accuracy and cost together. We report every negative result with
confidence intervals. The contribution is a characterization of where learned proposals
help continuous constraint solving, and a positive result on the decision that has no
classical baseline.

---

## Alternate (v1 — law-forward, no repair headline; the pre-repair version, kept for reference)

See git history / `paper/tex/marc.tex` prior abstract. Under-features the repair
positive; superseded by the primary above.
