# Abstract — MARC (AAAI 2026)

**Title:** When Do Learned Diffusion Proposals Help Constraint Solving?
A Controlled Study on Continuous Algebraic Systems

> Submission abstract. Numbers verified against `paper/RESULTS.md` + `paper/PROVENANCE.md`
> (Data Version 8). Two acts: (1) the value-diffusion characterization + factorization
> law (boundary of the claim); (2) structural repair, where relocating learning to the
> decision with no classical baseline yields a decisive positive. Keep both.

## Primary (v2 — repair co-headlined, em-dash-free) — ~250 words

Diffusion models are increasingly used to propose solutions for constraint and
optimization problems, but evaluations usually omit the control that matters most:
random multi-start under the same refinement budget. We run that control, and our own
headline claim shrinks. MARC represents a continuous algebraic constraint system as a
factor graph, proposes assignments with a graph-neural diffusion denoiser, polishes
each by descent on an exact computer-algebra energy, and accepts only assignments an
exact symbolic checker verifies. We first ask whether the learned proposal helps the
value decision (which numbers satisfy the equations). It does, but only in a narrow and
predictable regime: it ties random restart on trapped low-dimensional families, wins
only in high dimension where random search collapses, and loses that advantage once
variables couple. Because every method shares one polish operator and one checker,
best-of-$K$ random restart is exactly $1-(1-q(n))^K$ in the single-start reachability
$q(n)$, and the measured slope of $\log q(n)$ decides the regime; one measured constant
reproduces the whole random-restart curve with no free parameters (mean absolute error
0.012). The decision classical solvers cannot make is discrete: which structural
augmentation turns an unsolvable system solvable. Moving learning there, an
operator-aware repair ranker clears its controls decisively (0.997 versus 0.236 on
balanced nonlinear menus, $p<10^{-70}$; $0.982\pm0.006$ across optimization seeds) and
beats a cheap per-candidate solver probe on accuracy and cost together. We report every
negative with confidence intervals. The contribution is a characterization of where
learned proposals help continuous constraint solving, and a positive result once
learning moves to the decision that has no classical baseline.

---

## Alternate (v1 — law-forward, no repair headline; the pre-repair version, kept for reference)

See git history / `paper/tex/marc.tex` prior abstract. Under-features the repair
positive; superseded by the primary above.
