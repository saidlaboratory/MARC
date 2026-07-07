# Roadmap — What Worked, What's Deferred (Post-Submission)

Status snapshot across P0–P4, written alongside the P4 demo + geometry-domain eval.
See `TECHNICAL_GUIDE.md` §14 for the design philosophy this roadmap follows
("build the simplest thing that exhibits the phenomenon first").

## What worked

- **P0 infrastructure** — the constraint-graph schema, procedural generator, CAS
  (SymPy residuals/energy/gradient), and two-stage numeric+symbolic checker are
  solid and reused everywhere downstream without modification.
- **P1 value diffusion MVP** — the energy-gradient Langevin `refine()` baseline
  (`marc/refine/iterative.py`) reliably solves the linear-system suites to exact
  checker precision, and the noise-on/off ablation confirms the core hypothesis:
  noise measurably reduces entrapment (`results/p2_main/ablation_noise.json`,
  reduction 0.516 ± 0.086).
- **P2 checker-RL scaffolding** — Stage-A DSM pretraining and Stage-B GRPO
  (standard + purist reward) both run end-to-end on CPU in minutes
  (`scripts/train_p2_checkpoints.py`); the `a1df5a9` checkpoint-loading fix
  unblocked the guidance/purist ablations, which now report real (if toy-scale)
  numbers instead of `status: skipped`.
- **P3 auxiliary-object-usage MVP** — `marc/eval/structure_eval.py`'s
  energy-guided proxy shows the structure model *does* route through invented
  auxiliary quantities when available (39%–100% usage rate depending on the toy),
  a real, measurable, if preliminary, H2 signal (`results/p3_h2/h2_report.md`).
- **P4 domain expansion + demo** — the same factor-graph/CAS/checker machinery
  extends to a new domain (geometry: coordinates as variables, squared-distance
  relations as factors, `marc/data/geometry.py`) with no changes to `marc/graph`,
  `marc/cas`, or `marc/refine`; `scripts/demo_end_to_end.py` runs a full optional-NL
  → graph → solve → checker pipeline on a real geometry instance.

## What's deferred

- **Full GPU-scale training.** The only trained checkpoints in this repo are
  CPU/toy-scale (`scripts/train_p2_checkpoints.py`, D=128, L=4, minutes not
  GPU-hours). The learned solver currently solves ~0% of the P2 suites and cannot
  handle the geometry domain at all (see below) — it has not yet been trained
  at the scale `results/p4_scale/scaling_notes.md`'s D=512/L=8 configs target.
  Running the full Stage-A (50 epochs) + Stage-B (20 epochs) plan on a GPU is the
  single highest-leverage next step; everything else in this roadmap assumes it.
- **Full discrete structure diffusion (D3PM over `StructureHead`).** The H2 eval
  uses an energy-guided best-of-k proxy in place of a trained categorical policy
  (TECHNICAL_GUIDE §10 explicitly scopes this to preliminary results for the
  paper). Training the real D3PM sampler and re-running `structure_eval.py`'s
  exact harness against it is the natural next step — see `h2_report.md`'s
  discussion section.
- **General autoformalization.** `marc/nl/parser.py` recognizes three closed
  sentence templates (linear sum/difference, bilinear sum/product, two-distance
  geometry) — a real parser for those shapes, not a mock, but nowhere near general
  NL→graph translation. CONCEPT.md flags this as "a separate hard problem" by
  design; a learned semantic parser (or LLM-assisted formalization with the
  checker as a verifier) is the natural successor.
- **Geometry as a *training* template, not just an eval split.** `marc/data/
  geometry.py` is wired into `marc/eval/problems.py` for evaluation, but there is
  no `ProblemGenerator`-compatible geometry template yet for generating Stage-A/B
  *training* data (the way `marc/data/templates.py`'s `LinearSystem2x2/3x3Template`
  do). Needed before a learned model can be expected to solve geometry instances
  at all.
- **The formal (Lean) checker gate.** Only the numeric + symbolic (SymPy exact
  rational) gates exist. TECHNICAL_GUIDE §7's Lean 4 proof-kernel gate is
  explicitly [Frontier] and untouched — it requires the autoformalization step
  above to even have something to feed it.
- **Two infrastructure gaps found while building P4**, tracked separately rather
  than folded into this work: a float→rational snapping edge case in
  `Checker._to_exact` (spurious nearby-fraction snaps within ~1e-10 of the
  tolerance boundary), and `marc/eval/runner.py`'s `_evaluate_split` not yet
  handling the `None`-candidate case `LearnedSolver.sample()` can now produce
  (fixed and filtered in `scripts/demo_end_to_end.py`, but the shared eval runner
  used by `run_main_eval.py` doesn't guard against it yet).

## Why the geometry eval solves less than the algebra suites

`results/p4_scale/scaling_notes.md`'s "Geometry-domain eval" section reports
0.56 in-distribution / 0.28 held-out solve rate for the `refine` baseline —
markedly below the P1/P2 linear-system suites' ~1.0. This is a genuine domain
difference, not a bug: squared-distance factors are quadratic in the unknowns
(two circles intersecting), so the energy is a nonconvex quartic instead of the
linear suites' convex quadratic bowl. The default `refine()` hyperparameters
(tuned against the convex case) solve close to 0% of geometry instances; getting
to 0.56/0.28 required turning noise off and using a much longer, gentler polish
(`scripts/demo_end_to_end.py`'s `GEOMETRY_REFINE_KWARGS`) — itself a useful
finding: the noise term that helps `entrapment_suite` (RQ2) actively hurts this
domain's convergence, so "noise on" is not a universal default across domains.
