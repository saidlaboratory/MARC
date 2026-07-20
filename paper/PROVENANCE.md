# Provenance

**Fixing-plan item:** C3. House rule: no number enters the paper without solver label, N,
CI (where applicable), and an entry here — figure/number → exact command → seed → commit.
Keep this current; the entrapment 0.516-vs-0.525 drift between `main` and PR #50 is exactly
the kind of thing this prevents.

Commits are the short SHA at time of run. Re-record if a number is regenerated.

| # | Result | Value | Command | Seed | Commit |
|---|---|---|---|---|---|
| R1 | Learned solver, in-dist solve rate (convex) | 1.000 | `MARC_CKPT=checkpoints/denoiser_stage_b_standard.pt python scripts/run_p1_eval.py --solver learned --n-id 15 --n-ho 15 --k 4` | default | 4503b7b |
| R2 | Learned solver, held-out solve rate (convex) | 1.000 | same as R1 | default | 4503b7b |
| R3 | Entrapment: deterministic entrapment rate | 1.000 | `python -m marc.eval.ablations.noise_ablation --graphs 200` | seeds 0-4 | 0748a8d |
| R4 | Entrapment: Langevin entrapment rate | 0.475 | same as R3 | seeds 0-4 | 0748a8d |
| R5 | Entrapment reduction (off - on) | 0.525 ± 0.086 (95% CI) | same as R3 | seeds 0-4 | 0748a8d |
| R6 | Dimension scaling (learned, n=1..6) | 0.675 / 0.425 / 0.550 / 0.650 / 0.100 | `python scripts/run_dimension_scaling.py` | train 100+n, test 90000+n | 2d48235 |
| R7 | Dimension scaling (Langevin, n=1..6) | 0.225 / 0.025 / 0 / 0 / 0 | same as R6 | same | 2d48235 |
| R8 | Dimension scaling (mean-prior, all n) | 0.000 | same as R6 | same | 2d48235 |
| R9 | Hard suite (A1): refine cold, 4 families | 0.000 (all; CI [0,0.06]) | `python scripts/run_hard_eval.py` (best-of-8, 60/family) | test seed0 100000 | d65e3db |
| R10 | Hard suite: refine+Langevin, 4 families | 0.300 / 0.100 / 0.300 / 0.033 | same as R9 | same | d65e3db |
| R11 | **A8.1 learned hybrid, 4 families** | **0.550 / 0.683 / 0.683 / 0.000** | same as R9 | same | d65e3db |
| R11b | A8.1 significance (hybrid > langevin, 2-prop z) | p = 0.003 / <1e-4 / <1e-4 / 0.92 (sig on 3/4; CircleLine fails) | `python scripts/plot_hard_eval.py` (post-hoc from R9-R11 counts) | — | d65e3db |
| R12 | CoT baseline (Gemini flash-lite), N=25, k=1 | in-dist 1.000, held-out 1.000 | `GEMINI_API_KEY=… COT_N=25 python -m marc.eval.baselines.cot_baseline` | deterministic problems | 28f9b3b |

## Notes / caveats attached to specific numbers
- **R1/R2/R12 are saturated** (convex suite at ceiling) — not usable for H1 separation. See
  fixing-plan A1/A6; replace with hard-suite numbers (R9–R11) for headline claims.
- **R6** degrades at n=6 (0.100) — report the full curve, not a single point.
- **R12** is thin (N=25, k=1, flash-lite). A6 must scale it (N≥100, k≥4, stronger tier, Wilson CIs)
  before it can carry the H1 comparison.
- Guidance/purist ablations (fixing-plan A4/A5) are **not yet recorded here** because their current
  values are degenerate; re-run on the hard suite before entering them.

## Metric definitions (source of truth)
`marc/eval/metrics.py` — `solve_rate`, `generalization_gap`, `entrapment_rate`,
`perturbation_robustness`, `pass_at_k`. Cite these, not prose descriptions.
