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
| R6 | Dimension scaling (learned, n=1,2,3,4,6) | 0.675 / 0.425 / 0.550 / 0.650 / 0.100 | `python scripts/run_dimension_scaling.py` | train 100+n, test 90000+n | 2d48235 |
| R6b | **Dimension scaling (random-restart CONTROL, n=1,2,3,4,6)** | **0.875 / 0.700 / 0.050 / 0.000 / 0.025** (crossover at n=3) | same as R6 (control added to script) | same seeds | 6d70762 |
| R7 | Dimension scaling (Langevin, n=1..6) | 0.225 / 0.025 / 0 / 0 / 0 | same as R6 | same | 2d48235 |
| R8 | Dimension scaling (mean-prior, all n) | 0.000 | same as R6 | same | 2d48235 |
| R9 | Hard suite (A1): refine cold, 4 families | 0.000 (all; CI [0,0.06]) | `python scripts/run_hard_eval.py` (best-of-8, 60/family) | test seed0 100000 | d65e3db |
| R10 | Hard suite: refine+Langevin, 4 families | 0.300 / 0.100 / 0.300 / 0.033 | same as R9 | same | d65e3db |
| R11 | **A8.1 learned hybrid, 4 families** | **0.550 / 0.683 / 0.683 / 0.000** | same as R9 | same | d65e3db |
| R11b | A8.1 significance (hybrid > langevin, 2-prop z) | p = 0.003 / <1e-4 / <1e-4 / 0.92 (sig on 3/4; CircleLine fails) | `python scripts/plot_hard_eval.py` (post-hoc from R9-R11 counts) | — | d65e3db |
| R11c | **A8.1 random-restart CONTROL (4 families)** | **0.550 / 0.717 / 0.683 / 0.200** (ties/beats learned — learned has NO advantage over random multi-start here) | random_count in `scripts/run_hard_eval.py` (control added) | test seed0 100000 | 6d70762 |
| R12 | CoT baseline (Gemini flash-lite), N=25, k=1 | in-dist 1.000, held-out 1.000 | `GEMINI_API_KEY=… COT_N=25 python -m marc.eval.baselines.cot_baseline` | deterministic problems | 28f9b3b |
| R13 | Cross-family (leave-one-out) learned(cross), 4 held-out | 0.683 / 0.683 / 0.000 / 0.000 (Prod/Quad/Sys/Circle); p<1e-4 on 2/4 | `python scripts/run_crossfamily_eval.py` (best-of-8, 60/family) | train seed0 0, test seed0 100000 | df60ebe |
| R14 | MATH coverage (parser), MATH-500 sample | 0/48 = 0.000 | `python scripts/run_math_coverage.py` | fixed sample | 552fdcd |
| R15 | Dimension scaling, unified-v2 methodology | see `results/p_scaling/scaling.json` | `python3 scripts/run_dimension_scaling.py` (unified stats, PR #62) | in JSON | PR #62 |
| R16 | Structure selection (menu-based), in-pattern (single-shot) | policy **0.410 [0.37,0.45]** vs random 0.200 **p_holm<0.001 sig**; gold-oracle 1.000 (pos ctrl OK); `overlap_instances: 0` | `python3 scripts/train_structure_policy.py --data aux_required --epochs 200 --exclude-family shared --out checkpoints/structure_policy.pt` then `run_invention_eval.py --ckpt … --out results/p5_invention/invention.json --data aux_required` (LM reference solver) | seed-space v1 (train [0,500), val [500000,+50), test [900000,…)) | sparsh/fix-invention-families |
| R17 | **Factorization law: single-start reachability slope** (indep / coupled) | **b = −1.032 (R²=0.982) / −0.128 (R²=0.958)** | `PYTHONPATH=. python3 scripts/run_crossover_theory.py --trials 600 --K 8 --seed 20260721` | seed 20260721, 600 fresh inst/n | sparsh/crossover-theory |
| R18 | **Factorization law: parameter-free random-restart prediction MAE (indep)** | **0.012** (v = q(1) = 0.270; measured curve 0.90/0.47/0.16/0.03/0.00) | same as R17 | same | sparsh/crossover-theory |
| R19 | **Factorization law: expected restarts 1/q(n)** (indep / coupled) | **3.7→13→32→150→600 / 2.0→2.5→2.9→3.8→4.3** | same as R17 | same | sparsh/crossover-theory |
| R16h | Structure selection, held-out pattern (`shared` excluded from training) | policy **0.234 [0.20,0.27]** vs random 0.238 **ties (p_holm=1.0)**; vs no-context p_holm=0.023 sig; gold-oracle 1.000; `overlap_instances: 0` | same as R16 with `--families shared --out results/p5_invention/invention_heldout.json` (LM reference solver) | seed-space v1 | sparsh/fix-invention-families |

## Notes / caveats attached to specific numbers
- **Dimension scaling is now cited from R15** (methodology unified-v2, `results/p_scaling/scaling.json`,
  PR #62). The old R6/R6b/R7/R8 rows predate the stats unification and are **not comparable** to
  unified-v2 numbers; they stay in the table as history only.
- **R16/R17 (structure selection):** an earlier preliminary run produced numbers that are
  **withdrawn** — its eval seeds overlapped the validation seeds used for checkpoint selection,
  and it evaluated `toys` data against an `aux_required`-trained policy (source mismatch). Do not
  cite any `results/p5_invention` JSON that lacks a `seed_hygiene` block with
  `overlap_instances: 0`. See `paper/RESULTS.md` R8 and `paper/notes/REVIEW_ATTACKS.md` #3.
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
