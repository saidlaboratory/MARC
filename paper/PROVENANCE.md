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
| R20 | v0.3 repair ranker, linear `shared` held out | **0.380 [0.334,0.428]** vs candidate-only 0.195 / random 0.287 (N=400); all-pattern 0.339 [0.313,0.366], full>random p=7.8e-07, full>control p=3.5e-13 | `python3 scripts/run_repair_ranker.py --train-data aux_required --exclude-family shared --n-train 1600 --n-val 400 --n-test 1200 --epochs 160 --batch-size 32 --D 128 --L 4 --lr 0.0007 --out results/p_repair/random_support_holdout_shared.json --ckpt checkpoints/repair_random_support_holdout.pt`, then checkpoint-only paired replay | data seed 20260722; computed seed_hygiene overlap 0; **Data Version 8** (v6/v7 withdrawn: pin-prior leak + probe-artifact certificates) | after e81452c |
| R21 | v0.3 repair ranker, balanced nonlinear | **0.997 [0.984,1.000]** vs candidate-only 0.333 / random 0.236 (N=360); paired p=1.1e-72 (239 vs 0); certificates 356 exact CAS / 4 empirical | `python3 scripts/run_repair_ranker.py --train-data nonlinear --n-train 320 --n-val 120 --n-test 360 --epochs 120 --batch-size 16 --D 96 --L 3 --lr 0.0008 --out results/p_repair/nonlinear_balanced_full.json --ckpt checkpoints/repair_nonlinear_balanced_full.pt`, then checkpoint-only paired replay | data seed 20260722; **Data Version 8** one-sided (a, delta) supports, CAS no-real-roots distractor proofs | after e81452c |
| R22 | v0.3 repair ranker, multi-seed robustness (v8) | **nonlinear full 0.982 ± 0.006** (0.975/0.983/0.989); **linear full 0.317 ± 0.069** (0.333/0.392/0.227); random arm now draws independently per seed (sd 0.019/0.002, was the 0.0 artifact) | `python3 scripts/run_repair_multiseed.py --data nonlinear --n-train 320 --n-val 120 --n-test 360 --epochs 100 --batch-size 16 --D 96 --L 3 --lr 0.0008 --train-seeds 11,29,47 --jobs 3` (linear: `--data aux_required --exclude-family shared --n-train 1000 --n-val 300 --n-test 900 --epochs 110 --batch-size 32 --D 128 --L 4`) | data seed 20260722; per-seed eval seed = data_seed+42+1000003·train_seed (issue #103); **Data Version 8** | sparsh/fix-103-104 |
| R23 | v0.3 end-to-end repair + solver | nonlinear **0.933 = oracle = enumeration** vs 0.200/0.250 (N=60, 1 call vs 2.62); linear **0.300** vs 0.217/0.227, oracle=enum=1.000 (N=300, 1 call vs 2.54) | `scripts/run_repair_ranker.py --eval-only --solve-e2e ...`; exact solver linear, REFERENCE_SOLVER (scipy LM k=4) nonlinear; common restart seeds | exact commands/config in `linear_e2e.json`, `nonlinear_e2e_matched.json`; **Data Version 8** | after e81452c |
| R24 | v0.3 cross-budget K scaling | K=4/8/16 full 0.300/0.187/0.113; random 0.227/0.120/0.107; enum calls 2.54/4.40/9.05; wall speedup 1.26x/2.44x/4.65x — accuracy advantage gone at K=16 (cost win only); direct K=16 training at chance | checkpoint-only `--solve-e2e --K {4,8,16}`; `linear_e2e.json`, `linear_K8_e2e.json`, `linear_K16_e2e.json`, `linear_K16_trained.json` | shared data seed protocol; N=300/150/150; **Data Version 8** | after e81452c |
| R25 | **Geometry point-chain LEARNED arm** (the law's live prediction, tested) | learned **ties random at every n** and collapses with it: 0.625/0.175/0.025/0.000 (= random) at n=2/4/6/8, **0/4 significant wins (p=0.50)**; **3 seeds (N=120/k) confirm** (0.625/0.183/0.017/0.008 vs random 0.625/0.200/0.033/0.008) — reachability collapse (slope −0.77) is necessary but NOT sufficient; geometry is coupled, no per-var marginal to amortize | `PYTHONPATH=. python3 scripts/run_pointchain_learned.py --trials 40 --K 8 --epochs 200 --ntrain 200` | per-k inline x0 train (R5 methodology), test seed0 90000+k; `results/p_geometry/pointchain_learned.json` | sparsh/geometry-learned |
| R26 | **External validity: named real systems** (synthetic-only critique) | classical **LM solves 8/8**; gradient-polish random restart 4/8; Langevin 1/8; deterministic 0/8. Systems where random fails (Rosenbrock/Himmelblau/trilateration/3R-IK) are polish-limited (LM fixes them), not a basin collapse — no learning-favorable regime (low-dim + coupled) | `PYTHONPATH=. python3 scripts/run_real_systems.py --K 8 --trials 200` | numeric acceptance max\|r\|<1e-6; 8 named systems (robotics/positioning/optimization/geometry/algebra) | sparsh/real-systems |
| R27 | **Crossover replicates + beats LM** (strengthens R5) | learned beats BOTH random and LM at high-n on **2/3** separable families (baseline: learned 1.000 vs random 0.000 / LM 0.000 at n=6; wide_roots: 0.225 vs 0/0 at n=4). LM also collapses ~p^n (baseline LM 0.825/0.575/0.200/0.100/0.000). double_well: learned failed to amortize (honest limit) | `PYTHONPATH=.:scripts python3 scripts/run_crossover_families.py --K 8 --test 40 --epochs 200 --ntrain 200` | per-n inline x0 train; test seed0 90000+n; `results/p_scaling/crossover_families.json` | sparsh/crossover-families |

## Checkpoint regeneration
`checkpoints/` is gitignored, so any R-row command that loads a `.pt` needs the checkpoint
regenerated first:
- `checkpoints/denoiser_stage_b_standard.pt` (R1/R2): `python scripts/train_p2_checkpoints.py`
  (also writes `denoiser_stage_a.pt` and `denoiser_stage_b_purist.pt`).
- `checkpoints/structure_policy.pt` (R16/R16h): the training command recorded in the R16 row.
- Repair ranker checkpoints (R20–R24): the `--ckpt` paths in each row are written by the
  training command in the same row; the `*_paired.json` replays then load them with
  `--eval-only`.

| R28 | Geometry construction repair (bounded negative) | enumeration ceiling 0.603/0.600 (trained/transfer) vs restart scaling 0.54@+16 / 0.73@+32; ranker 0.22-0.24 = restart_control, 3 seeds, McNemar p=0.61; probe 0.667@~20 restarts is the one budget-beating selector | `python3 scripts/run_geo_repair.py --opt-seed {11,29,47} --train-ks 10,12 --transfer-ks 14 --n-train 250 --n-val 80 --n-test 120 --epochs 60` then `scripts/analyze_geo_repair.py` | 2-stream hard-failure population; GEO_REPAIR_VERSION 2; v3 (stable labels, 5x data) in flight | quang/geo-repair |

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
