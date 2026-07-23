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

| R28 | Geometry construction repair, v2 protocol scale (superseded by R28b — kept as the label-noise data point) | ranker 0.238 = restart_control (random 0.19–0.22), McNemar p=0.61; single-stream labels flip ~half on fresh streams | `python3 scripts/run_geo_repair.py --opt-seed {11,29,47} --train-ks 10,12 --transfer-ks 14 --n-train 250 --n-val 80 --n-test 120 --epochs 60` then `scripts/analyze_geo_repair.py` | 2-stream hard-failure population; GEO_REPAIR_VERSION 2; `geo_repair_s{11,29,47}.json`, `analysis.json` | quang/geo-repair |
| R28b | Geometry construction repair, v3 (stream-stable labels, 5x data, recipe head) — the citable negative | ranker separates from random for the first time (0.246 ± 0.016 vs 0.185 ± 0.019; McNemar 35/8, Holm p=1.3e-4; transfer 24/8, p=0.021) but ties recipe_only 0.249/best_fixed 0.259 and loses to restart_control 0.270 (Holm p=1.0); enumeration 0.692@72.7 restarts, restart +16/+32 0.572/0.725; probe 0.698@19.5 (transfer 0.680@19.5 vs enum 0.616); N=367/331 | `python3 scripts/run_geo_repair.py --opt-seed {11,29,47} --train-ks 10,12 --transfer-ks 14 --n-train 1250 --n-val 400 --n-test 600 --epochs 60` then `scripts/analyze_geo_repair.py results/p_geo_repair/geo_repair_v3_s{11,29,47}.json --out results/p_geo_repair/analysis_v3.json` | data seed 20260722; GEO_REPAIR_VERSION 3; label streams 3; `geo_repair_v3_s{11,29,47}.json`, `analysis_v3.json` | quang/geo-repair |
| R28c | Probe-concentration control + probe-outcome distillation — closes the probe-ranker gap | cross-fitted selection ceiling at K_REF: 0.199 [0.161,0.243] trained / 0.169 [0.133,0.213] transfer — BELOW restart_control 0.270; 11% of candidates ever pass a screen (probe = portfolio diversity, not selection); optimistic-vs-crossfit 0.762 vs 0.199 prices the selection-on-noise trap; distillation (3,874 probe-labeled train failures, epochs 15) lands 0.256 ± 0.004 trained / 0.182 ± 0.001 transfer — onto the prior (vs best_fixed W/L 6/6; vs restart_control 8/12, Holm p=1.0), as the ceiling predicts | `python3 scripts/probe_concentration.py --workers 6`; `python3 scripts/run_geo_repair.py --opt-seed {11,29,47} --train-ks 10,12 --transfer-ks 14 --n-train 6250 --n-val 400 --n-test 600 --train-label-restarts 1 --train-label-streams 1 --epochs 15` then `scripts/analyze_geo_repair.py ... --out analysis_p3.json` | screen salts +5/+6, grade salt +7 (protocol streams +0..+4 untouched); `probe_concentration.json`, `geo_repair_p3_s{11,29,47}.json`, `analysis_p3.json` | quang/geo-repair |

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

## Wall-clock receipts (one machine, per-instance)

R28 geometry repair — every arm graded in one pass (`analysis_v3.json`), so the restarts and wall time are directly comparable:

| arm | restarts/instance | wall ms/instance | vs ranker |
|---|---:|---:|---:|
| ranker | 4.0 | 696 | 1.00x |
| recipe_only | 4.0 | 568 | 0.82x |
| restart_control | 4.0 | 548 | 0.79x |
| restart_plus16 | 16.0 | 1346 | 1.93x |
| restart_plus32 | 32.0 | 2125 | 3.05x |
| probe | 19.5 | 4023 | 5.78x |
| enumeration | 72.7 | 10696 | 15.37x |

This prices the arms: a single augmented solve (696 ms) is 5.8x cheaper than the probe and 15x cheaper than enumeration. On R28 that buys nothing — the ranker's pick ties the prior and loses to the matched-budget restart control on accuracy (a negative). The cost win is R10:

R10 nonlinear repair — the learned selection is a single forward pass of 0.27 ms/instance over the candidate graphs (deterministic featurization cached) plus one reference solve, matching the enumeration ceiling at a fraction of its 2.62 solves per instance — cost win and accuracy at once (R10).

Regenerate: `python3 scripts/wall_clock_table.py` (reads committed JSONs).

## Checkpoint manifest (SHA-256)

The `.pt` files are gitignored (regenerable from the training rows above). This
manifest pins the citable checkpoints so the uploaded artifact tarball is verifiable;
regenerate/verify with `scripts/checkpoint_manifest.sh` (`--tar` writes
`dist/marc_checkpoints.tar.gz` for upload per lab convention).

```
# SHA-256 checkpoint manifest
6f96d329886920190af7bb9b884cbee1e15ec34516857e72f4211b41142a24a2  checkpoints/repair_nonlinear_balanced_full.pt
6c6ad84777e96dcef382b9912ce637d0747eea42efbbcb5ae45d0d2f05b7f109  checkpoints/repair_random_support_holdout.pt
7c3dc80710077fdb324c6cab6edb5fe67d9930c951ad6f53abfec70522b325fe  checkpoints/repair_nonlinear_holdout_vieta.pt
c942225e8444e1eed94a7be2b32073826192169e3210b2d250b9d0b072540deb  checkpoints/repair_nonlinear_holdout_quad.pt
6c112aaa72367c764e144f76eb8c48698bfff43ac2f8105c1be96625a30103b7  checkpoints/geo_repair_v3_s11.pt
aa89c837c562413a443bcea1a6962de10c2c59355f119572f23310a7836f3742  checkpoints/geo_repair_v3_s29.pt
5360a97cac3151a2a56a0e14aaad9fedd584a306171435744dcf6256baf8d052  checkpoints/geo_repair_v3_s47.pt
```
