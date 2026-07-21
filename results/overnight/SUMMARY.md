# Overnight run summary

- started: 2026-07-20T20:31:48+00:00
- commit: `d4c438c8e594eefe7e5220b29c5103c1c596b111`
- device: mps (torch 2.12.0)
- total wall time: 5.09 h
- smoke mode: False

## What to look at first

1. `results/p5_invention/invention.json` — menu-based structure selection (H2); `invention_heldout.json` is the held-out-pattern number.
2. `results/p_coupled/coupled.json` — coupled-family scaling.
3. `results/p_hard/hard_eval.json` — hard-suite headline table.

## Experiments (results JSON touched by this run)

### results/p1_baselines/metrics.json

- solver: refine
- overall_solve_rate: 1.000
- generalization_gap: 0.000
- in_distribution: solve_rate 1.000, pass@k 1.000
- held_out_structure: solve_rate 1.000, pass@k 1.000

### results/p1_baselines/metrics_learned.json

- solver: learned
- overall_solve_rate: 1.000
- generalization_gap: 0.000
- in_distribution: solve_rate 1.000, pass@k 1.000
- held_out_structure: solve_rate 1.000, pass@k 1.000

### results/p2_main/ablation_guidance.json

- ablation: guidance
- status: ok
- checkpoint: /Users/sparsh/Desktop/Research/AAAI/MARC/checkpoints/scale_D512_L8/best.pt
- k: 4
- best_weight: 0.000
- guidance_helps: False

### results/p2_main/ablation_noise.json

- n_graphs: 50
- tol: 0.000
- entrapment_rate_noise_off: 1.000
- entrapment_rate_noise_on_mean: 0.484
- entrapment_rate_noise_on_ci95: 0.086
- entrapment_reduction_mean: 0.516
- entrapment_reduction_ci95: 0.086
- noise_helps: True

### results/p2_main/ablation_purist.json

- ablation: purist_reward
- status: ok
- k: 4
- shaping_gain_overall_solve_rate: 0.000
- shaping_helps: False

### results/p2_main/cot_baseline.json

- solver: cot_baseline
- overall_solve_rate: 1.000
- generalization_gap: 0.000
- model: gemini-flash-lite-latest
- in_distribution: solve_rate 1.000, pass@k 1.000
- held_out_structure: solve_rate 1.000, pass@k 1.000

### results/p2_main/generalization.json

- solver: refine
- overall_solve_rate: 1.000
- generalization_gap: 0.000
- in_distribution: solve_rate 1.000, pass@k 1.000
- held_out_structure: solve_rate 1.000, pass@k 1.000

### results/p2_main/length_extrapolation.json

- suite: length_extrapolation
- solver: refine
- k: 4
- train_length: 2

### results/p2_main/perturbation.json

- suite: perturbation
- solver: refine
- k: 4

### results/p2_main_learned/generalization.json

- solver: learned
- overall_solve_rate: 1.000
- generalization_gap: 0.000
- in_distribution: solve_rate 1.000, pass@k 1.000
- held_out_structure: solve_rate 1.000, pass@k 1.000

### results/p2_main_learned/perturbation.json

- suite: perturbation
- solver: learned
- k: 4

### results/p3_h2/summary.json

- k: 10
- n_instances_per_toy: 25
- overall_fixed_solve_rate: 0.453
- overall_structure_solve_rate: 0.453
- overall_auxiliary_usage_rate: 0.412

### results/p3_structure/structure_toys_baseline.json

- solver: refine
- k: 4
- gold_path: marc/data/structure_toys/gold.json

### results/p4_scale/geometry_eval.json

- solver: refine
- overall_solve_rate: 0.420
- generalization_gap: 0.280
- in_distribution: solve_rate 0.560, pass@k 0.560
- held_out_structure: solve_rate 0.280, pass@k 0.320

### results/p5_invention/invention.json

- status: ok
- positive_control_ok: True

### results/p_coupled/coupled.json

- K: 8
- test_per_n: 60
- epochs: 250

| n | langevin | random | lm | learned | p_learned_gt_random | p_learned_gt_lm |
|---|---|---|---|---|---|---|
| 2 | 0.167 [0.09, 0.28] | 0.483 [0.36, 0.61] | 0.767 [0.65, 0.86] | 0.233 [0.14, 0.35] | 0.998 | 1.000 |
| 3 | 0.100 [0.05, 0.20] | 0.600 [0.47, 0.71] | 0.950 [0.86, 0.98] | 0.533 [0.41, 0.65] | 0.769 | 1.000 |
| 4 | 0.050 [0.02, 0.14] | 0.517 [0.39, 0.64] | 0.983 [0.91, 1.00] | 0.517 [0.39, 0.64] | 0.500 | 1.000 |
| 6 | 0.000 [0.00, 0.06] | 0.367 [0.26, 0.49] | 1.000 [0.94, 1.00] | 0.333 [0.23, 0.46] | 0.649 | 1.000 |
| 8 | 0.000 [0.00, 0.06] | 0.467 [0.35, 0.59] | 1.000 [0.94, 1.00] | 0.483 [0.36, 0.61] | 0.427 | 1.000 |

### results/p_hard/crossfamily.json

- K: 8
- test_per_family: 60
- epochs: 200

| held_out | trained_on | refine_langevin | learned_cross | p_hybrid_gt_langevin |
|---|---|---|---|---|
| BilinearSystem | — | 0.300 [0.20, 0.43] | 0.000 [0.00, 0.06] | 1.000 |
| BilinearProduct | — | 0.100 [0.05, 0.20] | 0.683 [0.56, 0.79] | 0.000 |
| QuadraticSystem | — | 0.300 [0.20, 0.43] | 0.683 [0.56, 0.79] | 0.000 |
| CircleLine | — | 0.033 [0.01, 0.11] | 0.000 [0.00, 0.06] | 0.923 |

### results/p_hard/hard_eval.json

- K: 8
- test_per_family: 60
- epochs: 250
- n_significant: 2

| family | refine_cold | refine_langevin | random_restart | lm | learned_hybrid | hybrid_beats_langevin_sig | p_learned_gt_lm |
|---|---|---|---|---|---|---|---|
| BilinearSystem | 0.000 [0.00, 0.06] | 0.300 [0.20, 0.43] | 0.550 [0.42, 0.67] | 1.000 [0.94, 1.00] | 0.550 [0.42, 0.67] | False | 1.000 |
| BilinearProduct | 0.000 [0.00, 0.06] | 0.100 [0.05, 0.20] | 0.683 [0.56, 0.79] | 1.000 [0.94, 1.00] | 0.683 [0.56, 0.79] | True | 1.000 |
| QuadraticSystem | 0.000 [0.00, 0.06] | 0.300 [0.20, 0.43] | 0.683 [0.56, 0.79] | 1.000 [0.94, 1.00] | 0.683 [0.56, 0.79] | True | 1.000 |
| CircleLine | 0.000 [0.00, 0.06] | 0.033 [0.01, 0.11] | 0.200 [0.12, 0.32] | 1.000 [0.94, 1.00] | 0.000 [0.00, 0.06] | False | 1.000 |

### results/p_math/coverage.json

- n: 48
- parser_coverage: 0.000
- solve_accuracy_on_covered: 0.000
- parsed: 0
- solved: 0

### results/p_scaling/scaling.json

- K: 8
- test_per_n: 40
- epochs: 200

| n | deterministic | langevin | mean_prior | random_restart | learned_x0 | p_learned_gt_random | p_learned_gt_langevin |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 [0.00, 0.09] | 0.775 [0.62, 0.88] | 0.000 [0.00, 0.09] | 1.000 [0.91, 1.00] | 0.950 [0.83, 0.99] | 0.924 | 0.012 |
| 2 | 0.000 [0.00, 0.09] | 0.225 [0.12, 0.38] | 0.000 [0.00, 0.09] | 0.725 [0.57, 0.84] | 0.950 [0.83, 0.99] | 0.003 | 0.000 |
| 3 | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.075 [0.03, 0.20] | 0.975 [0.87, 1.00] | 0.000 | 0.000 |
| 4 | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.925 [0.80, 0.97] | 0.000 | 0.000 |
| 6 | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.000 [0.00, 0.09] | 0.250 [0.14, 0.40] | 0.000 | 0.000 |

## Phase status

| phase | status | wall (s) | reason |
|---|---|---|---|
| env_check | ok | 0.8 |  |
| tests | skipped | 0.0 | --skip |
| train_stage_a | skipped | 0.0 | --skip |
| train_stage_b | skipped | 0.0 | --skip |
| train_structure_policy | ok | 222.4 | first attempt failed (exit code 1 (see results/overnight/logs/train_structure_policy.log)); retry with --data toys succeeded |
| eval_p1_learned | ok | 866.2 |  |
| eval_p1_refine | ok | 3.6 |  |
| eval_main | ok | 3210.2 |  |
| eval_main_learned | failed | 7550.1 | exit code -15 (see results/overnight/logs/eval_main_learned.log) |
| eval_hard | ok | 1349.8 |  |
| eval_crossfamily | ok | 2139.1 |  |
| eval_coupled | ok | 1812.3 |  |
| eval_dimension_scaling | ok | 910.8 |  |
| eval_geometry | ok | 28.7 |  |
| eval_h2 | ok | 17.0 |  |
| eval_structure_toys | ok | 0.6 |  |
| eval_invention | ok | 220.9 |  |
| eval_invention_heldout | skipped | 0.0 | policy trained on fallback 'toys' data — no held-out 'shared' pattern to eval |
| eval_math_coverage | ok | 2.3 |  |
| eval_cot | ok | 3.0 |  |
| figures | ok | 0.9 |  |

## Checkpoint inventory

- `checkpoints/denoiser_stage_a.pt` — 3.6 MB, epoch ?
- `checkpoints/denoiser_stage_b_purist.pt` — 3.6 MB, epoch ?
- `checkpoints/denoiser_stage_b_standard.pt` — 3.6 MB, epoch ?
- `checkpoints/scale_D512_L8/best.pt` — 426.6 MB, epoch 48
- `checkpoints/scale_D512_L8/epoch_0005.pt` — 426.6 MB, epoch 5
- `checkpoints/scale_D512_L8/epoch_0010.pt` — 426.6 MB, epoch 10
- `checkpoints/scale_D512_L8/epoch_0015.pt` — 426.6 MB, epoch 15
- `checkpoints/scale_D512_L8/epoch_0020.pt` — 426.6 MB, epoch 20
- `checkpoints/scale_D512_L8/epoch_0025.pt` — 426.6 MB, epoch 25
- `checkpoints/scale_D512_L8/epoch_0030.pt` — 426.6 MB, epoch 30
- `checkpoints/scale_D512_L8/epoch_0035.pt` — 426.6 MB, epoch 35
- `checkpoints/scale_D512_L8/epoch_0040.pt` — 426.6 MB, epoch 40
- `checkpoints/scale_D512_L8/epoch_0045.pt` — 426.6 MB, epoch 45
- `checkpoints/scale_D512_L8/epoch_0050.pt` — 426.6 MB, epoch 50
- `checkpoints/scale_D512_L8/latest.pt` — 426.6 MB, epoch 50
- `checkpoints/scale_D512_L8/stage_b_epochs/epoch_1.pt` — 319.9 MB, epoch 1
- `checkpoints/scale_D512_L8/stage_b_epochs/epoch_2.pt` — 319.9 MB, epoch 2
- `checkpoints/scale_D512_L8/stage_b_epochs/epoch_3.pt` — 319.9 MB, epoch 3
- `checkpoints/scale_D512_L8/stage_b_epochs/epoch_4.pt` — 319.9 MB, epoch 4
- `checkpoints/scale_D512_L8/stage_b_final.pt` — 0.2 MB, epoch 1
- `checkpoints/stage_a_epochs/epoch_1.pt` — 10.9 MB, epoch 1
- `checkpoints/stage_a_epochs/epoch_10.pt` — 10.9 MB, epoch 10
- `checkpoints/stage_a_epochs/epoch_11.pt` — 10.9 MB, epoch 11
- `checkpoints/stage_a_epochs/epoch_12.pt` — 10.9 MB, epoch 12
- `checkpoints/stage_a_epochs/epoch_13.pt` — 10.9 MB, epoch 13
- `checkpoints/stage_a_epochs/epoch_14.pt` — 10.9 MB, epoch 14
- `checkpoints/stage_a_epochs/epoch_15.pt` — 10.9 MB, epoch 15
- `checkpoints/stage_a_epochs/epoch_16.pt` — 10.9 MB, epoch 16
- `checkpoints/stage_a_epochs/epoch_17.pt` — 10.9 MB, epoch 17
- `checkpoints/stage_a_epochs/epoch_18.pt` — 10.9 MB, epoch 18
- `checkpoints/stage_a_epochs/epoch_19.pt` — 10.9 MB, epoch 19
- `checkpoints/stage_a_epochs/epoch_2.pt` — 10.9 MB, epoch 2
- `checkpoints/stage_a_epochs/epoch_20.pt` — 10.9 MB, epoch 20
- `checkpoints/stage_a_epochs/epoch_21.pt` — 10.9 MB, epoch 21
- `checkpoints/stage_a_epochs/epoch_22.pt` — 10.9 MB, epoch 22
- `checkpoints/stage_a_epochs/epoch_23.pt` — 10.9 MB, epoch 23
- `checkpoints/stage_a_epochs/epoch_24.pt` — 10.9 MB, epoch 24
- `checkpoints/stage_a_epochs/epoch_25.pt` — 10.9 MB, epoch 25
- `checkpoints/stage_a_epochs/epoch_26.pt` — 10.9 MB, epoch 26
- `checkpoints/stage_a_epochs/epoch_27.pt` — 10.9 MB, epoch 27
- `checkpoints/stage_a_epochs/epoch_28.pt` — 10.9 MB, epoch 28
- `checkpoints/stage_a_epochs/epoch_29.pt` — 10.9 MB, epoch 29
- `checkpoints/stage_a_epochs/epoch_3.pt` — 10.9 MB, epoch 3
- `checkpoints/stage_a_epochs/epoch_30.pt` — 10.9 MB, epoch 30
- `checkpoints/stage_a_epochs/epoch_31.pt` — 10.9 MB, epoch 31
- `checkpoints/stage_a_epochs/epoch_32.pt` — 10.9 MB, epoch 32
- `checkpoints/stage_a_epochs/epoch_33.pt` — 10.9 MB, epoch 33
- `checkpoints/stage_a_epochs/epoch_34.pt` — 10.9 MB, epoch 34
- `checkpoints/stage_a_epochs/epoch_35.pt` — 10.9 MB, epoch 35
- `checkpoints/stage_a_epochs/epoch_36.pt` — 10.9 MB, epoch 36
- `checkpoints/stage_a_epochs/epoch_37.pt` — 10.9 MB, epoch 37
- `checkpoints/stage_a_epochs/epoch_38.pt` — 10.9 MB, epoch 38
- `checkpoints/stage_a_epochs/epoch_39.pt` — 10.9 MB, epoch 39
- `checkpoints/stage_a_epochs/epoch_4.pt` — 10.9 MB, epoch 4
- `checkpoints/stage_a_epochs/epoch_40.pt` — 10.9 MB, epoch 40
- `checkpoints/stage_a_epochs/epoch_41.pt` — 10.9 MB, epoch 41
- `checkpoints/stage_a_epochs/epoch_42.pt` — 10.9 MB, epoch 42
- `checkpoints/stage_a_epochs/epoch_43.pt` — 10.9 MB, epoch 43
- `checkpoints/stage_a_epochs/epoch_44.pt` — 10.9 MB, epoch 44
- `checkpoints/stage_a_epochs/epoch_45.pt` — 10.9 MB, epoch 45
- `checkpoints/stage_a_epochs/epoch_46.pt` — 10.9 MB, epoch 46
- `checkpoints/stage_a_epochs/epoch_47.pt` — 10.9 MB, epoch 47
- `checkpoints/stage_a_epochs/epoch_48.pt` — 10.9 MB, epoch 48
- `checkpoints/stage_a_epochs/epoch_49.pt` — 10.9 MB, epoch 49
- `checkpoints/stage_a_epochs/epoch_5.pt` — 10.9 MB, epoch 5
- `checkpoints/stage_a_epochs/epoch_50.pt` — 10.9 MB, epoch 50
- `checkpoints/stage_a_epochs/epoch_51.pt` — 10.9 MB, epoch 51
- `checkpoints/stage_a_epochs/epoch_52.pt` — 10.9 MB, epoch 52
- `checkpoints/stage_a_epochs/epoch_53.pt` — 10.9 MB, epoch 53
- `checkpoints/stage_a_epochs/epoch_54.pt` — 10.9 MB, epoch 54
- `checkpoints/stage_a_epochs/epoch_55.pt` — 10.9 MB, epoch 55
- `checkpoints/stage_a_epochs/epoch_56.pt` — 10.9 MB, epoch 56
- `checkpoints/stage_a_epochs/epoch_57.pt` — 10.9 MB, epoch 57
- `checkpoints/stage_a_epochs/epoch_58.pt` — 10.9 MB, epoch 58
- `checkpoints/stage_a_epochs/epoch_59.pt` — 10.9 MB, epoch 59
- `checkpoints/stage_a_epochs/epoch_6.pt` — 10.9 MB, epoch 6
- `checkpoints/stage_a_epochs/epoch_60.pt` — 10.9 MB, epoch 60
- `checkpoints/stage_a_epochs/epoch_7.pt` — 10.9 MB, epoch 7
- `checkpoints/stage_a_epochs/epoch_8.pt` — 10.9 MB, epoch 8
- `checkpoints/stage_a_epochs/epoch_9.pt` — 10.9 MB, epoch 9
- `checkpoints/stage_b_purist_epochs/epoch_1.pt` — 10.9 MB, epoch 1
- `checkpoints/stage_b_purist_epochs/epoch_2.pt` — 10.9 MB, epoch 2
- `checkpoints/stage_b_purist_epochs/epoch_3.pt` — 10.9 MB, epoch 3
- `checkpoints/stage_b_purist_epochs/epoch_4.pt` — 10.9 MB, epoch 4
- `checkpoints/stage_b_purist_epochs/epoch_5.pt` — 10.9 MB, epoch 5
- `checkpoints/stage_b_standard_epochs/epoch_1.pt` — 10.9 MB, epoch 1
- `checkpoints/stage_b_standard_epochs/epoch_2.pt` — 10.9 MB, epoch 2
- `checkpoints/stage_b_standard_epochs/epoch_3.pt` — 10.9 MB, epoch 3
- `checkpoints/stage_b_standard_epochs/epoch_4.pt` — 10.9 MB, epoch 4
- `checkpoints/stage_b_standard_epochs/epoch_5.pt` — 10.9 MB, epoch 5
- `checkpoints/structure_policy.pt` — 0.6 MB, epoch ?
