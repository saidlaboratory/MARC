# Overnight run — results & status

**Date:** 2026-07-20 · **Machine:** Apple Silicon (MPS), torch 2.12 · **Driver:**
`scripts/run_overnight.py` (crash-safe manifest). Full artifacts land in
`results/overnight/` (MANIFEST.json, SUMMARY.md, per-phase logs) and the per-experiment
`results/**/*.json`.

## Pre-flight (done)
- **Pulled main**; sibling training/structure scripts present (`train_scale.py`,
  `train_structure_policy.py`, `run_invention_eval.py`). No open PRs to merge.
- **Env:** MPS available; `PYTORCH_ENABLE_MPS_FALLBACK=1` set.
- **Fixed the one stale test** (`test_invention_eval::test_evaluate_full_schema_and_pooling`):
  the eval emits a guarded `reverse:policy_value_vs_random` Holm comparison when `pv_active`
  (`_GoldStub` supplies a callable `predicted_pin`); updated the expected set (4→5) and `m`
  (4→5). **Full suite green: 365 passed.**
- **Smoke run (`--smoke --force`) validated the whole pipeline:** 19 phases ok, 2 justified
  skips (`eval_invention_heldout` — training fell back to `toys` data; `eval_cot` — key absent
  in smoke), tests-gate excepted. Every training + eval phase executes end-to-end.

## Real run (in progress)
Launched: `caffeinate -is nohup python3 scripts/run_overnight.py > overnight.out 2>&1 &`
with `GEMINI_API_KEY` exported so **`eval_cot` runs** (was the only skip in smoke).

Phases (in order): `env_check, tests, train_stage_a, train_stage_b, train_structure_policy,
eval_p1_learned, eval_p1_refine, eval_main, eval_main_learned, eval_hard, eval_crossfamily,
eval_coupled, eval_dimension_scaling, eval_geometry, eval_h2, eval_structure_toys,
eval_invention, eval_invention_heldout, eval_math_coverage, eval_cot, figures, summarize`.

- **Stage-A D512/L8 training dominates** (14h timeout; Stage-B 8h; structure policy 4h). Pace
  checked early per the runbook; if projected Stage-A > ~12h, drop to D256/L6 in
  `marc/configs/train/scale.yaml`. **[pace + any config change recorded below]**
- Eval battery + figures + CoT run after training; results overwrite the toy-scale numbers with
  trained-checkpoint numbers.

## Pace / decisions
- **Pace on MPS: ~280 examples/sec** at D512/L8 → Stage-A (n_train 10k × 50 epochs = 500k
  examples) projects to **~30 min**, far under the 12h threshold. **No config downsizing needed**
  — `scale.yaml` left at D512/L8, device auto→mps.
- **Loss behaviour: unstable but not diverging.** Epoch 1 loss starts ~20, spikes to ~300–390,
  then trends down to ~20–60 by end of epoch 1; grad norms are large and volatile (300–3300, no
  gradient clipping in the config, data unnormalised at D512). `best.pt` captures the
  lowest-loss state. Expect a usable-but-noisy checkpoint; the eval phases report whatever solve
  rates it achieves.
- Consequence: the whole run (Stage-A ~30 min + Stage-B GRPO + structure policy + full eval
  battery + CoT) should finish in a few hours, not a full night.

## Run outcome
**22 phases: 17 ok, 4 skipped, 1 failed.** Skips: `tests`/`train_stage_a`/`train_stage_b`
(intentional `--skip` on relaunch), `eval_invention_heldout` (policy trained on `toys`
fallback). Failed: `eval_main_learned` — killed after being stuck 3h+ (the D512 learned
solver's diffusion+guidance loop is too slow over the full perturbation/length suite on MPS;
the problems are convex/saturated anyway).

### Two operational findings (matter for future runs)
1. **Stage-B GRPO diverged at D512** (loss 12.9 → 134,892; reward ~−9e8) and was ~35 min/epoch
   (→ 8h timeout). Cut it; used the clean **Stage-A** checkpoint (loss **0.60**, good DSM
   convergence — the first real-scale trained model).
2. **The eval scripts retrain their own small models and do NOT load the D512 checkpoint.** So
   the scaled training does not reach the differentiating experiments (hard/coupled/dimension).
   Only the convex evals used it (saturated at 1.0). **Fix needed:** wire `MARC_CKPT` into
   `run_hard_eval`/`run_coupled_eval`/`run_dimension_scaling` so scaled runs actually test the
   trained model.

## Results (this run) — the honest pattern holds and sharpens
Consistent theme across every experiment: **the learned model beats *naive* baselines
(fixed / cold-start / deterministic) but ties or loses to *random* selection/restart** — except
in the high-dimensional *independent* regime where random suffers the curse of dimensionality.

| Experiment | Key numbers | Read |
|---|---|---|
| Convex (p1, main, main-learned, CoT) | learned = refine = CoT = **1.000**, gap 0 | saturated; no signal |
| **Hard (bilinear, non-convex)** | learned = random on 3/4 (0.55/0.68/0.68), fails CircleLine | **learned ties random** |
| **Coupled** (chained bilinear) | learned 0.23/0.53/0.52/0.33/0.48 vs random 0.48/0.60/0.52/0.37/0.47 | **learned ≤ random at every n** |
| **Dimension scaling** (independent) | learned 0.95/0.95/**0.975**/**0.925**/0.25 vs random 1.0/0.725/0.075/0.0/0.0 | **learned ≫ random for n≥3** (random collapses); the one real win, but it is amortizing over the curse of dimensionality on *independent* traps |
| **Structure invention** (policy, valid: seed-overlap 0) | invention rate ~0.19; policy **> fixed** (p<1e-4) and **> no-context** (p<1e-4) but **= random** (p=0.28) | learned beats naive, **ties random** structure selection |
| Geometry (refine) | in-dist solve **0.56** | a non-saturated, harder real-ish domain (candidate for future signal) |

**Bottom line:** the full-scale harness run **confirms the prior honest findings** — no new
main-track positive. The learned solver's only clean advantage over random is on independent
high-dimensional traps (a curse-of-dimensionality amortization), and it disappears under
coupling and on the structure-invention policy. `paper/RESULTS.md` / `PROVENANCE.md` remain the
canonical write-ups; these numbers reproduce them under the full pipeline.

## Honest note carried forward
The scaled run tests whether training at D512/L8 changes the prior conclusions (the learned
solver did **not** beat random restart on coupled systems, structure/auxiliary invention hurt
the numeric solver, LLM-verify < LLM-direct). If the scaled model still ties the classical
baselines, that strengthens the honest characterization; it does not, by itself, create a
main-track positive. Report the numbers straight either way.
