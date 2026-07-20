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

## Results
_[filled from `results/overnight/MANIFEST.json` + SUMMARY.md when the run completes; the eval
JSONs under `results/**` carry the trained-checkpoint numbers, and `paper/RESULTS.md` /
`PROVENANCE.md` remain the canonical honest write-ups. Prior-session findings (learned ties
random on coupled systems, etc.) are the baseline these scaled numbers are compared against.]_

## Honest note carried forward
The scaled run tests whether training at D512/L8 changes the prior conclusions (the learned
solver did **not** beat random restart on coupled systems, structure/auxiliary invention hurt
the numeric solver, LLM-verify < LLM-direct). If the scaled model still ties the classical
baselines, that strengthens the honest characterization; it does not, by itself, create a
main-track positive. Report the numbers straight either way.
