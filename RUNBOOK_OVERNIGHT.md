# Overnight run — runbook

Everything below assumes a **MacBook (Apple Silicon M5, 16 GB unified memory,
~100 GB free storage)** running macOS, with the repo cloned and you in the repo
root. The whole night is driven by one script, `scripts/run_overnight.py`. It
never crashes on a missing script or a failed experiment — it records the
outcome in `results/overnight/MANIFEST.json` and moves on. Your job is: set up,
sanity-check, launch, go to sleep, tar the results in the morning.

Training runs on the Apple GPU via PyTorch **MPS** (the trainer's
`device: auto` resolves cuda > mps > cpu, so it picks `mps` automatically).
Mixed precision is CUDA-only in this repo — MPS runs fp32; that's expected, no
config change needed.

## 1. Setup (~10 min)

Python 3.10+ required (`python3 --version`; macOS system Python or
`brew install python@3.12` both work).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a `requirements-lock.txt` exists in the repo root, prefer
`pip install -r requirements-lock.txt` for exact pins.

On Apple Silicon the default PyPI wheels are the right ones — **no CUDA index
URLs, no `pyg_lib`/`torch_scatter` extras** (those are Linux/CUDA builds; plain
`pip install torch torch_geometric` is all this repo needs).

Verify MPS is available:

```bash
python3 -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expect `True`. If `False`, stop and check your torch install — the whole night
would silently run on CPU.

One env var for the real run: a few torch-geometric scatter ops may lack MPS
kernels depending on the torch version. Exporting

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

makes torch run just those ops on CPU instead of erroring out. Harmless if
unneeded; put it in the same shell you launch from.

**Storage:** 100 GB free is far more than needed — the generated dataset,
checkpoints, and results together are a few GB at most.

## 2. Sanity (~5 min)

```bash
python3 -m pytest -q
```

Expect all green (~414 tests, a couple of minutes). **If anything is red, don't
launch the overnight run — ping the team instead.**

```bash
python3 scripts/run_overnight.py --smoke
```

This threads tiny arguments through every phase (the test suite still runs in
full) and should finish well under 15 minutes. Afterwards check:

- `results/overnight/MANIFEST.json` — every phase `"ok"` or `"skipped"`, none
  `"failed"`.
- `results/overnight/SUMMARY.md` exists.
- `results/overnight/logs/` has one log per executed phase.

Phases skipped with reason `"script not present — sibling PR not merged"` are
**normal**: the trainer (`scripts/train_scale.py`) and the structure-selection
scripts (`scripts/train_structure_policy.py`, `scripts/run_invention_eval.py` —
menu-based structure selection, "invention" only in the code identifiers)
live in sibling PRs. **Before the real run, merge all open PRs** so those
phases actually execute — the run is still valid without them, but the training
phases are the whole point of the night.

Phase list (the `"phase"` fields in `MANIFEST.json`): `env_check, tests,
train_stage_a, train_stage_b, train_structure_policy, eval_p1_learned,
eval_p1_refine, eval_main, eval_main_learned, eval_hard, eval_crossfamily,
eval_coupled, eval_dimension_scaling, eval_geometry, eval_h2,
eval_structure_toys, eval_invention, eval_invention_heldout,
eval_math_coverage, eval_cot, figures, summarize`. Notes on the newer ones:

- `eval_main_learned` — the main table with `--solver learned` (separate output
  dir `results/p2_main_learned/`); `eval_main` stays the classical `refine` row.
  Skipped if no denoiser checkpoint exists.
- `eval_invention` — evals the structure policy on the SAME data source the
  training phase actually used (the harness tracks whether the `aux_required`
  → `toys` fallback fired, so train/eval never mismatch).
- `eval_invention_heldout` — evals ONLY the `shared` pattern, which training
  excludes by default (`--exclude-family shared`): the cross-pattern
  generalization number, written to `results/p5_invention/invention_heldout.json`.
  Skipped with a reason if training fell back to `toys` or the eval script
  doesn't support `--families` yet.

Running a script standalone (outside the harness)? Prefix `PYTHONPATH=.` —
`python3 scripts/foo.py` puts `scripts/` (not the repo root) on `sys.path`, and
not every script self-fixes `import marc`:

```bash
PYTHONPATH=. python3 scripts/plot_hard_eval.py
```

**Red flag — invalid invention numbers:** `results/p5_invention` numbers from
runs before the seed-protocol fix are **invalid** (eval seeds overlapped
validation seeds, and eval data could mismatch training data). Only cite runs
whose JSON has a `seed_hygiene` block with `overlap_instances: 0`.

`eval_cot` is skipped unless `GEMINI_API_KEY` or `OPENAI_API_KEY` is exported —
export one before the real run if you have it, otherwise let it skip.

## 3. The real run

**Keep the MacBook plugged in and leave the lid open** (the display can turn
off). Closing the lid sleeps the machine regardless of any tool, and that
pauses the run. `caffeinate` prevents idle/system sleep for exactly as long as
the run lives:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
caffeinate -is nohup python3 scripts/run_overnight.py > overnight.out 2>&1 &
```

Close every heavy app first (browsers, Slack, Docker, Xcode) — 16 GB is
unified memory shared between CPU, GPU, and the rest of macOS. The model
itself is small (~27 M params at D512/L8); memory pressure will come from
other apps, not from MARC.

Expected shape of the night — honest caveats: nobody has timed the training
scripts end-to-end on Apple Silicon, MPS is substantially slower than a
datacenter GPU for training, and a large share of this pipeline is CPU-bound
sympy work regardless of device. Rough expectations:

- Data generation and the eval-only phases: minutes each.
- **Stage-A training at D512/L8 dominates the night** — the harness gives it a
  14 h timeout, Stage-B 8 h, the structure policy 4 h; a phase that hits its
  timeout is marked `failed` and the run continues with the best checkpoint
  written so far.
- **Check the pace early** (~15 min in): look at `examples_per_sec` in the
  training log below. If projected Stage-A time blows past ~12 h, don't burn
  the night — stop the run, drop `model.D: 512 → 256` and `model.L: 8 → 6` (or
  halve `training.epochs_A`) in `marc/configs/train/scale.yaml`, and relaunch.
  A finished D256 run beats a timed-out D512 one. Note what you changed.
- The eval battery afterwards: tens of minutes to a couple of hours total.

Monitoring (macOS has no `nvidia-smi`; `watch` needs `brew install watch`, so
plain loops below):

```bash
tail -f overnight.out                                  # harness + phase output
tail -f checkpoints/scale_D512_L8/train_log.jsonl      # loss + examples_per_sec, once training starts
sudo powermetrics --samplers gpu_power -i 5000         # GPU utilisation (Ctrl-C to stop)
```

Activity Monitor (Window → GPU History) works too. If GPU sits at ~0% during a
training phase, check `results/overnight/logs/train_stage_a.log` — but note
sympy-heavy phases (data generation, Stage-B rollouts, most evals) are
legitimately CPU-bound; ~0% GPU there is normal.

## 4. If it crashes (or the Mac sleeps/reboots)

Just rerun the same command:

```bash
caffeinate -is nohup python3 scripts/run_overnight.py > overnight.out 2>&1 &
```

Training resumes from `checkpoints/scale_D512_L8/latest.pt` and data
generation is cached, so you lose little. Useful flags:

- `--skip PHASE[,PHASE]` — skip phases (e.g. `--skip tests` to skip the test
  gate, or skip a training phase that already finished).
- `--only PHASE[,PHASE]` — run only the listed phases (e.g.
  `--only eval_hard,figures,summarize` to redo one eval and the summary).
- `--force` — continue even if the test suite is red (default: a red suite
  aborts everything after it).

Phase names are the `"phase"` fields in `MANIFEST.json`. Every phase's full
stdout/stderr is in `results/overnight/logs/<phase>.log` — read that before
anything else when a phase says `failed`.

## 5. What to send back

Everything below (one tar, from the repo root):

```bash
tar czf overnight_$(date +%Y%m%d).tar.gz \
    results/ \
    overnight.out \
    checkpoints/scale_D512_L8/latest.pt \
    checkpoints/scale_D512_L8/best.pt \
    checkpoints/scale_D512_L8/train_log.jsonl \
    checkpoints/structure_policy.pt \
    --ignore-failed-read
```

(`--ignore-failed-read` tolerates checkpoints that don't exist because a
training phase was skipped.) That covers the whole `results/overnight/` dir
(MANIFEST, SUMMARY, logs, env), every `results/**/*.json` the run touched, and
the checkpoints + training log. If you changed `scale.yaml` (the pace fallback
above), say so when you send it — the config is also embedded in every
checkpoint, but say it anyway.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `NotImplementedError: ... not implemented for MPS` during training | `export PYTORCH_ENABLE_MPS_FALLBACK=1` and rerun. If it persists, force CPU: edit `training.device: "cpu"` in `marc/configs/train/scale.yaml` — slower but correct. |
| macOS memory pressure yellow/red, machine crawling | Close other apps; if it persists, lower `training.batch_size` in `marc/configs/train/scale.yaml` and rerun (resumes from `latest.pt`). |
| Stage-A projected to blow the night (check `examples_per_sec` early) | Drop to D256/L6 or halve `epochs_A` in `scale.yaml` and relaunch — see §3. |
| Run paused overnight / laptop slept | The lid was closed or `caffeinate` wasn't used. Relaunch with the §3 command; everything resumes. |
| `mps available: False` in setup | Reinstall torch from PyPI inside the venv (`pip install --force-reinstall torch`); make sure you're on the arm64 Python, not an x86 one under Rosetta (`python3 -c "import platform; print(platform.machine())"` → `arm64`). |
| `pytest` red on arrival | Don't run anything — ping the team. |
| A phase says `skipped: script not present` | The sibling PR with that script isn't merged. Merge open PRs and rerun with `--only <phase>` (plus `figures,summarize`). |
| `eval_cot` skipped: no API key | Export `GEMINI_API_KEY` (or `OPENAI_API_KEY`) and rerun `--only eval_cot,summarize`, or ignore — it's a baseline, not a blocker. |

## 7. Regenerating R8 (structure-selection numbers)

The old R8 numbers are withdrawn (seed overlap — see the red flag in §2). To
regenerate them cleanly, three commands from the repo root:

**Step 1 — retrain the structure policy.** Not optional: only a freshly trained
checkpoint records `seed_space_version: 1` provenance in its `train_config`
(`scripts/train_structure_policy.py` writes it at save time). Reusing an old
checkpoint drops the eval's seed-hygiene check into the weaker
"reconstructed"/legacy path, and those numbers stay uncitable.

```bash
PYTHONPATH=. python3 scripts/train_structure_policy.py \
    --data aux_required --epochs 200 --device auto \
    --exclude-family shared --out checkpoints/structure_policy.pt
```

**Step 2 — main eval** (same data source as training):

```bash
PYTHONPATH=. python3 scripts/run_invention_eval.py \
    --ckpt checkpoints/structure_policy.pt \
    --out results/p5_invention/invention.json --data aux_required
```

**Step 3 — held-out pattern** (the `shared` family training excluded):

```bash
PYTHONPATH=. python3 scripts/run_invention_eval.py \
    --ckpt checkpoints/structure_policy.pt \
    --out results/p5_invention/invention_heldout.json \
    --data aux_required --families shared
```

Then check each output JSON for the `seed_hygiene` block with
`"overlap_instances": 0`. **Nothing gets cited until that block is present.**

## R28: geometry construction-repair seeds (branch quang/geo-repair, PR #118)

Seed 11 is running on Quang's machine. Seeds 29 and 47 are yours; each run
regenerates the identical dataset deterministically (that is the slow part,
~1.5-2h solo per run on CPU) and then trains + evaluates (~1h). Run them one
at a time if the box is busy, both in parallel if not:

    OMP_NUM_THREADS=6 python3 scripts/run_geo_repair.py --opt-seed 29 \
        --out results/p_geo_repair/geo_repair_s29.json --ckpt checkpoints/geo_repair_s29.pt
    OMP_NUM_THREADS=6 python3 scripts/run_geo_repair.py --opt-seed 47 \
        --out results/p_geo_repair/geo_repair_s47.json --ckpt checkpoints/geo_repair_s47.pt

If Quang's seed-11 run dies, the same command with `--opt-seed 11` and the
matching out/ckpt paths reproduces it exactly (data seed is fixed; opt-seed
only moves torch init, shuffle, and the random arm).

When any subset of the three JSONs exists:

    python3 scripts/analyze_geo_repair.py

aggregates whatever landed (multiseed mean/SD per arm per pool, Holm over the
six ranker-vs-baseline McNemars, label-agreement stats) into
results/p_geo_repair/analysis.json plus paste-ready RESULTS/tex blocks. The
result JSONs and analysis.json are gitignore-whitelisted — commit them to the
branch as they land. The paper stub they fill is in marc_aaai.tex (search
TODO(R28)); framing rules for a mixed outcome are in
paper/notes/REVIEW_ATTACKS.md ("If R28 comes back mixed").
