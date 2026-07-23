#!/usr/bin/env bash
# One-command reproduction for the MARC paper.
#
#   scripts/repro_all.sh verify   # minutes: fast tests + recompute every cited
#                                 # number from the committed JSONs, fail on drift
#   scripts/repro_all.sh rerun    # hours: regenerate the JSONs from scratch, in
#                                 # dependency order, with wall-time estimates
#
# verify is the guard we lacked every time a number moved across data versions;
# rerun is the audit trail for the checkpoints that are gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

mode="${1:-verify}"

case "$mode" in
verify)
  echo "== import every marc module =="
  python3 - <<'PY'
import importlib, pkgutil, marc
n = 0
for m in pkgutil.walk_packages(marc.__path__, "marc."):
    importlib.import_module(m.name); n += 1
print(f"imported {n} marc submodules")
PY
  echo "== fast test suite (~2 min) =="
  python3 -m pytest -q
  echo "== recompute cited numbers from results/*.json, diff against the paper =="
  python3 scripts/verify_paper_numbers.py
  echo "== OK: tests green and every cited number matches its JSON and the tex =="
  ;;

rerun)
  cat <<'PLAN'
Full regeneration in dependency order. Checkpoints (.pt) are gitignored; the
training rows below rebuild them, and the *_paired / --eval-only replays reload
them. Estimates are single-machine CPU wall time (Apple silicon).

  [~2 min]   scripts/verify_paper_numbers.py            # sanity on the committed JSONs first
  [~25 min]  scripts/run_repair_ranker.py --data nonlinear ...        # R10 nonlinear (repair_nonlinear_balanced_full.pt)
  [~35 min]  scripts/run_repair_ranker.py --train-data aux_required --exclude-family shared ...  # R10 linear holdout
  [~40 min]  scripts/run_repair_multiseed.py --data nonlinear --train-seeds 11,29,47 --jobs 3    # R22 multiseed
  [~30 min]  PYTHONPATH=. scripts/run_dimension_scaling.py --seeds 3   # R15 law inputs
  [~90 min]  PYTHONPATH=. scripts/run_crossover_theory.py              # R9 law fit + fig_crossover_theory
  [~2 min]   scripts/run_real_systems.py --K 8 --trials 200            # R26 external suite
  [~2 min]   PYTHONPATH=. scripts/pilot_real_repair.py --n 200 --out results/p_real_repair/real_repair.json  # R30
  [~22 h]    scripts/run_geo_repair.py --opt-seed {11,29,47} ... (v3, n=1250/400/600)  # R28 (dataset cache warmed once)
             scripts/probe_concentration.py --workers 6                # R28 concentration control
             scripts/analyze_geo_repair.py results/p_geo_repair/geo_repair_v3_s{11,29,47}.json --out .../analysis_v3.json

Exact flags for every row are in paper/PROVENANCE.md. Run `verify` after any
subset to confirm the paper still matches.
PLAN
  ;;

*)
  echo "usage: $0 {verify|rerun}" >&2; exit 2 ;;
esac
