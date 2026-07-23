#!/usr/bin/env bash
# SHA-256 manifest + tarball of the citable checkpoints. The .pt files are
# gitignored (regenerable from the PROVENANCE training commands), so this closes
# the artifact reviewer's "regenerable is not the same as available" flag: the
# manifest is committed to PROVENANCE, the tarball is uploaded per lab convention.
#
#   scripts/checkpoint_manifest.sh            # print manifest
#   scripts/checkpoint_manifest.sh --tar      # also write dist/marc_checkpoints.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."

# Checkpoints a paper result reloads (--eval-only replays / geo-repair analysis).
CKPTS=(
  checkpoints/repair_nonlinear_balanced_full.pt   # R10/R21 nonlinear headline
  checkpoints/repair_random_support_holdout.pt    # R10/R20 linear holdout
  checkpoints/repair_nonlinear_holdout_vieta.pt    # R10 transfer
  checkpoints/repair_nonlinear_holdout_quad.pt     # R10 transfer (reverse)
  checkpoints/geo_repair_v3_s11.pt                 # R28b
  checkpoints/geo_repair_v3_s29.pt
  checkpoints/geo_repair_v3_s47.pt
)

present=()
for c in "${CKPTS[@]}"; do [ -f "$c" ] && present+=("$c") || echo "WARN missing: $c" >&2; done

echo "# SHA-256 checkpoint manifest"
shasum -a 256 "${present[@]}"

if [ "${1:-}" = "--tar" ]; then
  mkdir -p dist
  tar -czf dist/marc_checkpoints.tar.gz "${present[@]}"
  echo "wrote dist/marc_checkpoints.tar.gz ($(du -h dist/marc_checkpoints.tar.gz | cut -f1))" >&2
fi
