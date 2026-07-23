#!/bin/bash
# #126: 3x3 nonlinear transfer matrix. Train each single family (exclude the other
# two), evaluate on all three; result['per_family'] gives that family's row. The 6
# off-diagonal cells are the transfer numbers. Scale matches nonlinear_holdout_quad.
set -e
cd "$(dirname "$0")/.."
FAMS=(vieta quad_link sq_sum_xy)
for train in "${FAMS[@]}"; do
  ex=()
  for f in "${FAMS[@]}"; do [ "$f" != "$train" ] && ex+=(--exclude-family "$f"); done
  echo "=== training on $train (excluding: ${ex[*]}) ==="
  OMP_NUM_THREADS=8 PYTHONPATH=. python3 scripts/run_repair_ranker.py \
    --train-data nonlinear --eval-data nonlinear "${ex[@]}" \
    --n-train 240 --n-val 100 --n-test 300 --epochs 120 --lr 0.0008 --seed 20260722 \
    --out "results/p_repair/transfer/transfer_train_${train}.json" \
    --ckpt "checkpoints/repair_transfer_${train}.pt"
done
echo "ALL THREE TRANSFER CELLS DONE"
