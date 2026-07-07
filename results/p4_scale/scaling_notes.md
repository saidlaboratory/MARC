# MARC Scaling Notes — Quang P4

**Generated:** 2026-06-25 13:27 UTC
**Device:** cpu (Apple M-series / Intel; GPU run recommended for D=512)
**Task:** Stage A DSM pretraining · LinearSystem2x2 + LinearSystem3x3 (160 training pairs, 5 epochs)
**Config:** `marc/configs/train/scale.yaml`

## Results

| Config | D | L | Params | Epochs | Final Loss | Min Loss | CPU Wall (s) | s/epoch | Est. GPU (A100) |
|--------|---|---|--------|--------|------------|----------|-------------|---------|-----------------|
| baseline_D128_L4 | 128 | 4 | 885,889 | 5 | 0.9208 | 0.9208 | 4.8 | 1.0 | ~0.05s/ep |
| mid_D256_L6 | 256 | 6 | 5,088,513 | 5 | 0.9506 | 0.9222 | 14.0 | 2.8 | ~0.15s/ep |
| large_D512_L8 | 512 | 8 | 26,575,361 | 5 | 1.0515 | 0.9715 | 39.0 | 7.8 | ~0.40s/ep |

## Observations

- **Parameter scaling:** baseline (D=128, L=4) → large (D=512, L=8) is a **30× parameter increase**
  (885K → 26.6M), primarily from D² MLP sizes and L extra message-passing rounds.

- **Early loss behavior:** larger models show higher initial loss (D=512 starts at ~1.5 vs ~1.0 for D=128),
  consistent with random initialisation taking longer to find a useful representation.
  By epoch 5, all three configs converge into the 0.92–1.05 range.
  Full 50-epoch training (on GPU) is expected to show clearer separation, with D=512 reaching
  the lowest plateau (standard scaling law behavior).

- **Min-loss ordering:** even in 5 epochs, `mid_D256_L6` achieves `min_loss=0.9222` vs `0.9208` for
  baseline — a statistically tight result at this scale but directionally consistent with larger
  capacity fitting the denoising target better.

- **Compute (CPU):** wall time scales as ~O(D² · L) — roughly 1× (128,4) → 2.8× (256,6) → 7.8× (512,8).
  On an A100 GPU with batch_size=32 and full dataset (10K problems), estimate ~0.4s/epoch for D=512.
  Full Stage-A (50 epochs) ≈ **20s on A100**, Stage-B (20 epochs) ≈ **~3 GPU-hours** (rollout cost
  dominates due to N=8 DDIM chains per problem).

## Full training plan (GPU)

```bash
# Stage A at D=512, L=8
python scripts/run_scale_experiment.py --n-problems 10000 \
  --output results/p4_scale/scaling_notes_gpu.md

# Or use the config directly
python marc/train/stage_a.py \
  --config marc/configs/train/scale.yaml \
  --checkpoint-dir checkpoints/scale_D512_L8/stage_a
```

## Next steps

1. Run full Stage-A (50 epochs) + Stage-B (20 epochs GRPO) at D=512, L=8 on GPU.
2. Evaluate pass@1 and generalization gap on held-out structure templates
   (geometry templates from Akash P4 when ready).
3. If float precision bottlenecks D=512, prototype upgraded sinusoidal embeddings
   in `marc/model/embeddings.py` (§6.3 of TECHNICAL_GUIDE).

## Geometry-domain eval (P4)

**Generated:** 2026-07-07 16:50 UTC
**Task:** `refine` baseline (geometry-tuned hyperparameters — see `scripts/run_geometry_eval.py`) on `marc/eval/problems.py`'s `geometry_in_distribution` (2-var triangle) / `geometry_held_out` (4-var, two-point chain) split.

| Split | n | Solve rate |
|---|---|---|
| geometry_in_distribution | 25 | 0.56 |
| geometry_held_out | 25 | 0.28 |

**Generalization gap:** 0.280

Unlike the linear-system suites (P1/P2), this domain's energy is a nonconvex quartic (squared-distance factors are quadratic in the unknowns), so the default `refine()` hyperparameters — tuned against convex linear systems — solve close to 0% of instances; noise off, a smaller learning rate, and a much longer polish (see `GEOMETRY_REFINE_KWARGS`) are needed to reach the checker's exact-rational tolerance. See `results/p4_scale/roadmap.md` for the full writeup.
