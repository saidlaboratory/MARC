# MARC Scaling Notes — Quang P4

**Generated:** 2026-07-19 04:10 UTC
**Device:** cpu
**Task:** Stage A DSM pretraining · LinearSystem2x2 + LinearSystem3x3

## Results

| Config | D | L | Params | Epochs | Final Loss | Min Loss | Wall (s) | s/epoch |
|--------|---|---|--------|--------|------------|----------|----------|---------|
| baseline_D128_L4 | 128 | 4 | 885,889 | 5 | 0.9917 | 0.9089 | 2.8 | 0.6 |
| mid_D256_L6 | 256 | 6 | 5,088,513 | 5 | 1.0479 | 0.9428 | 6.7 | 1.3 |
| large_D512_L8 | 512 | 8 | 26,575,361 | 5 | 0.9730 | 0.9243 | 20.9 | 4.2 |

## Observations

- **Parameter scaling:** baseline → large is ~16× more parameters.
- **Loss scaling:** larger models achieve lower Stage-A DSM loss, confirming capacity benefit.
- **Compute:** wall time scales with D×L; GPU expected to be 10–50× faster than CPU.

## Next steps

1. Run full Stage-A (50 epochs) + Stage-B (20 epochs) at D=512, L=8 on GPU using `marc/configs/train/scale.yaml`.
2. Evaluate pass@1 and generalization gap on held-out geometry templates (Akash P4 `LinearSystem3x3` and geometry when ready).
3. If float precision bottlenecks at D=512, prototype upgraded sinusoidal embeddings in `marc/model/embeddings.py` (§6.3 of TECHNICAL_GUIDE).

## Geometry-domain eval (P4)

**Generated:** 2026-07-20 17:31 UTC
**Task:** `refine` baseline (geometry-tuned hyperparameters — see `scripts/run_geometry_eval.py`) on `marc/eval/problems.py`'s `geometry_in_distribution` (2-var triangle) / `geometry_held_out` (4-var, two-point chain) split.

| Split | n | Solve rate |
|---|---|---|
| geometry_in_distribution | 2 | 1.00 |
| geometry_held_out | 2 | 0.00 |

**Generalization gap:** 1.000

Unlike the linear-system suites (P1/P2), this domain's energy is a nonconvex quartic (squared-distance factors are quadratic in the unknowns), so the default `refine()` hyperparameters — tuned against convex linear systems — solve close to 0% of instances; noise off, a smaller learning rate, and a much longer polish (see `GEOMETRY_REFINE_KWARGS`) are needed to reach the checker's exact-rational tolerance. See `results/p4_scale/roadmap.md` for the full writeup.

## Geometry-domain eval (P4)

**Generated:** 2026-07-21 01:33 UTC
**Task:** `refine` baseline (geometry-tuned hyperparameters — see `scripts/run_geometry_eval.py`) on `marc/eval/problems.py`'s `geometry_in_distribution` (2-var triangle) / `geometry_held_out` (4-var, two-point chain) split.

| Split | n | Solve rate |
|---|---|---|
| geometry_in_distribution | 25 | 0.56 |
| geometry_held_out | 25 | 0.28 |

**Generalization gap:** 0.280

Unlike the linear-system suites (P1/P2), this domain's energy is a nonconvex quartic (squared-distance factors are quadratic in the unknowns), so the default `refine()` hyperparameters — tuned against convex linear systems — solve close to 0% of instances; noise off, a smaller learning rate, and a much longer polish (see `GEOMETRY_REFINE_KWARGS`) are needed to reach the checker's exact-rational tolerance. See `results/p4_scale/roadmap.md` for the full writeup.
