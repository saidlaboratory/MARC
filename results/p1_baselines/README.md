# P1 Baselines

`metrics.json` — capability + generalization metrics (TECHNICAL_GUIDE §11) for the
current solver over two structural splits.

## Reproduce

```bash
python scripts/run_p1_eval.py                 # energy-gradient refinement baseline
python scripts/run_p1_eval.py --solver learned --k 8   # once Quang's checkpoint lands
```

## Current numbers (solver = `refine`)

| Split | solve_rate (pass@1) | pass@4 | perturbation_robustness |
|---|---|---|---|
| in-distribution (2-var linear) | 1.00 | 1.00 | 0.00 |
| held-out-structure (3-var linear) | 1.00 | 1.00 | 0.00 |

* **generalization_gap = 0.00.** The baseline is the exact energy-gradient solver
  (TECHNICAL_GUIDE §3.4, noise on, deterministic polish tail). It *derives* the
  answer, so held-out structure costs it nothing — this is the zero-gap ideal the
  learned model is measured against, not a recall artifact.
* **perturbation_robustness = 0.00** (drop in solve rate when constants are shifted
  and the solver re-derives). The solver re-solves perturbed problems perfectly, so
  zero drop — the expected signature of a deriving (vs. memorizing) solver.

## Status of the learned solver

The learned diffusion `solve()` (`marc/diffusion/solve.py`) and GNN denoiser
(`marc/model/denoiser.py`) exist on `main`, but **no trained checkpoint exists yet**
and the GNN path needs `torch_geometric`. So these P1 numbers come from the
energy-gradient baseline. The learned solver plugs into the same `Solver` contract
(`marc/eval/solver.py::LearnedSolver`) with no harness change:

```bash
MARC_CKPT=/path/to/denoiser.pt python scripts/run_p1_eval.py --solver learned
```
