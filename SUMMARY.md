# MARC — project summary

**MARC** is a graph-diffusion approach to constraint solving: math problems are encoded as
**factor graphs** (variables + constraint factors), a **GNN denoiser** proposes candidate
assignments by reverse diffusion, and a **CAS** (SymPy) checks/guides them. This file is an
honest snapshot of what currently exists and works.

## What works today

### Pipeline (P0–P4, all implemented, 205 tests passing)
- **Graph core** — `VariableNode`/`FactorNode`/`Edge`, `FactorGraph`, JSON I/O, PyG
  `HeteroData` builder (now encodes each factor's constant term).
- **CAS** — residuals, energy, energy-gradient, numeric+symbolic checker.
- **Diffusion** — cosine schedule, forward corruption, DDIM sampler, CAS-guided `solve()`.
- **Model** — `GraphDenoiser`: bipartite message passing with timestep + constant conditioning.
- **Training** — Stage A (denoising) + Stage B (GRPO RL against the checker reward).
- **Refinement** — energy-gradient / annealed-Langevin solver (`marc/refine`).
- **Eval** — solve rate, generalization gap, entrapment, perturbation, length extrapolation;
  ablations (noise / guidance / purist reward); CoT LLM baseline.
- **Data** — generators for linear systems (2×2, 3×3) and geometry; structure toys; NL parser
  for a small set of sentence templates.

### Results
1. **The learned solver converges.** It previously diverged (inference ≈ 1e4, 0% solve). After
   fixing five bugs — the denoiser never saw the noised input; equation constants were absent
   from the graph tensors; the timestep didn't condition variables; inference guidance exploded;
   a 1-variable `squeeze` bug — it now **solves 100%** on in-distribution and held-out linear
   systems (generalization gap 0), and Stage-A loss drops from a flat ~1.0 to ~0.37.
   Method = diffusion proposal + energy-descent polish. See `paper/notes/learned_solver_fix.md`.
2. **Noise escapes entrapment (H1/RQ2).** On 200 non-convex problems, deterministic descent is
   **100% trapped**; annealed-noise (Langevin) descent cuts entrapment to 0.48 — reduction
   **0.52 ± 0.09** (95% CI excludes 0). `results/p1_entrapment/`.
3. **Learned inference beats classical + prior, scaling in dimension.** On bundled non-convex
   traps with per-instance-varying solutions, the learned model beats deterministic (0),
   Langevin (→0 by n=3), and a mean-prior (0) at every dimension, though it also degrades at
   n=6 (0.68 → 0.10). Required a specific architectural fix (condition variables on incident
   constraint constants + a direct skip to the output). See `paper/notes/dimension_scaling_result.md`
   and `scripts/run_dimension_scaling.py`.
4. **CoT baseline** runs on Gemini (`gemini-flash-lite-latest`) via the OpenAI-compatible
   endpoint, with backoff + resume cache; full N=25 completed (in-dist 1.0, held-out 1.0).

## What this is NOT (honest scope)
- **Toy problems only.** Largest solved: a 3×3 linear system, or a bundle of 1-variable cubic
  traps. Problems arrive *already encoded* as factor graphs with explicit residuals.
- **No proofs, no natural math.** It finds numeric solutions to pre-encoded equations. It does
  **not** do olympiad/competition math (IMO/USAMO): no NL understanding of real problems, no
  multi-step proof, no lemma invention. That is a different research program entirely.
- **Checkpoints are toy-scale** CPU runs (D=128, L=4, minutes) — enough to exercise the
  pipeline, not a scaling claim. Full-scale training (`results/p4_scale/roadmap.md`) is future
  work.
- The scaling result's mechanism (amortized learned proposals beating blind search in high-D)
  is a **known principle**; the contribution is a concrete graph-constraint-solving instance of
  it plus the conditioning architecture it needs.

## Key entry points
| Area | Path |
|---|---|
| Solve one problem | `scripts/solve_one.py` |
| Train checkpoints | `scripts/train_p2_checkpoints.py` |
| Main paper eval | `scripts/run_main_eval.py` |
| Entrapment ablation | `python -m marc.eval.ablations.noise_ablation --graphs 200` |
| Dimension scaling | `scripts/run_dimension_scaling.py` |
| CoT baseline | `GEMINI_API_KEY=… python -m marc.eval.baselines.cot_baseline` |
| Investigation notes | `paper/notes/learned_solver_fix.md`, `paper/notes/dimension_scaling_result.md` |

## Status for AAAI
A **working** method with one clean scaling result and honest, well-characterized limitations —
a plausible submission if scoped to small constraint solving. It is not a state-of-the-art or
competition-math result, and the write-ups say so.
