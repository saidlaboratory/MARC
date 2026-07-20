# Guidance & purist-reward ablations — reframe

**Fixing-plan items:** A4 (guidance), A5 (purist reward). Both current ablations are
**degenerate** and must not ship in their current form.

## Current (broken) state
- **Guidance sweep** (`summary_table.md`): "best w = 0.0" — guidance weight appears useless.
- **Purist reward** (`summary_table.md`): "shaping gain 0.000 (standard 0.000 vs purist 0.000)".

Both ran on the **convex suite**, which is saturated: on convex linear systems the
diffusion-proposes / refine-polishes hybrid solves regardless of the guidance weight
(the polish converges from any finite start), so every weight ties at the ceiling. Zero
vs. zero distinguishes nothing.

## Why guidance weight genuinely doesn't drive the result (honest finding)
Our investigation (see `paper/learned_solver_fix.md`, `paper/dimension_scaling_result.md`)
established the actual mechanism:
1. The eps-prediction + CAS-energy-guidance sampler **wanders** and degrades in higher
   dimension (it scored 0 at n≥2 in the scaling study). Classifier-style guidance is **not**
   what makes the learned solver work.
2. The learned solver's power comes from **x0-prediction proposals + an energy-descent polish**
   (the hybrid). The proposal puts the iterate in the right basin; the polish reaches
   tolerance. Guidance weight is largely subsumed by the polish.

So "guidance weight doesn't separate" is a **real finding**, not a null artifact — but it means
the guidance sweep is the wrong ablation to headline.

## What replaces them (already have the numbers)
The **A8.1 hybrid-vs-refine ablation** (`scripts/run_hard_eval.py`, `paper/figures/hard_suite_table.md`)
is the correct mechanism ablation and it is **not** degenerate:

| Family | refine cold | refine + Langevin | learned hybrid |
|---|---|---|---|
| BilinearSystem | 0.000 | 0.350 | **0.625** |
| BilinearProduct | 0.000 | 0.125 | **0.725** |

This isolates the denoiser's contribution on non-convex problems where solve rates are off
the floor — which is exactly what A4 was trying (and failing) to show at the ceiling.

## Paper actions
1. **Drop** the convex guidance-weight sweep and the purist-reward table from the results.
2. **Headline** the A8.1 hybrid-vs-refine ablation as the mechanism validation.
3. One honest sentence on guidance: *"Classifier-style CAS guidance did not improve solve rate
   over the proposal+polish hybrid; the learned proposal, not the guidance term, is what lets
   refinement succeed on non-convex problems."*
4. One honest sentence on reward shaping: *"Reward shaping (standard vs. checker-only) showed no
   measurable effect at this scale."*

## If time allows (P1, not required)
Run a guidance sweep on a **hard-trained** eps-checkpoint over the bilinear suite to report a
non-saturated guidance curve. Expected outcome per the above: flat / no benefit. Record in
`paper/PROVENANCE.md` before it enters the draft.
