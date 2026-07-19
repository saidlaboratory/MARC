# H1 result: learned per-instance inference beats classical refinement AND a prior

## Claim
On high-dimensional non-convex constraint problems, the learned diffusion model does
**per-instance inference** that solves where **all** classical baselines fail: deterministic
descent (trapped), Langevin/noise (collapses geometrically in dimension), and a trivial
"guess-the-average" prior (fails, because solutions vary per instance). The learned model's
advantage **holds as dimension grows**, though it too degrades at the largest n.

## Setup
`n` non-convex traps bundled into one problem (`scripts/run_dimension_scaling.py`). Each
factor `r_i = (x_i - R_i)((x_i - m_i)^2 + h_i)/15` has its only real root at
`R_i = ±U[3,8]` — **wide magnitude and random sign**, so the solution genuinely varies per
instance and a constant guess cannot work — and a spurious energy basin near `m_i ~ 0`
where the start sits. Reaching the solution requires crossing the barrier for **every**
variable. All methods share one local solver (backtracking line-search energy descent,
optional annealed Langevin noise) and differ **only** in the starting point (maximally
fair). Acceptance is solution-space (`|x_i - R_i| < 0.05`), the scaling-invariant metric.
Equal best-of-K budget (K=8), 40 held-out problems per n; learned model trained on a
disjoint 200-problem split per n.

## Result (real `GraphDenoiser`, end-to-end)

| n | deterministic | Langevin | mean-prior | **learned** |
|---|---|---|---|---|
| 1 | 0.000 | 0.225 | 0.000 | **0.675** |
| 2 | 0.000 | 0.025 | 0.000 | **0.425** |
| 3 | 0.000 | 0.000 | 0.000 | **0.550** |
| 4 | 0.000 | 0.000 | 0.000 | **0.650** |
| 6 | 0.000 | 0.000 | 0.000 | **0.100** |

Langevin decays geometrically (0.225 → 0.025 ≈ 0.225²) and is 0 by n=3. The mean-prior is
0 everywhere (its constant guess sits at the barrier). The learned model beats every baseline
at every n. **Honest caveat:** the learned model is *not* dimension-immune — its per-instance
inference is imperfect (~0.68 per variable), so errors compound and it falls to 0.10 by n=6.
The accurate claim is "decays much slower than Langevin and always beats the prior," not
"flat in n."

## The architectural finding (what made it work)
The learned model must **infer each variable's value from its constraint**, and two things
were required (both now in `marc/model/`):
1. **Condition variables directly on their incident factor constants** (not only via message
   passing). Without it the model collapses to predicting the mean solution (sign- and
   magnitude-blind: mean|err| ≈ 5.4 on ±[3,8] roots).
2. **A direct skip from the incident constants to the output.** The message-passing stack's
   per-round LayerNorm washes out the constants' *magnitude*; the skip restores it. With both,
   the model recovers roots at mean|err| ≈ 0.9 (correct sign every time).

The eps-prediction + energy-guidance sampler does **not** produce this (it wanders and scores
0 at n≥2); the result uses a direct **x0-prediction proposal** + energy-descent polish.

## Honest caveats
- Synthetic family, constructed to isolate the mechanism; not natural math problems.
- Learned solve rates are 0.10–0.68, not ≈1.0, and degrade at n=6 (see above).
- The mechanism (amortized learned proposal beating blind search in high dimension) is a
  known principle; the contribution here is demonstrating it concretely for graph-based
  constraint solving and identifying the conditioning architecture it requires.

## Reproduce
```
python scripts/run_dimension_scaling.py          # full: n = 1,2,3,4,6
python scripts/run_dimension_scaling.py --quick   # CI smoke
```
Writes `results/p_scaling/scaling.json` and `paper/figures/fig_dimension_scaling.pdf`.
