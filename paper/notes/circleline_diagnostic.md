# Why the learned proposal scores 0.000 on CircleLine

Diagnostic for the R3/R4 failure (learned hybrid 0.000 vs random-init+polish 0.200
on CircleLine; adding CircleLine to the training mix also collapses BilinearSystem
transfer, R4). Produced by `PYTHONPATH=. python3 scripts/diagnose_circleline.py`
(200 held-out instances, best-of-8, x0 model trained on
CircleLine only, 100 epochs x 300 instances — the run_hard_eval
recipe). Full numbers in `results/p_hard/circleline_diag.json`.

## The target is bimodal and its mean is infeasible

Every CircleLine instance (x²+y²=r, x+y=s) has exactly two real roots, and they are
mirror images under x↔y: (x\*, y\*) and (y\*, x\*). Measured over
200 instances, the two roots sit 4.13 apart on
average (never closer than 1.41). Their mean — the chord
midpoint (s/2, s/2) — satisfies the line exactly but misses the circle on every
instance: energy E = (x\*−y\*)⁴/8 at the midpoint averages
31.11 and is never below 0.125.
So the minimizer of the MSE/x0 objective, the conditional mean of the roots, is an
infeasible point by construction.

## The trained proposal lands on the diagonal, near the midpoint

The x and y nodes of a CircleLine factor graph have identical neighborhoods, so the
permutation-equivariant denoiser can only tell them apart through its noise input —
which x0-regression trains it to ignore. The measurement agrees: proposals from the
CircleLine-only model sit 1.5e-09 off the diagonal x=y on average
(max 1.5e-08 — float precision; the x and y outputs are identical),
against a mean root gap of 4.13.
They land 1.48 from the chord midpoint versus
2.81 from the nearest root, and
100% of proposals are closer to the midpoint
than to either root — the regression-to-the-mean signature.

## The diagonal is in neither root's basin

The energy is symmetric under x↔y, so deterministic gradient descent started on the
diagonal stays on it, and no root lies there (roots have x≠y). Polish from the exact
midpoint reaches a root on 0/200 instances. Best-of-8 polish
from the learned proposals solves 0/200 = 0.000
[0.00,0.02]; the same polish budget from random inits
in [−5,5] solves 47/200 = 0.235
[0.18,0.30]. Random starts break the symmetry the
proposal cannot; that asymmetry is the whole 0.000-vs-0.200 gap.

## Implication for the paper

This is the cleanest concrete instance of the central claim: an MSE/x0 proposal
learns the mean of a multimodal solution set, not its modes. CircleLine is the
worst case because its two modes are exact mirror images, the mean lies on a
symmetry axis that is invariant under the polish dynamics, and the architecture
itself cannot break the tie. It also offers a plausible reading of the R4 transfer
collapse — CircleLine gradients pull shared weights toward symmetric (diagonal)
predictions that are wrong for BilinearSystem too — though that link is untested
here. Fixes would need a multimodal proposal head (sampling the reverse chain
rather than one-shot x0, or symmetry-breaking features), not more training.
