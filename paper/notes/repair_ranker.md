# Candidate-conditioned structural repair: paper-facing note

## Claim

MARC should not learn continuous values that mature numerical solvers already find
well.  It should amortize the discrete decision those solvers cannot make: **which
representation-changing augmentation makes this problem solvable?**

The v0.3 model scores candidate-*augmented graphs*, rather than encoding the fixed
graph once and classifying menu slots.  Its polynomial-semantic GNN exposes degree,
linear/squared/cross participation, constants, and incidence coefficients.  A
listwise softmax selects one repair, a classical backend finds values, and an exact
checker gates acceptance.

This is a substantive successor to the v0.2 menu policy:

1. the candidate and problem interact throughout message passing;
2. nonlinear operators are visible instead of being collapsed to a constant and a
   nominal edge coefficient;
3. graph scoring is invariant to menu order and naturally accepts a different K;
4. the learned computation is attached to the discrete structural choice, not the
   continuous solve.

## Strongest numbers

- Held-out linear pattern: 0.565 vs 0.343 candidate-only and 0.283 random (N=400).
  The prior slot policy was 0.234 vs 0.238 random on this split.
- Balanced nonlinear: 0.889 vs 0.422 and 0.253 (N=360); exact paired p=4.1e-46.
- Nonlinear optimization seeds: 0.889/0.875/0.881 (SD 0.006).
- End-to-end nonlinear: 0.883 vs 0.433 and 0.250, oracle/enumeration 0.950 (N=60),
  with one solver call instead of 2.62.
- Vieta-only → unseen quadratic relation: 0.367 vs 0.240 and 0.213 (N=150):
  positive but partial transfer, not relation-level invariance.
- K=4→16 zero-shot menu scaling: 0.540→0.247 while random falls 0.253→0.020;
  measured enumeration speedup grows 1.21x→3.91x.

## Honest scope

This remains menu-based structural repair, not free-form theorem or expression
generation.  Linear tasks are synthetic and exactly certified.  Nonlinear
“unsolvable distractor” means failure of a stated multi-restart refinement probe,
not a theorem that no real root exists.  Candidate-only is stronger on Vieta than
on quadratic links; report per-family rows.  Direct K=16 training underperforms
zero-shot transfer.  Autoformalization and open-ended multi-aux generation remain
future work.

The AAAI story is therefore not “a universal solver.”  It is a controlled result
about **structural amortization**: operator-aware scoring learns which graph repair
to spend a solver call on, transfers across one unseen graph pattern and partially
across one unseen nonlinear relation, and approaches an enumeration oracle at a
cost advantage that grows with menu size.

