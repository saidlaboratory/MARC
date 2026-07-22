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

## The v8 protocol is the strongest part of the result

One reference solver (scipy LM, k=4, owned by `invention_data`) certifies the data,
grades every arm, trains the reward, and runs the end-to-end solves; the identity is
test-enforced.  Nonlinear "unsolvable distractor" is an **exact CAS no-real-roots
proof** for 99% of test menus — a theorem, immune to solver budget and seed — and the
disclosed budget-relative probe for the rest.  Gold and distractor parameters share
one per-family support and one prior; anything mathematically equal to the gold is
excluded; golds must solve at eval grade under two independent seeds.  Both nonlinear
families use one-sided defining templates (`u = a·x² + δ`, `u = a·(x²+y²) + δ`,
a = ±1) because that is what makes rootless corruptions plentiful on both sides of
every gold; the old linear defining relation could not produce any (line ∩ hyperbola
is almost never empty), so the earlier menus were probe artifacts.

## Strongest numbers (Data Version 8; v6/v7 withdrawn — never cite 0.565 or 0.889)

- Balanced nonlinear: **0.997** [0.984, 1.000] vs 0.333 candidate-only and 0.236
  random (N=360); exact paired McNemar 239 vs 0, p=1.1e-72.  Per family: vieta
  1.000, quad_link 0.994.
- End-to-end nonlinear: the ranker's **single solver call sits at the
  oracle/enumeration ceiling** (0.933 = 0.933 = 0.933, N=60) vs 0.200/0.250
  controls, while enumeration spends 2.62 calls.
- Vieta-only → unseen quadratic relation: **0.420** [0.344, 0.500] vs 0.120 and
  0.253 (N=150): genuine but partial transfer, not relation-level invariance
  (in-relation vieta is 0.98).
- Held-out linear pattern: **0.380** [0.334, 0.428] vs 0.195 and 0.287 (N=400);
  all-pattern 0.339 [0.313, 0.366], full>random p=7.8e-07 (N=1,200).  The prior
  slot policy was 0.234 vs 0.238 random on this split.
- Checkpoint-only replays reproduce both headline evals exactly.

The movement across data versions is itself evidence: closing each shortcut lowered
the linear headline (0.565 → 0.445 → 0.380) and pushed the candidate-only control to
chance, while the nonlinear result rose (0.889 → 0.997) once menus carried
theorem-grade semantics.  Learning is decisive exactly where operator identity
matters, which is what the operator-aware encoder was built for — and the controls
now support that reading cleanly.

## Honest scope

This remains menu-based structural repair, not free-form expression generation.
All tasks are synthetic; linear certificates are exact rank theorems, nonlinear ones
exact CAS real-root proofs for 99% of menus.  The linear advantage is significant
but modest, and the K=4 checkpoint's zero-shot accuracy edge at larger linear menus
closes by K=16 (0.113 vs 0.107 random) — at that budget the win is wall-clock only
(1 call and ~4–5 ms vs 9.05 calls and 22.5 ms).  Direct K=16 training performs at
chance; both negatives stay in the record.  Near-saturation on balanced nonlinear
(0.997) means that benchmark is close to solved for this model class; the unseen-
relation transfer (0.420) is where headroom remains.  The cheap-probe control (#102)
is in: on nonlinear the ranker's one call (solve 0.939, 3.1 ms) beats the strongest
5-call probe (0.881, 34 ms) on accuracy and cost together — rootless distractors
cannot be probed into solving, while short probes do miss the gold.  On linear the
probe saturates (0.99 at ~4.4 calls) and enumeration is already perfect at 2.5, so
the linear rows are a mechanism demonstration, not a deployment case, and we say so.
Multiseed robustness is still being regenerated under v8 (#103).  Autoformalization
and open-ended multi-aux generation remain future work.

The AAAI story is therefore not "a universal solver."  It is a controlled result
about **structural amortization**: operator-aware scoring learns which graph repair
to spend a solver call on, reaches the enumeration ceiling at a third of its cost on
theorem-certified nonlinear menus, transfers partially to an unseen relation, and
reports exactly where its advantage runs out.
