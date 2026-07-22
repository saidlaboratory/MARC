# External validity: named real systems (R26)

**Status:** canonical writeup for the real-data / external-validity result. Directly
answers the "synthetic-only" critique — the study's biggest exposure. Numbers from
`results/p_real/real_systems.json` (`scripts/run_real_systems.py --K 8 --trials 200`).

## Why this exists

Every other family in the paper is procedurally generated. This suite is eight
*recognized* test problems, encoded once as factor graphs so the same solver battery
and the same single-start reachability measurement apply:

| system | domain | vars |
|---|---|---|
| circle intersection | geometry | 2 |
| conic–line intersection | geometry | 2 |
| trilateration (GPS-style positioning) | positioning | 2 |
| Rosenbrock stationary point | optimization | 2 |
| Himmelblau stationary points | optimization | 2 |
| 2R inverse kinematics | robotics | 4 |
| 3R inverse kinematics | robotics | 6 |
| cyclic-4 | computer-algebra benchmark | 4 |

Real roots here are irrational, so acceptance is a **numeric residual tolerance**
($\max_j |r_j(x)| < 10^{-6}$) — the fair criterion for comparing numerical solvers —
rather than the exact-rational checker the synthetic families use (those are
constructed to have rational solutions). Every system is verified real-solvable
(`tests/test_real_systems.py`).

## Result (best-of-8; acceptance $\max|r|<10^{-6}$)

| arm | solved / 8 |
|---|---|
| deterministic (one fixed start) | 0 |
| Langevin (best-of-8) | 1 |
| random restart + gradient polish (best-of-8) | 4 |
| **Levenberg–Marquardt (best-of-8)** | **8** |

Single-start reachability $q$ (gradient polish): 1.00 on the two circle/conic
geometries and 2R-IK, 0.38–0.41 on cyclic-4, and **0.00** on trilateration,
Rosenbrock, Himmelblau, and 3R-IK.

## What it says (three honest points)

1. **Classical solvers are near-unbeatable on real systems.** LM with multi-start
   solves 8/8. This is the paper's central thesis, now on problems from robotics,
   positioning, optimization, and the algebra-benchmark literature rather than on a
   family we built.
2. **Where MARC's gradient polish fails, the bottleneck is the polish, not the
   proposal.** Random restart fails exactly on the ill-conditioned systems (the
   Rosenbrock banana valley, Himmelblau saddles, an overdetermined trilateration
   basin, the 6-variable 3R chain), and LM — a stronger *classical* polish — fixes
   every one. A learned proposal would inherit the same weak gradient polish, so
   learning does not address the bottleneck; a better classical finisher does.
3. **No learning-favorable regime appears.** The regime the factorization law
   identifies (random restart collapses because it must independently hit $n$
   separable basins, and a learned proposal that memorizes marginals holds) requires
   high dimension *and* separability. These real systems are low-dimensional and
   coupled, so where $q$ collapses it is polish conditioning, not $v^{-n}$ basin
   scaling. The law's real-domain prediction — classical search suffices, learning has
   no room — holds.

## How to report it (honest, not defensive)

This is external validity, not a new positive. It converts "we characterized this on
synthetic families" into "we tested the characterization on eight standard real
systems, and it held: classical LM solves all eight, and no real system in the suite
falls in the amortization regime where a learned proposal would help." That is exactly
the honest boundary the paper argues for, now demonstrated outside the constructed
families. The learned arm is not run per system (each has a distinct structure, so
there is nothing to amortize across a suite of size one per structure); the reachability
numbers are what test the law's prediction.

`PYTHONPATH=. python3 scripts/run_real_systems.py --K 8 --trials 200`
