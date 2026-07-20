# Structure invention (③) — feasibility test: negative

**Premise:** a model that *invents auxiliary variables/lemmas* to make hard problems solvable
would be a novel, main-track contribution. **Prerequisite:** the auxiliary must make the problem
*easier for the solver*. It does not.

## Result — adding the auxiliary HURTS the numeric solver (refine, best-of-8, 40 instances)
| toy | fixed | augmented (with auxiliary) | lift |
|---|---|---|---|
| sum_product | 0.250 | 0.100 | -0.150 |
| bilinear_product | 0.050 | 0.000 | -0.050 |
| quadratic_link | 0.250 | 0.000 | -0.250 |

## Why (fundamental, not a tuning issue)
The Vieta-style auxiliary (`d=x-y`, `d²=s²-4p`) helps *human/symbolic* solving: solve the 1-var
`d` sub-problem first, then substitute to linearize. A gradient/energy numeric solver does **not**
exploit that ordering — it just sees a **higher-dimensional constraint system with more factors**
and searches a bigger space, so solve rate *drops*. The auxiliary's value is realized only by
**staged/symbolic** solving — which is precisely what a CAS (SymPy) already does, and is not the
learned solver's contribution.

## Implication for main-track
Structure invention as "add an auxiliary → easier for MARC's solver" is **closed**. To make it
work would require a *staged/hierarchical* solver that solves the auxiliary sub-problem first and
substitutes — i.e. re-implementing symbolic elimination, where the novelty (over CAS) is unclear.
Combined with R7 (learned proposal ties random on coupled systems), **both main-track routes we
could test in-week are negative.** The honest target is a **workshop** paper; a main-track result
would need a real domain with strong results and/or GPU-scale training — a longer effort than the
7-day window. `python -c` premise test in git history on branch sparsh/structure-invention.
