# Structure toys - auxiliary-variable requirement (H2)

Three hand-built factor-graph problems for the H2 baseline: a fixed-structure
solver cannot solve them because the correct solution needs one more variable than
the graph exposes. Each toy ships in two forms - a `*_fixed` graph (the baseline
that must fail) and a `*_augmented` graph (the same problem with the auxiliary
variable added, which solves cleanly). The pair is a positive control: it shows the
fixed graph fails because the node is missing, not for any incidental reason.

Source of truth for the graphs is `marc/eval/structure_toys.py` (explicit
`FactorGraph` literals, mirroring `marc/eval/problems.py`). The JSON graphs here and
`gold.json` are emitted by `scripts/run_structure_toys.py`, which also runs the
baseline.

## Why a latent, not an inline expression

The checker (`marc/cas/checker.py`) accepts an assignment iff it satisfies every
factor's expression; it does not compare against the gold solution. So a merely
under-determined fixed graph is still satisfiable and would be scored as "solved."
For the fixed graph to genuinely fail, it must be inconsistent - no assignment
over its variables satisfies all factors at once.

That rules out any auxiliary quantity that is a closed-form function of the base
variables (`u = x + y`, `u = x*y`, `u = x**2`): those can be written inline in a
`FactorNode.expression`, keeping the fixed graph solvable. The auxiliary variable
must be a free latent with its own defining constraint. Removing its node (and
that defining factor) then deletes a real degree of freedom and tips an otherwise
consistent system into contradiction. `GradientRefinementSolver` sizes each
candidate to `len(graph.variables)` and cannot drive an inconsistent system's energy
to zero, so the checker rejects every sample, giving solve_rate = 0.0.

The three toys differ in how the latent enters, so the failure is not one
construction repeated three times.
## The toys

Every factor expression `g` denotes the constraint `g == 0`. In each augmented toy
the latent is `u`, defined by `u - 2` (so `u = 2`).

### toy1 - isolated latent offset (2 base vars)

Augmented `(x, y, u)`, gold (x, y, u) = (2, 1, 2):

eq1: x + y - u - 1     eq2: x - y + u - 3     eq3: x + u - 4     aux: u - 2

Fixed `(x, y)` drops `u` and its defining factor:

eq1: x + y - 1     eq2: x - y - 3     eq3: x - 4

Why fixed fails: eq1, eq2 fix (x, y) = (2, -1); the leftover eq3 demands
x = 4. Three constraints, two unknowns, no common solution. Auxiliary variable:
u (offset appearing in eq1-eq3).

### toy2 - latent coupled to one variable (3 base vars; extends held_out_structure)

Augmented `(x, y, z, u)`, gold (x, y, z, u) = (3, 2, 1, 2):

eq1: x + y + z - 6     eq2: x - y - 1     eq3: y - z - 1     eq4: x + u - 5     aux: u - 2

Fixed `(x, y, z)`:

eq1: x + y + z - 6     eq2: x - y - 1     eq3: y - z - 1     eq4: x - 5

Why fixed fails: the sum + two differences fix (x, y, z) = (3, 2, 1); the
leftover eq4 demands x = 5. Over-determined, inconsistent. Auxiliary variable:
u (couples to x in eq4).

### toy3 - latent shared across two measurements (2 base vars)

Augmented `(x, y, u)`, gold (x, y, u) = (2, 4, 2):

eq1: x + u - 4     eq2: y + u - 6     eq3: x + y - 6     aux: u - 2

Fixed `(x, y)`:

eq1: x - 4     eq2: y - 6     eq3: x + y - 6

Why fixed fails: eq1, eq2 force (x, y) = (4, 6); the leftover eq3 wants the
sum to be 6, but 4 + 6 = 10. Contradiction. Auxiliary variable: u (shared by
eq1 and eq2).
## Baseline result (fill in from the run)

`python -m scripts.run_structure_toys` (solver = `refine`, k = 4):

| toy | fixed pass@1 | fixed pass@k | augmented pass@1 |
| --- | --- | --- | --- |
| toy1 | no | no | yes |
| toy2 | no | no | yes |
| toy3 | no | no | yes |

fixed     solve_rate = 0.00   (inconsistent, latent missing)
augmented solve_rate = 1.00   (latent present, unique solution)

The augmented column is the positive control: adding the auxiliary node - and
nothing else - turns every failure into a solve.

## Files

- `toy{1,2,3}_fixed.json`, `toy{1,2,3}_augmented.json` - serialized `FactorGraph`s.
- `gold.json` - per-toy solution, variable order, and auxiliary-variable metadata.
- generator: `marc/eval/structure_toys.py`; runner: `scripts/run_structure_toys.py`.