"""Hand-built structure toys for H2 (auxiliary-variable requirement).

Each toy is a linear system that is consistent and uniquely solvable over
``(base variables + one latent u)`` but becomes **over-determined and
inconsistent** the moment u's node (and its defining factor) is removed. The
latent u is the auxiliary variable H2 claims a fixed-structure solver cannot
introduce.

Why a *latent* and not an inline expression (u = x+y, u = x*y, ...): the checker
(``marc.cas.checker.Checker``) only tests whether a candidate satisfies every
factor's expression; any aux quantity that is a closed-form function of the base
variables can simply be written inline inside a ``FactorNode.expression``, leaving
the fixed graph solvable. Only a *free* latent with its own defining constraint
makes dropping the node delete a real degree of freedom, tipping an otherwise
consistent system into contradiction. That contradiction is the baseline failure:
``GradientRefinementSolver`` (name="refine") sizes its candidate to
``len(graph.variables)`` and can never drive the energy of an inconsistent system
to zero, so the checker rejects every sample -> solve_rate == 0.0.

Pattern mirrors :func:`marc.eval.problems.held_out_structure`: explicit
``FactorGraph`` literals with a known exact solution so the checker is ground
truth. Each toy ships in two forms:

* ``*_fixed``     — base variables only; inconsistent (the H2 baseline that fails).
* ``*_augmented`` — base variables + latent u; consistent, unique solution.

The three differ in how the latent enters (isolated offset / single-variable
coupling / shared across two measurements) so the failure is not one construction
repeated three times.
"""

from __future__ import annotations

from typing import Dict, List

from marc.eval.runner import Problem
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode


# --- per-toy auxiliary-variable documentation (single source of truth) --------
#: toy -> (auxiliary symbol, its value in the augmented gold, one-line reason the
#: fixed graph is inconsistent). Consumed by scripts/run_structure_toys.py to emit
#: gold.json and by the README table.
AUX_INFO: Dict[str, dict] = {
    "toy1": {
        "aux_var": "u",
        "aux_value": 2.0,
        "why_fixed_fails": (
            "Drop u: eq1/eq2 fix (x,y)=(2,-1); the leftover eq3 (x-4) demands x=4. "
            "3 constraints, 2 unknowns, no common solution."
        ),
    },
    "toy2": {
        "aux_var": "u",
        "aux_value": 2.0,
        "why_fixed_fails": (
            "Drop u: sum + two differences fix (x,y,z)=(3,2,1); the leftover eq4 "
            "(x-5) demands x=5. Over-determined, inconsistent."
        ),
    },
    "toy3": {
        "aux_var": "u",
        "aux_value": 2.0,
        "why_fixed_fails": (
            "Drop u (which appeared in two measurements): eq1/eq2 force (x,y)=(4,6); "
            "the leftover eq3 (x+y-6) wants the sum to be 6. Contradiction."
        ),
    },
}


def _edge(var: str, factor: str, coef: float = 1.0) -> Edge:
    return Edge(var, factor, coef)


# --- toy 1: 2 base vars + isolated latent offset ------------------------------

def _toy1_augmented() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y"), VariableNode("u")],
        factors=[
            FactorNode("eq1", "x + y - u - 1"),
            FactorNode("eq2", "x - y + u - 3"),
            FactorNode("eq3", "x + u - 4"),
            FactorNode("aux", "u - 2"),          # defines the latent
        ],
        edges=[
            _edge("x", "eq1"), _edge("y", "eq1"), _edge("u", "eq1", -1),
            _edge("x", "eq2"), _edge("y", "eq2", -1), _edge("u", "eq2", 1),
            _edge("x", "eq3"), _edge("u", "eq3", 1),
            _edge("u", "aux", 1),
        ],
    )
    return Problem(
        id="toy1_augmented",
        graph=graph,
        solution=[2.0, 1.0, 2.0],               # (x, y, u)
        description="2 base vars + latent u; consistent (gold x=2, y=1, u=2)",
        metadata={"split": "structure_toy_augmented", "n_vars": 3,
                  "toy": "toy1", "requires_aux": True},
    )


def _toy1_fixed() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[
            FactorNode("eq1", "x + y - 1"),
            FactorNode("eq2", "x - y - 3"),
            FactorNode("eq3", "x - 4"),
        ],
        edges=[
            _edge("x", "eq1"), _edge("y", "eq1"),
            _edge("x", "eq2"), _edge("y", "eq2", -1),
            _edge("x", "eq3"),
        ],
    )
    return Problem(
        id="toy1_fixed",
        graph=graph,
        # intended projection of the true solution; UNREACHABLE in this graph
        # (the graph is inconsistent). Stored for documentation only — the checker
        # validates candidate length against graph.variables, not against this.
        solution=[2.0, 1.0],                    # (x, y)
        description="2 base vars, latent dropped; inconsistent (H2 baseline)",
        metadata={"split": "structure_toy_fixed", "n_vars": 2,
                  "toy": "toy1", "requires_aux": True},
    )


# --- toy 2: 3 base vars + latent coupled to one variable ----------------------

def _toy2_augmented() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y"),
                   VariableNode("z"), VariableNode("u")],
        factors=[
            FactorNode("eq1", "x + y + z - 6"),
            FactorNode("eq2", "x - y - 1"),
            FactorNode("eq3", "y - z - 1"),
            FactorNode("eq4", "x + u - 5"),
            FactorNode("aux", "u - 2"),
        ],
        edges=[
            _edge("x", "eq1"), _edge("y", "eq1"), _edge("z", "eq1"),
            _edge("x", "eq2"), _edge("y", "eq2", -1),
            _edge("y", "eq3"), _edge("z", "eq3", -1),
            _edge("x", "eq4"), _edge("u", "eq4", 1),
            _edge("u", "aux", 1),
        ],
    )
    return Problem(
        id="toy2_augmented",
        graph=graph,
        solution=[3.0, 2.0, 1.0, 2.0],          # (x, y, z, u)
        description="3 base vars + latent u; consistent (gold x=3, y=2, z=1, u=2)",
        metadata={"split": "structure_toy_augmented", "n_vars": 4,
                  "toy": "toy2", "requires_aux": True},
    )


def _toy2_fixed() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y"), VariableNode("z")],
        factors=[
            FactorNode("eq1", "x + y + z - 6"),
            FactorNode("eq2", "x - y - 1"),
            FactorNode("eq3", "y - z - 1"),
            FactorNode("eq4", "x - 5"),
        ],
        edges=[
            _edge("x", "eq1"), _edge("y", "eq1"), _edge("z", "eq1"),
            _edge("x", "eq2"), _edge("y", "eq2", -1),
            _edge("y", "eq3"), _edge("z", "eq3", -1),
            _edge("x", "eq4"),
        ],
    )
    return Problem(
        id="toy2_fixed",
        graph=graph,
        solution=[3.0, 2.0, 1.0],               # (x, y, z) intended projection
        description="3 base vars, latent dropped; inconsistent (H2 baseline)",
        metadata={"split": "structure_toy_fixed", "n_vars": 3,
                  "toy": "toy2", "requires_aux": True},
    )


# --- toy 3: 2 base vars + latent shared across two measurements ---------------

def _toy3_augmented() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y"), VariableNode("u")],
        factors=[
            FactorNode("eq1", "x + u - 4"),
            FactorNode("eq2", "y + u - 6"),
            FactorNode("eq3", "x + y - 6"),
            FactorNode("aux", "u - 2"),
        ],
        edges=[
            _edge("x", "eq1"), _edge("u", "eq1", 1),
            _edge("y", "eq2"), _edge("u", "eq2", 1),
            _edge("x", "eq3"), _edge("y", "eq3"),
            _edge("u", "aux", 1),
        ],
    )
    return Problem(
        id="toy3_augmented",
        graph=graph,
        solution=[2.0, 4.0, 2.0],               # (x, y, u)
        description="2 base vars + shared latent u; consistent (gold x=2, y=4, u=2)",
        metadata={"split": "structure_toy_augmented", "n_vars": 3,
                  "toy": "toy3", "requires_aux": True},
    )


def _toy3_fixed() -> Problem:
    graph = FactorGraph(
        variables=[VariableNode("x"), VariableNode("y")],
        factors=[
            FactorNode("eq1", "x - 4"),
            FactorNode("eq2", "y - 6"),
            FactorNode("eq3", "x + y - 6"),
        ],
        edges=[
            _edge("x", "eq1"),
            _edge("y", "eq2"),
            _edge("x", "eq3"), _edge("y", "eq3"),
        ],
    )
    return Problem(
        id="toy3_fixed",
        graph=graph,
        solution=[2.0, 4.0],                    # (x, y) intended projection
        description="2 base vars, shared latent dropped; inconsistent (H2 baseline)",
        metadata={"split": "structure_toy_fixed", "n_vars": 2,
                  "toy": "toy3", "requires_aux": True},
    )


# --- public collections -------------------------------------------------------

def structure_toys_fixed() -> List[Problem]:
    """The 3 fixed-structure toys (H2 baseline; each is inconsistent)."""
    return [_toy1_fixed(), _toy2_fixed(), _toy3_fixed()]


def structure_toys_augmented() -> List[Problem]:
    """The 3 augmented toys (base vars + latent; each consistent + unique)."""
    return [_toy1_augmented(), _toy2_augmented(), _toy3_augmented()]


def all_structure_toys() -> List[Problem]:
    """All 6 problems (3 fixed + 3 augmented), fixed first."""
    return structure_toys_fixed() + structure_toys_augmented()
