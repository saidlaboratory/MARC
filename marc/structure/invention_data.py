"""Menu-based structure-invention data (U5).

Scope ladder (TECHNICAL_GUIDE §14 — "simplest thing that exhibits the phenomenon";
§10 defines invention as an ABSENT slot going concrete during denoising):

  1. THIS MODULE: menu-based activation. Given an unsolvable fixed graph and a
     procedurally generated menu of K candidate augmentations (aux variable ``u``
     with a defining factor + insertion coefficients), exactly one of which makes
     the graph solvable, the policy must pick the right invention.
  2. OUT OF SCOPE: free-form expression invention (emitting an arbitrary defining
     expression). The menu formulation is the honest preliminary rung.

Each :class:`Candidate` is a complete augmentation recipe; :func:`build_menu`
certifies every distractor unsolvable before it enters the menu. HOW it certifies
depends on the graph, and the certificate is recorded per instance:

* Linear graphs (sources ``"toys"``/``"aux_required"``): an exact symbolic rank
  check — ``certificate="exact"``; "exactly one solvable option" is a theorem
  about the data.
* Nonlinear graphs (source ``"nonlinear"``): the rank check does not apply.
  Primary certificate: an exact CAS real-root decision — a distractor whose
  augmented system provably has no real solution is unsolvable by ANY solver at
  ANY budget (``certificate="exact"``, method ``cas_no_real_roots``). Only when
  the CAS cannot decide does the empirical probe run (``DEFAULT_PROBE``:
  ``n_seeds * k_refine`` restarts of ``REFERENCE_SOLVER`` — the same scipy-LM
  protocol that grades every eval arm); those instances carry
  ``certificate="empirical"`` with the probe config attached, an empirical
  claim at a stated budget, not a theorem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp
import torch

from marc.cas.checker import Checker
from marc.eval.solver import load_solver
from marc.graph.graph import FactorGraph
from marc.graph.schema import Edge, FactorNode, VariableNode

from .schema import ABSENT, PaddedGraph, SlotType

#: Bump whenever identical seeds generate different menus.  Version 8 unifies
#: the reference solver: certification (gold solvability AND distractor
#: unsolvability) now uses the same scipy-LM protocol that grades the eval
#: arms, expression distractors can no longer duplicate the gold via a zero
#: shift, and menu-filler insertion support is size-uniform like the gold's.
DATA_VERSION: int = 8

SOURCES: Tuple[str, ...] = ("toys", "aux_required", "nonlinear")

# aux_required owns the shared gold/distractor priors (pin values, support
# sampling) and its OWN pattern vocabulary (offset/coupled/shared). Literal
# fallbacks preserve the "aux_required import is optional" design: this module
# still works on a branch without marc/data/aux_required.py.
try:  # pragma: no cover - trivial import guard
    from marc.data.aux_required import _AUX_PIN
    from marc.data.aux_required import PATTERNS as _AUX_PATTERNS
    from marc.data.aux_required import sample_support
    _AUX_FAMILIES: Tuple[str, ...] = tuple(_AUX_PATTERNS)
except Exception:  # pragma: no cover
    _AUX_PIN = [-4, -3, -2, -1, 1, 2, 3, 4]
    _AUX_FAMILIES = ("offset", "coupled", "shared")

    def sample_support(rng: random.Random, items) -> list:
        items = list(items)
        return rng.sample(items, rng.randint(1, len(items)))

#: gold pins AND menu-distractor pins draw from this one prior — deriving it
#: from aux_required's gold prior is the matched-pin anti-leak invariant
SCALAR_PIN_PRIOR: Tuple[float, ...] = tuple(float(v) for v in _AUX_PIN)

#: THE reference solver. Single source of truth for gold certification (here),
#: the eval arms (run_invention_eval.py), the training reward
#: (train_structure_policy.py), and nonlinear end-to-end repair grading
#: (run_repair_ranker.py) — all of them import this dict, so a distractor is
#: only ever "unsolvable" under the same solver family that grades the arms.
REFERENCE_SOLVER = {"name": "lm", "k_refine": 4}

#: budget for the empirical unsolvability probe: n_seeds x k_refine restarts of
#: the REFERENCE_SOLVER (12 LM multistarts >= the 4-start grading budget, so a
#: certified-unsolvable distractor is vetted at least as hard as any eval arm
#: can try to solve it).
DEFAULT_PROBE = {"n_seeds": 3, "k_refine": 4}

#: family -> the exchangeable defining-expression support: every menu option
#: (gold AND distractors) instantiates the family template at one (a, delta)
#: combo from this table, so neither representation shape nor parameter
#: frequency can reveal the gold. Golds cycle the support uniformly by seed;
#: distractors draw from the same support and must be certified rootless.
#: Both templates are ONE-SIDED (a square / a sum of squares): substituting a
#: wrong-parameter defining relation forces "square = negative constant"
#: (quad_link) or an empty circle (vieta: eq1 becomes
#: (x+h)^2 + (y+h)^2 = R with R linear in delta, slope -1/a), so rootless
#: distractors exist on BOTH sides of every gold. A linear defining relation
#: (the old "u - (x-y) - d") cannot work: eliminating u leaves a line meeting a
#: hyperbola, which almost always intersects over the reals — those menus were
#: only ever "unsolvable" as probe artifacts.
#: quad_link (-1, 1) is excluded: u0 = -1 + 1 = 0 leaves fixed == augmented.
NONLINEAR_SUPPORTS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "vieta": tuple(
        (float(a), float(d)) for a in (1, -1) for d in range(-4, 5)
    ),
    "quad_link": tuple(
        (float(s), float(d)) for s in (1, -1) for d in range(5) if (s, d) != (-1, 1)
    ),
}


def nonlinear_expression(family: str, a: float, d: float) -> str:
    """The family's canonical defining-expression template at combo (a, d).
    Gold and distractors go through this one function so every menu option has
    the same syntactic shape."""
    if family == "vieta":
        return f"u - ({a})*(x**2 + y**2) - ({d})"
    if family == "quad_link":
        return f"u - ({a})*x**2 - ({d})"
    raise ValueError(f"unknown nonlinear family {family!r}")


@dataclass(frozen=True)
class Candidate:
    """One augmentation recipe: add latent ``aux_var`` with a defining factor and
    insert it (with the given coefficients) into a subset of fixed-graph factors.

    The defining factor is the scalar pin ``"u - (pin_value)"`` unless
    ``defining_expression`` is set, in which case that expression (which must
    mention ``aux_var``) is the defining factor — e.g. ``"u - (x - y)"`` couples
    the aux to the fixed variables instead of pinning it to a constant.
    ``pin_value`` is retained either way (= the aux value at the gold solution
    for gold candidates) so downstream features stay well-defined.
    """

    aux_var: str                      # always "u"
    pin_value: float                  # v in the defining factor "u - (v)"
    insert_coeffs: Dict[str, float]   # fixed-graph factor_id -> coefficient of u
    defining_expression: Optional[str] = None  # appended LAST: positional back-compat

    def __post_init__(self) -> None:
        if self.defining_expression is not None:
            names = {s.name for s in sp.sympify(self.defining_expression).free_symbols}
            if self.aux_var not in names:
                raise ValueError(
                    f"defining_expression {self.defining_expression!r} does not "
                    f"mention aux_var {self.aux_var!r}"
                )

    def apply(self, fixed: FactorGraph) -> FactorGraph:
        """Build the augmented graph. Pure — ``fixed`` is not mutated."""
        variables = [VariableNode(v.id, v.value) for v in fixed.variables]
        factors: List[FactorNode] = []
        edges = [Edge(e.variable_id, e.factor_id, e.coefficient) for e in fixed.edges]
        known = {f.id for f in fixed.factors}
        unknown = set(self.insert_coeffs) - known
        if unknown:
            raise ValueError(f"insert_coeffs reference unknown factors: {sorted(unknown)}")
        if "aux" in known or any(v.id == self.aux_var for v in fixed.variables):
            raise ValueError("fixed graph already contains an 'aux' factor or the aux variable")
        for f in fixed.factors:
            c = self.insert_coeffs.get(f.id)
            if c is None:
                factors.append(FactorNode(f.id, f.expression))
            else:
                factors.append(FactorNode(f.id, f"({f.expression}) + ({c})*{self.aux_var}"))
                edges.append(Edge(self.aux_var, f.id, float(c)))
        variables.append(VariableNode(self.aux_var))
        if self.defining_expression is None:
            factors.append(FactorNode("aux", f"{self.aux_var} - ({self.pin_value})"))
            edges.append(Edge(self.aux_var, "aux", 1.0))
        else:
            expr = sp.sympify(self.defining_expression)
            factors.append(FactorNode("aux", self.defining_expression))
            fixed_ids = {v.id for v in fixed.variables}
            for s in sorted(expr.free_symbols, key=lambda s: s.name):
                if s.name != self.aux_var and s.name not in fixed_ids:
                    continue
                d = sp.diff(expr, s)
                edges.append(Edge(s.name, "aux", float(d) if d.is_number else 1.0))
        return FactorGraph(variables=variables, factors=factors, edges=edges)


@dataclass
class InventionInstance:
    """A fixed (inconsistent) graph plus a K-candidate menu with one gold fix."""

    id: str
    family: str
    seed: int
    fixed_graph: FactorGraph
    candidates: List[Candidate]        # length K, gold at gold_idx (shuffled per seed)
    gold_idx: int
    aux_value: float                   # value of u in the augmented gold solution
    solution: Dict[str, float] = field(default_factory=dict)  # augmented gold incl. u
    certificate: str = "exact"         # "exact" (rank/CAS theorem) | "empirical" (probe)
    certificate_config: Optional[dict] = None  # probe budget + honest claim wording


def _is_inconsistent(graph: FactorGraph) -> bool:
    """Exact certificate that a *linear* factor graph has no solution.

    rank(A) < rank([A|b]) for the system A x = b read off the factor expressions.
    Raises (sympy NonlinearError, a ValueError) on nonlinear factors — callers that
    may see nonlinear graphs go through :func:`certify_unsolvable` instead.
    """
    # ponytail: self-contained ~10-line certificate; deliberately NOT imported from
    # marc/data/aux_required.py so this module works on main today.
    syms = [sp.Symbol(v.id) for v in graph.variables]
    exprs = [sp.sympify(f.expression) for f in graph.factors]
    A, b = sp.linear_eq_to_matrix(exprs, syms)
    return A.rank() < A.row_join(b).rank()


def _solvable_at_eval_grade(graph: FactorGraph, *, rng_seed: int) -> bool:
    """Best-of-k solve at the REFERENCE_SOLVER budget + Checker gate — the same
    grade run_invention_eval.py holds every arm to. Used for GOLD solvability."""
    solver = load_solver(REFERENCE_SOLVER["name"], seed=rng_seed)
    prob = SimpleNamespace(graph=graph, metadata={})
    cands = [c for c in solver.sample(prob, REFERENCE_SOLVER["k_refine"]) if c is not None]
    return Checker().first_accepted(graph, cands) is not None


def _real_solutions_exist(graph: FactorGraph) -> Optional[bool]:
    """Exact CAS decision: does the polynomial system have any real solution?

    ``None`` when sympy cannot decide (unsolved-for symbols in a solution,
    solver failure) — callers fall back to the empirical probe. The nonlinear
    menu families reduce to single quadratics under substitution, so this
    decides them exactly.
    """
    syms = [sp.Symbol(v.id) for v in graph.variables]
    exprs = [sp.sympify(f.expression) for f in graph.factors]
    try:
        sols = sp.solve(exprs, syms, dict=True)
    except Exception:
        return None
    if not sols:
        return False
    saw_undecided = False
    for sol in sols:
        vals = list(sol.values())
        if len(sol) < len(syms) or any(v.free_symbols for v in vals):
            saw_undecided = True  # positive-dimensional/parametric: not decided here
            continue
        try:
            if all(abs(complex(sp.N(v)).imag) < 1e-9 for v in vals):
                return True
        except Exception:
            saw_undecided = True
    return None if saw_undecided else False


def certify_unsolvable(
    graph: FactorGraph, *, rng_seed: int, probe: dict = DEFAULT_PROBE
) -> dict:
    """Certify a graph unsolvable; returns ``{"unsolvable": bool, "method": str}``.

    Linear graphs get the exact rank certificate (``method="linear_rank"`` — a
    theorem). Nonlinear graphs first get an exact CAS real-root decision
    (``method="cas_no_real_roots"``/``"cas_real_roots"`` — also a theorem: no
    solver at any budget can solve a system with no real roots). Only when the
    CAS cannot decide does the empirical probe run
    (``method="empirical_probe"``): ``n_seeds * k_refine`` REFERENCE_SOLVER
    restarts, any Checker-accepted candidate means solvable — an "unsolvable"
    probe verdict is budget-relative, and callers must surface that honestly
    (see ``InventionInstance.certificate``).
    """
    try:
        return {"unsolvable": _is_inconsistent(graph), "method": "linear_rank"}
    except (ValueError, sp.PolynomialError):
        pass  # nonlinear graph (sympy NonlinearError <= ValueError) -> CAS/probe
    real = _real_solutions_exist(graph)
    if real is not None:
        return {
            "unsolvable": not real,
            "method": "cas_no_real_roots" if not real else "cas_real_roots",
        }
    prob = SimpleNamespace(graph=graph, metadata={})
    checker = Checker()
    for s in range(probe["n_seeds"]):
        solver = load_solver(REFERENCE_SOLVER["name"], seed=rng_seed + s)
        cands = [c for c in solver.sample(prob, probe["k_refine"]) if c is not None]
        if checker.first_accepted(graph, cands) is not None:
            return {"unsolvable": False, "method": "empirical_probe"}
    return {"unsolvable": True, "method": "empirical_probe"}


def _candidate_key(c: Candidate) -> Tuple:
    return (
        float(c.pin_value),
        tuple(sorted(c.insert_coeffs.items())),
        c.defining_expression,
    )


def build_menu(
    fixed: FactorGraph,
    gold_candidate: Candidate,
    K: int,
    rng: random.Random,
    hard_negatives: bool = True,
    expression_support: Optional[Sequence[str]] = None,
) -> Tuple[List[Candidate], int]:
    """Assemble a K-candidate menu around the gold fix; returns (menu, gold_idx).

    Scalar-pin golds: distractors are random size-uniform factor subsets +
    coefficients from {-2,-1,1,2} + a pin from SCALAR_PIN_PRIOR, each CERTIFIED
    unsolvable via :func:`certify_unsolvable`; when ``hard_negatives``, one
    distractor shares the gold insert_coeffs with a wrong pin.

    Expression golds need ``expression_support`` — the family's full canonical
    template support (see NONLINEAR_SUPPORTS). Every option keeps the gold's
    insertion coefficients and the family's syntactic shape; distractors are
    certified unsolvable, and anything mathematically equal to the gold is
    excluded so the menu can never contain a second correct answer. All K-1
    distractors are coefficient-matched hard negatives by construction
    (``hard_negatives`` has no additional effect here).
    """
    if K < 2:
        raise ValueError("K must be >= 2 (gold + at least one distractor)")

    def unsolvable(cand: Candidate) -> bool:
        return certify_unsolvable(cand.apply(fixed), rng_seed=rng.randrange(2 ** 31))[
            "unsolvable"
        ]

    factor_ids = [f.id for f in fixed.factors]
    menu = [gold_candidate]
    seen = {_candidate_key(gold_candidate)}

    # Expression-defined nonlinear menus need an exchangeable candidate prior.
    # If fillers are scalar pins, or if their parameters come from a different
    # distribution than the gold's, a candidate-only classifier can spot the
    # gold from representation type or parameter frequency.  Here every option
    # instantiates the same family template over the same support with the same
    # insertion coefficients.  Only compatibility with ``fixed`` distinguishes
    # the gold.
    if gold_candidate.defining_expression is not None:
        if expression_support is None:
            raise ValueError(
                "expression-defined golds need expression_support (the family's "
                "canonical template support, see NONLINEAR_SUPPORTS)"
            )
        gold_expr = sp.sympify(gold_candidate.defining_expression)
        gold_delta = -float(gold_expr.subs({s: 0 for s in gold_expr.free_symbols}))
        options = []
        for expr_str in expression_support:
            expr = sp.sympify(expr_str)
            if sp.simplify(expr - gold_expr) == 0:
                continue  # mathematically the gold — a second correct answer
            options.append((expr_str, expr))
        rng.shuffle(options)
        for expr_str, expr in options:
            if len(menu) == K:
                break
            delta_c = -float(expr.subs({s: 0 for s in expr.free_symbols}))
            cand = Candidate(
                gold_candidate.aux_var,
                # pin is masked for expression candidates in every feature path;
                # keep it deterministic and derivable from the expression alone
                # (offset difference) so nothing about the gold leaks through it
                gold_candidate.pin_value + (delta_c - gold_delta),
                # Match coefficients exactly across the menu.  Gold coefficient
                # draws are conditioned by the generator's solvability gates;
                # resampling filler coefficients from the unconditional prior
                # makes that conditioning a candidate-only label leak.
                dict(gold_candidate.insert_coeffs),
                expr_str,
            )
            if _candidate_key(cand) in seen or not unsolvable(cand):
                continue
            menu.append(cand)
            seen.add(_candidate_key(cand))
        if len(menu) < K:
            raise RuntimeError(
                "could not fill the exchangeable expression menu with certified "
                "distractors from the family support"
            )
        rng.shuffle(menu)
        return menu, menu.index(gold_candidate)

    if hard_negatives:
        for _ in range(200):
            cand = Candidate(
                gold_candidate.aux_var,
                float(rng.choice(SCALAR_PIN_PRIOR)),
                dict(gold_candidate.insert_coeffs),
            )
            if _candidate_key(cand) in seen:
                continue
            if unsolvable(cand):
                menu.append(cand)
                seen.add(_candidate_key(cand))
                break
        else:
            raise RuntimeError("could not certify a hard negative in 200 draws")

    attempts = 0
    while len(menu) < K:
        attempts += 1
        if attempts > 500:
            raise RuntimeError("could not fill the menu with certified distractors")
        # size-uniform, matching the gold insertion-support prior — a Bernoulli
        # subset has a different size marginal, a candidate-only label leak
        subset = sample_support(rng, factor_ids)
        cand = Candidate(
            gold_candidate.aux_var,
            float(rng.choice(SCALAR_PIN_PRIOR)),
            {fid: float(rng.choice((-2, -1, 1, 2))) for fid in subset},
        )
        if _candidate_key(cand) in seen:
            continue
        if not unsolvable(cand):
            continue  # accidental-solvable draw -> resample
        menu.append(cand)
        seen.add(_candidate_key(cand))

    rng.shuffle(menu)
    return menu, menu.index(gold_candidate)


# --- toy-variant resampler (main-only data source) ----------------------------

#: family -> (variable names,
#:            base factors as (factor_id, {var: coeff}),
#:            ELIGIBLE factor ids the latent u may touch — the gold's actual
#:            touched set is a per-instance random nonempty subset of these
#:            (DATA_VERSION 2), so gold support is NOT family-constant and a
#:            support-matching shortcut cannot identify the gold candidate.
#:            toy2's eligible set covers all equations for the same reason: a
#:            singleton set would make its gold support constant again.
_TEMPLATES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, Dict[str, int]]], Tuple[str, ...]]] = {
    "toy1": (("x", "y"),
             [("eq1", {"x": 1, "y": 1}), ("eq2", {"x": 1, "y": -1}), ("eq3", {"x": 1})],
             ("eq1", "eq2", "eq3")),
    "toy2": (("x", "y", "z"),
             [("eq1", {"x": 1, "y": 1, "z": 1}), ("eq2", {"x": 1, "y": -1}),
              ("eq3", {"y": 1, "z": -1}), ("eq4", {"x": 1})],
             ("eq1", "eq2", "eq3", "eq4")),
    "toy3": (("x", "y"),
             [("eq1", {"x": 1}), ("eq2", {"y": 1}), ("eq3", {"x": 1, "y": 1})],
             ("eq1", "eq2")),
}

FAMILIES: Tuple[str, ...] = tuple(_TEMPLATES)

FAMILIES_BY_SOURCE: Dict[str, Tuple[str, ...]] = {
    "toys": FAMILIES,
    "aux_required": _AUX_FAMILIES,
    "nonlinear": ("vieta", "quad_link"),
}


def _toy_variant(family: str, seed: int) -> Tuple[FactorGraph, Candidate, Dict[str, float]]:
    """Resample one instance of a structure_toys pattern with fresh integer
    constants/coefficients. Returns (fixed_graph, gold_candidate, augmented solution).

    Constants are chosen so the augmented system holds exactly at an integer gold
    solution; the fixed graph (u terms dropped) is certified inconsistent. The
    gold's touched-factor set is a random nonempty subset of the template's
    eligible ids, drawn once per (family, seed) — support randomization
    (DATA_VERSION 2) so gold support is not a family fingerprint.
    """
    # ponytail: local constant-resampler; superseded by marc.data.aux_required when present
    if family not in _TEMPLATES:
        raise ValueError(f"unknown family {family!r} (expected one of {FAMILIES})")
    var_names, base_factors, eligible = _TEMPLATES[family]
    rng = random.Random(f"toy_variant:{family}:{seed}")
    touched = tuple(rng.sample(eligible, rng.randint(1, len(eligible))))
    for _ in range(200):
        sol = {v: float(rng.randint(-3, 3)) for v in var_names}
        u0 = float(rng.choice(SCALAR_PIN_PRIOR))  # u0 == 0 would leave fixed consistent
        coeffs = {fid: float(rng.choice((-2, -1, 1, 2))) for fid in touched}
        factors, edges = [], []
        for fid, terms in base_factors:
            const = -(sum(c * sol[v] for v, c in terms.items()) + coeffs.get(fid, 0.0) * u0)
            expr = " + ".join(f"({c})*{v}" for v, c in terms.items()) + f" + ({const})"
            factors.append(FactorNode(fid, expr))
            edges.extend(Edge(v, fid, float(c)) for v, c in terms.items())
        fixed = FactorGraph(
            variables=[VariableNode(v) for v in var_names], factors=factors, edges=edges
        )
        if not _is_inconsistent(fixed):
            continue
        # ponytail: keep the gold-augmented system inside the frozen baseline
        # solver's stability region. refine's polish (fixed lr=0.2) diverges when
        # lambda_max(A^T A) >= 10; resample stiff draws so gold_oracle (the
        # positive control) always solves. Ceiling: drop this filter if refine
        # ever gets a line search / adaptive lr.
        rows = [
            [float(terms.get(v, 0)) for v in var_names] + [coeffs.get(fid, 0.0)]
            for fid, terms in base_factors
        ]
        rows.append([0.0] * len(var_names) + [1.0])  # the aux-defining factor
        A = np.asarray(rows)
        if float(np.linalg.eigvalsh(A.T @ A)[-1]) >= 9.0:
            continue
        gold = Candidate(aux_var="u", pin_value=u0, insert_coeffs=coeffs)
        solution = dict(sol)
        solution["u"] = u0
        return fixed, gold, solution
    raise RuntimeError(f"could not sample an inconsistent {family} variant (seed={seed})")


# --- nonlinear aux-required families (source "nonlinear") ---------------------
#
# Built augmented-first, like _toy_variant: pick an integer gold solution, derive
# the constants so the augmented system holds exactly, then DROP the c*u terms to
# get the fixed graph. The rank certificate does not apply to these graphs:
#   fixed graph  — must be certified unsolvable (CAS no-real-roots theorem when
#                  decidable, DEFAULT_PROBE otherwise)
#   gold-applied — must SOLVE at eval grade (REFERENCE_SOLVER, best of k_refine)
#                  under two independent seeds
# Draws failing either check are resampled.

_NONZERO = (-4, -3, -2, -1, 1, 2, 3, 4)


def _nonlinear_variant(family: str, seed: int) -> Tuple[FactorGraph, Candidate, Dict[str, float]]:
    """Resample one nonlinear aux-required instance; returns
    (fixed_graph, gold_candidate, augmented solution incl. u)."""
    nl_fams = FAMILIES_BY_SOURCE["nonlinear"]
    if family not in nl_fams:
        raise ValueError(f"unknown nonlinear family {family!r} (expected one of {nl_fams})")
    rng = random.Random(f"nonlinear_variant:{family}:{seed}")
    # Fix the offset before the rejection loop.  Drawing it inside the loop makes
    # offsets with larger feasible regions appear more often, a label leak to a
    # candidate-only model.  Consecutive seeds cycle uniformly through each
    # family's empirically feasible support (gcd(2, |support|)=1 for alternating
    # two-family datasets, so each family remains balanced).
    support = NONLINEAR_SUPPORTS[family]
    a_g, delta = support[int(seed) % len(support)]
    # 1000 attempts: the cheap constant filters reject most draws; only
    # survivors pay for the CAS check + eval-grade solve gates. (The old
    # lambda_max stability filter is gone: it protected the frozen refine
    # polish's fixed lr, and the LM reference solver has no such bound — its
    # removal is what un-degenerates the gold coefficient distribution.)
    for _ in range(1000):
        c1 = float(rng.choice((-2, -1, 1, 2)))
        c2 = float(rng.choice((-2, -1, 1, 2)))
        if family == "vieta":
            # augmented: x + y + c1*u = k1, x*y + c2*u = k2,
            # u = a*(x**2 + y**2) + delta, with (a, delta) cycled from the support.
            x_star = float(rng.choice(_NONZERO))
            y_star = float(rng.choice(_NONZERO))
            u0 = a_g * (x_star ** 2 + y_star ** 2) + delta
            if u0 == 0:
                continue  # fixed would equal augmented
            k1 = x_star + y_star + c1 * u0
            k2 = x_star * y_star + c2 * u0
            if k1 ** 2 - 4 * k2 >= 0:
                continue  # fixed x+y=k1, x*y=k2 has real roots -> not aux-required
            exprs = [("eq1", f"x + y - ({k1})", {"x": 1, "y": 1}),
                     ("eq2", f"x*y - ({k2})", {"x": 1, "y": 1})]
            sol = {"x": x_star, "y": y_star}
        else:  # quad_link
            # augmented: 0.5*x**2 + 0.5*y + c1*u = k1,
            # 0.5*x**2 - 0.5*y + c2*u = k2, u = a*x**2 + delta.
            x_star = float(rng.choice((-1, 1)))
            y_star = float(rng.randint(-6, 6))
            u0 = a_g * x_star ** 2 + delta
            if u0 == 0:
                continue
            x_sq = x_star ** 2
            k1 = 0.5 * x_sq + 0.5 * y_star + c1 * u0
            k2 = 0.5 * x_sq - 0.5 * y_star + c2 * u0
            if k1 + k2 >= 0:
                continue  # fixed has real roots (0.5*x**2 = (k1+k2)/2)
            exprs = [("eq1", f"0.5*x**2 + 0.5*y - ({k1})", {"x": 1, "y": 0.5}),
                     ("eq2", f"0.5*x**2 - 0.5*y - ({k2})", {"x": 1, "y": -0.5})]
            sol = {"x": x_star, "y": y_star}
        defining = nonlinear_expression(family, a_g, delta)
        fixed = FactorGraph(
            variables=[VariableNode(v) for v in ("x", "y")],
            factors=[FactorNode(fid, e) for fid, e, _ in exprs],
            edges=[Edge(v, fid, float(c)) for fid, _, terms in exprs for v, c in terms.items()],
        )
        gold = Candidate("u", u0, {"eq1": c1, "eq2": c2}, defining)
        solution = dict(sol)
        solution["u"] = u0
        aug = gold.apply(fixed)
        if not certify_unsolvable(fixed, rng_seed=rng.randrange(2 ** 31))["unsolvable"]:
            continue  # fixed graph has a real root -> not aux-required
        # gold must be RELIABLY reachable at eval grade: multistart LM is seed-
        # stochastic, so demand two independent successes or the e2e oracle
        # ceiling wobbles with the evaluation's restart stream
        gate_seed = rng.randrange(2 ** 31)
        if not (_solvable_at_eval_grade(aug, rng_seed=gate_seed)
                and _solvable_at_eval_grade(aug, rng_seed=gate_seed + 1000003)):
            continue  # reference solver can't reach the gold basin -> resample
        return fixed, gold, solution
    raise RuntimeError(f"could not sample a certified {family} variant (seed={seed})")


def _pin_from_defining_expression(expression: str, aux_var: str) -> float:
    """v from a defining factor 'u - (v)': the expression at u=0 is -v."""
    expr = sp.sympify(expression)
    return float(-expr.subs(sp.Symbol(aux_var), 0))


def make_dataset(
    source: str,
    n: int,
    seed: int,
    K: int = 4,
    families: Optional[Sequence[str]] = None,
    hard_negatives: bool = True,
) -> List[InventionInstance]:
    """Build ``n`` menu-based invention instances.

    ``source="toys"`` uses the local toy-variant resampler (works on main today).
    ``source="aux_required"`` consumes ``marc.data.aux_required`` (contract C3) when
    present; raises an actionable ImportError otherwise. ``source="nonlinear"``
    uses the local nonlinear resampler (families ``vieta``/``quad_link``); those
    instances carry ``certificate="exact"`` when every distractor has a CAS
    no-real-roots proof and ``certificate="empirical"`` otherwise — see the
    module docstring. Deterministic per
    ``(source, n, seed, K, families, hard_negatives)``.
    """
    fams = tuple(families) if families else FAMILIES_BY_SOURCE.get(source, FAMILIES)
    raw: List[Tuple[str, int, FactorGraph, Candidate, Dict[str, float]]] = []
    if source == "toys":
        for i in range(n):
            family = fams[i % len(fams)]
            inst_seed = seed + i
            fixed, gold, solution = _toy_variant(family, inst_seed)
            raw.append((family, inst_seed, fixed, gold, solution))
    elif source == "nonlinear":
        for i in range(n):
            family = fams[i % len(fams)]
            inst_seed = seed + i
            fixed, gold, solution = _nonlinear_variant(family, inst_seed)
            raw.append((family, inst_seed, fixed, gold, solution))
    elif source == "aux_required":
        try:
            from marc.data import aux_required
        except ImportError as exc:
            raise ImportError(
                "make_dataset(source='aux_required') needs marc/data/aux_required.py "
                "(contract C3: generate_instances(n, seed, patterns)), which is not on "
                "this branch yet. Use source='toys' (the local toy-variant resampler) "
                "or merge the aux_required data unit first."
            ) from exc
        for r in aux_required.generate_instances(n, seed, patterns=list(fams)):
            gold = Candidate(
                aux_var=r.aux_var,
                pin_value=_pin_from_defining_expression(r.defining_expression, r.aux_var),
                insert_coeffs={k: float(v) for k, v in r.insert_coeffs.items()},
            )
            raw.append((r.pattern, r.seed, r.fixed_graph, gold, dict(r.solution)))
    else:
        raise ValueError(f"unknown source {source!r} (expected one of {SOURCES})")

    n_restarts = DEFAULT_PROBE["n_seeds"] * DEFAULT_PROBE["k_refine"]
    empirical_config = {
        **DEFAULT_PROBE,
        "solver": REFERENCE_SOLVER["name"],
        "claim": (
            f"at least one distractor is only probe-certified: it failed a "
            f"{n_restarts}-restart {REFERENCE_SOLVER['name']}+Checker probe; "
            "'exactly one solvable option' is an empirical claim at this probe "
            "budget, not a theorem"
        ),
    }
    exact_nl_config = {
        "method": "cas_no_real_roots",
        "claim": (
            "every distractor-augmented system is CAS-certified to have no real "
            "solution; 'exactly one solvable option' is exact for this instance"
        ),
    }

    out: List[InventionInstance] = []
    for family, inst_seed, fixed, gold, solution in raw:
        # A variant whose feasible-offset set is too small to certify K-1
        # distractors under the reference solver cannot fill its menu; resample
        # the variant deterministically far outside the base seed range.
        support = (
            [nonlinear_expression(family, a, d) for a, d in NONLINEAR_SUPPORTS[family]]
            if source == "nonlinear" else None
        )
        for _retry in range(20):
            rng = random.Random(f"menu:{family}:{inst_seed}:{K}:{hard_negatives}")
            try:
                menu, gold_idx = build_menu(fixed, gold, K, rng,
                                            hard_negatives=hard_negatives,
                                            expression_support=support)
                break
            except RuntimeError:
                if source != "nonlinear":
                    raise
                inst_seed += 9999991
                fixed, gold, solution = _nonlinear_variant(family, inst_seed)
        else:
            raise RuntimeError(
                f"could not fill a certified {family} menu after 20 variant resamples"
            )
        if source == "nonlinear":
            # per-instance certificate: exact iff every distractor carries the
            # CAS no-real-roots proof (re-deriving it is cheap and deterministic)
            methods = {
                certify_unsolvable(c.apply(fixed), rng_seed=inst_seed)["method"]
                for j, c in enumerate(menu) if j != gold_idx
            }
            if methods == {"cas_no_real_roots"}:
                certificate, certificate_config = "exact", exact_nl_config
            else:
                certificate, certificate_config = "empirical", empirical_config
        else:
            certificate, certificate_config = "exact", None
        out.append(
            InventionInstance(
                id=f"{source}_{family}_s{inst_seed}",
                family=family,
                seed=inst_seed,
                fixed_graph=fixed,
                candidates=menu,
                gold_idx=gold_idx,
                aux_value=float(solution[gold.aux_var]),
                solution=solution,
                certificate=certificate,
                certificate_config=dict(certificate_config) if certificate_config else None,
            )
        )
    return out


def to_padded(instance: InventionInstance) -> PaddedGraph:
    """Clean state c_0 over 2K slots: candidate j owns var slot 2j + factor slot 2j+1.

    Gold's var slot is VARIABLE carrying ``aux_value``; gold's factor slot is FACTOR
    carrying the pin value; every other slot is ABSENT/0.
    """
    K = len(instance.candidates)
    types = torch.full((2 * K,), ABSENT, dtype=torch.long)
    values = torch.zeros(2 * K, dtype=torch.float32)
    g = instance.gold_idx
    types[2 * g] = int(SlotType.VARIABLE)
    values[2 * g] = instance.aux_value
    types[2 * g + 1] = int(SlotType.FACTOR)
    # An expression-defined factor has no scalar pin to regress.  Exposing the
    # gold aux value here leaks a target-dependent statistic to the menu policy.
    values[2 * g + 1] = (
        instance.candidates[g].pin_value
        if instance.candidates[g].defining_expression is None else 0.0
    )
    return PaddedGraph(types, values)
