"""
Two-stage accept/reject gate (TECHNICAL_GUIDE §7, §14).

Stage 1 numeric: fast reject if any factor's violation > tol.
Stage 2 symbolic: snap values to exact rationals (within snap_tol) and re-check
exactly — the authoritative gate that catches false-accepts the numeric tolerance
lets through.

Factor expressions carry their own constraint type (§2.3): a bare expression g is
the equality g == 0; a relational ("g <= 0", "g >= 0", "Eq(a, b)") is an
inequality/equality. Inequalities are treated as non-strict (boundary accepted).
The per-factor violation is |g| for equalities and the hinge max(0, g) for
inequalities, so it is 0 exactly when the constraint holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import List, Sequence, Tuple

import sympy as sp


@lru_cache(maxsize=8192)
def _parse(expr_str: str) -> sp.Expr:
    return sp.sympify(expr_str)


def _residual_and_kind(expr: sp.Expr) -> Tuple[sp.Expr, str]:
    """Reduce a factor to (g, kind): 'eq' means g == 0, 'le' means g <= 0."""
    if isinstance(expr, sp.Equality):
        return expr.lhs - expr.rhs, "eq"
    if isinstance(expr, (sp.LessThan, sp.StrictLessThan)):
        return expr.lhs - expr.rhs, "le"
    if isinstance(expr, (sp.GreaterThan, sp.StrictGreaterThan)):
        return expr.rhs - expr.lhs, "le"
    return expr, "eq"


@dataclass
class CheckResult:
    accepted: bool
    failed_factors: List[str]
    max_residual: float
    stage: str = ""  # "", "numeric", or "symbolic"

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "failed_factors": list(self.failed_factors),
            "max_residual": self.max_residual,
            "stage": self.stage,
        }


class Checker:
    def __init__(
        self,
        tol: float = 1e-6,
        snap_tol: float = 1e-9,
        max_denominator: int = 10 ** 9,
    ) -> None:
        if snap_tol > tol:
            raise ValueError("snap_tol must be <= tol for the symbolic gate to be stricter")
        self.tol = tol
        self.snap_tol = snap_tol
        self.max_denominator = max_denominator

    def verify(self, G, x: Sequence[float] | None = None) -> CheckResult:
        """Accept/reject the assignment x against factor graph G."""
        symbols, factors, values = self._unpack(G, x)
        return self.check(symbols, factors, values)

    def accepts(self, G, x: Sequence[float] | None = None) -> bool:
        return self.verify(G, x).accepted

    def first_accepted(self, G, candidates: Sequence[Sequence[float]]):
        """Return (index, CheckResult) of the first accepted candidate, else None (pass@k)."""
        for i, x in enumerate(candidates):
            result = self.verify(G, x)
            if result.accepted:
                return i, result
        return None

    def check(
        self,
        symbols: Sequence[sp.Symbol],
        factors: Sequence[Tuple[str, sp.Expr]],
        x: Sequence[float],
    ) -> CheckResult:
        """Core gate over pre-parsed (symbol, expr) data; shared by verify() and CASEngine."""
        known = set(symbols)
        for fid, expr in factors:
            missing = expr.free_symbols - known
            if missing:
                raise ValueError(f"factor {fid} references unknown variables: {missing}")

        numeric = self._numeric_violations(symbols, factors, x)
        # guard finiteness: max(0, nan) and abs(nan) are not > tol, so NaN/inf slip through
        all_finite = all(math.isfinite(v) for _, v in numeric)
        max_res = max((v for _, v in numeric), default=0.0) if all_finite else math.inf

        numeric_failed = [
            fid for fid, v in numeric if not math.isfinite(v) or v > self.tol
        ]
        if numeric_failed:
            return CheckResult(False, numeric_failed, max_res, stage="numeric")

        symbolic_failed = self._symbolic_failed(symbols, factors, x)
        if symbolic_failed:
            return CheckResult(False, symbolic_failed, max_res, stage="symbolic")

        return CheckResult(True, [], max_res, stage="")

    def explain_rejection(self, G, x: Sequence[float] | None = None) -> str:
        """Per-factor breakdown for RL debug logs."""
        symbols, factors, values = self._unpack(G, x)
        result = self.check(symbols, factors, values)

        assignment = ", ".join(f"{s}={v:.10g}" for s, v in zip(symbols, values))
        if result.accepted:
            header = f"ACCEPTED  max|r|={result.max_residual:.3e}"
        else:
            header = (
                f"REJECTED [{result.stage}]  "
                f"{len(result.failed_factors)}/{len(factors)} factors failed  "
                f"max|r|={result.max_residual:.3e}"
            )

        lines = [header, f"  assignment: {assignment or '(none)'}"]
        violations = dict(self._numeric_violations(symbols, factors, x=values))
        for fid, expr in factors:
            mark = "FAIL" if fid in result.failed_factors else "ok"
            lines.append(f"  [{mark:>4}] {fid}: {expr}  ->  violation {violations[fid]:.3e}")
        return "\n".join(lines)

    def _unpack(self, G, x):
        symbols = [sp.Symbol(v.id) for v in G.variables]
        factors = [(f.id, _parse(f.expression)) for f in G.factors]
        values = list(G.get_values()) if x is None else list(x)
        if len(values) != len(symbols):
            raise ValueError(
                f"assignment has {len(values)} values but graph has {len(symbols)} variables"
            )
        return symbols, factors, values

    def _numeric_violations(self, symbols, factors, x) -> List[Tuple[str, float]]:
        subs = {s: float(v) for s, v in zip(symbols, x)}
        out = []
        for fid, expr in factors:
            g, kind = _residual_and_kind(expr)
            try:
                r = float(g.subs(subs))
            except TypeError:
                # non-real residual (e.g. sqrt of a negative): constraint can't hold
                out.append((fid, math.inf))
                continue
            out.append((fid, abs(r) if kind == "eq" else max(0.0, r)))
        return out

    def _symbolic_failed(self, symbols, factors, x) -> List[str]:
        subs = {s: self._to_exact(float(v)) for s, v in zip(symbols, x)}
        failed = []
        for fid, expr in factors:
            g, kind = _residual_and_kind(expr)
            val = g.subs(subs)
            if kind == "eq":
                # fast path covers polynomials; confirm a seemingly-nonzero value with
                # simplify so radicals (e.g. sqrt(8)-2*sqrt(2)) aren't false-rejected
                bad = val != 0 and sp.simplify(val) != 0
            else:
                bad = sp.simplify(val) > 0
            if bad:
                failed.append(fid)
        return failed

    def _to_exact(self, val: float) -> sp.Rational:
        frac = Fraction(val).limit_denominator(self.max_denominator)
        if abs(float(frac) - val) <= self.snap_tol:
            return sp.Rational(frac.numerator, frac.denominator)
        return sp.Rational(Fraction(val))
