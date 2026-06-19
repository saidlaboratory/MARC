from dataclasses import dataclass
from typing import List
import sympy as sp
from marc.cas.engine import CASEngine


@dataclass
class CheckResult:
    accepted: bool
    gate: str  # "numeric", "symbolic", or "none"
    explanation: str


class Checker:
    """Two-gate checker for factor graph solutions.

    Gate 1 (fast): numeric — max |r_i| <= tol
    Gate 2 (exact): symbolic — substitute rational values into SymPy expressions

    check() runs numeric first; if it passes, runs symbolic.
    A solution is accepted only if BOTH gates pass (or only symbolic if numeric
    tolerance is generous).
    """

    def __init__(self, cas_engine: CASEngine, sympy_exprs: List[sp.Expr]):
        """
        Args:
            cas_engine: existing CASEngine for fast numeric residual eval
            sympy_exprs: list of SymPy residual expressions (e.g. [x+y-3, x-y-1])
                         These are the symbolic forms for exact checking.
        """
        self.cas = cas_engine
        self.exprs = sympy_exprs
        # Pre-compute sorted symbol list once; exprs are immutable after construction.
        self._symbols = sorted(
            set().union(*[e.free_symbols for e in sympy_exprs]), key=str
        )

    def check_numeric(self, x_vals: list, tol: float = 1e-6) -> CheckResult:
        """Numeric gate: accept iff max|r_i| <= tol."""
        residuals = self.cas.residuals(x_vals)
        max_res = max(abs(r) for r in residuals)
        if max_res <= tol:
            return CheckResult(True, "numeric", f"max|r|={max_res:.2e} <= tol={tol:.2e}")
        return CheckResult(False, "none", f"numeric failed: max|r|={max_res:.2e} > tol={tol:.2e}")

    def check_symbolic(self, x_vals: list) -> CheckResult:
        """Symbolic gate: substitute rational approximations and verify exact zeros.

        Converts float values to sympy.Rational (via nsimplify) then substitutes
        into each expression and checks == 0 exactly.
        """
        if len(self._symbols) != len(x_vals):
            return CheckResult(
                False,
                "none",
                f"symbol count mismatch: {len(self._symbols)} symbols, {len(x_vals)} values",
            )

        subs = {
            sym: sp.nsimplify(val, rational=True, tolerance=1e-9)
            for sym, val in zip(self._symbols, x_vals)
        }

        for i, expr in enumerate(self.exprs):
            # After rational substitution the result is a Rational; no need for sp.simplify.
            result = expr.subs(subs)
            if result != 0:
                return CheckResult(
                    False,
                    "none",
                    f"symbolic failed: expr[{i}] = {result} != 0",
                )

        return CheckResult(True, "symbolic", "all expressions evaluate to 0 symbolically")

    def check(self, x_vals: list, tol: float = 1e-6) -> CheckResult:
        """Run numeric gate first, then symbolic if numeric passes.

        Returns the symbolic result when numeric passes, or the numeric
        failure result otherwise. This is the strictest / most conservative mode.
        """
        numeric_result = self.check_numeric(x_vals, tol)
        if not numeric_result.accepted:
            return numeric_result
        return self.check_symbolic(x_vals)
