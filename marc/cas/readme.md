### Usage

from marc.cas.engine import CASEngine
engine = CASEngine('two_equations.json', 'x1 x2')
energy = engine.energy([1.0, 0.0])
engine.accepts([2.0, 1.0])      # bool, via the Checker
engine.verify([2.0, 1.0])       # full CheckResult

### Checker

`Checker.verify(G, x)` is the accept/reject gate over a `FactorGraph`. Two stages:

1. numeric — fast reject if any factor violation > `tol`
2. symbolic — snap values to exact rationals (within `snap_tol`) and re-check
   exactly; this catches false-accepts the numeric tolerance lets through

Factor expressions carry their constraint type: a bare expression `g` means
`g == 0`; relationals (`"g <= 0"`, `"x >= 1"`, `"Eq(a, b)"`) are checked as
written. `CheckResult` reports `accepted`, `failed_factors`, `max_residual`, and
`stage`.

    from marc.cas.checker import Checker
    Checker().verify(graph, [2.0, 1.0])
