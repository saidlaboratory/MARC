"""Property tests for the CircleLine failure-mode diagnostic.

Fast: no training. Verifies the root helper returns checker-accepted points for
both roots (the x<->y swap is exact) and that the chord midpoint carries positive
energy — i.e. the conditional mean of the two roots is off the solution manifold.
The full measurement is produced by ``scripts/diagnose_circleline.py``."""
from marc.cas.checker import Checker

from conftest import load_script

dc = load_script("diagnose_circleline")


def test_both_roots_satisfy_both_constraints():
    chk = Checker()
    for seed in range(8):
        g, sol = dc.gen(dc.TEMPLATE, 1, seed0=seed)[0]
        roots = dc.true_roots(sol)
        assert len(roots) == 2 and roots[0] != roots[1]
        for root in roots:
            assert chk.verify(g, list(root)).accepted


def test_chord_midpoint_energy_positive():
    # E(midpoint) = (x*-y*)^4 / 8 >= 1/8 since the template forces x* != y*
    # (integer solutions), so the mean of the two roots is never feasible
    for seed in range(8):
        g, sol = dc.gen(dc.TEMPLATE, 1, seed0=seed)[0]
        mid = dc.chord_midpoint(sol)
        e = dc.energy_at(g, mid)
        assert e > 0.1
        assert abs(e - (sol[0] - sol[1]) ** 4 / 8) < 1e-9
