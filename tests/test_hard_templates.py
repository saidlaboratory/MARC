"""Hard non-convex templates (A1 de-saturation tier).

Verifies the bilinear families are well-formed (true root accepted by the checker)
and that they are genuinely hard: deterministic gradient descent from a cold start is
trapped (0% solved), which is what pulls the eval suite off its 1.000 ceiling."""
from marc.cas.checker import Checker
from marc.data.templates import (
    BilinearSystemTemplate, BilinearProductTemplate, HARD_TEMPLATES, HARD_TEMPLATES_EXT,
)
from marc.eval.metrics import wilson_interval
from marc.refine.iterative import refine


def test_hard_templates_true_root_accepted():
    chk = Checker()
    for T in HARD_TEMPLATES_EXT:  # all 4 non-convex families
        for seed in range(5):
            g, sol = T.generate(seed=seed)
            assert chk.verify(g, list(sol.values())).accepted, f"{T.name} seed {seed}"


def test_wilson_interval_basic():
    lo, hi = wilson_interval(20, 40)  # p=0.5
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo0, hi0 = wilson_interval(0, 40)  # p=0 stays in [0,1)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.2
    # more data -> tighter interval
    assert (wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0]) < (hi - lo)


def test_bilinear_traps_cold_start_descent():
    # deterministic descent from zero is trapped on the bilinear family -> this is
    # what de-saturates the suite (convex linear systems would solve ~always).
    chk = Checker()
    T = BilinearSystemTemplate()
    solved = 0
    for seed in range(20):
        g, sol = T.generate(seed=seed)
        tr = refine(g, [0.0] * len(sol), noise=False, seed=0)
        solved += chk.verify(g, tr.x).accepted
    assert solved <= 3, f"expected cold-start descent to be mostly trapped, solved {solved}/20"


def test_shapes():
    g2, s2 = BilinearSystemTemplate().generate(seed=0)
    g3, s3 = BilinearProductTemplate().generate(seed=0)
    assert len(g2.variables) == 2 and len(g2.factors) == 2
    assert len(g3.variables) == 3 and len(g3.factors) == 3
