"""Hard non-convex templates (A1 de-saturation tier).

Verifies the bilinear families are well-formed (true root accepted by the checker)
and that they are genuinely hard: deterministic gradient descent from a cold start is
trapped (0% solved), which is what pulls the eval suite off its 1.000 ceiling."""
from marc.cas.checker import Checker
from marc.data.templates import BilinearSystemTemplate, BilinearProductTemplate, HARD_TEMPLATES
from marc.refine.iterative import refine


def test_hard_templates_true_root_accepted():
    chk = Checker()
    for T in HARD_TEMPLATES:
        for seed in range(5):
            g, sol = T.generate(seed=seed)
            assert chk.verify(g, list(sol.values())).accepted, f"{T.name} seed {seed}"


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
