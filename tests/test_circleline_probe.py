"""Property tests for the CircleLine symmetry-breaking probe.

Fast: no training. Covers the canonicalization helper (target ordering follows
the tag, both orderings stay checker-feasible) and the index-feature
construction (the tag makes the otherwise-tied x/y outputs distinct). The full
measurement is produced by ``scripts/circleline_symmetry_probe.py``."""
import torch

from marc.cas.checker import Checker

from conftest import load_script

sp = load_script("circleline_symmetry_probe")


def test_canon_target_orders_by_tag():
    chk = Checker()
    for seed in range(8):
        g, sol = sp.dc.gen(sp.dc.TEMPLATE, 1, seed0=seed)[0]
        asc = sp.canon_target(sol, sp.INDEX_TAG)
        assert asc[0] < asc[1]
        assert sorted(asc) == sorted(sol)
        desc = sp.canon_target(sol, -sp.INDEX_TAG)
        assert desc == [asc[1], asc[0]]
        # both orderings are real roots, so canonicalization never leaves the manifold
        assert chk.verify(g, asc).accepted
        assert chk.verify(g, desc).accepted


def test_index_tag_breaks_output_tie():
    g, _ = sp.dc.gen(sp.dc.TEMPLATE, 1, seed0=3)[0]
    data = sp.build_heterodata(g)
    data["variable"].x = torch.zeros(2, 1)  # identical inputs => the tie at issue
    t = torch.tensor([sp.dc.T])

    torch.manual_seed(0)
    plain = sp.GraphDenoiser(D=32, L=2).eval()
    with torch.no_grad():
        tied = plain(data, t)
    assert torch.allclose(tied[0], tied[1])

    torch.manual_seed(0)
    net = sp.tagged_net(D=32, L=2).eval()
    net.var_encoder.tag = sp.INDEX_TAG
    with torch.no_grad():
        out = net(data, t)
    assert not torch.allclose(out[0], out[1], atol=1e-6)
