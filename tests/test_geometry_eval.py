"""Schema tests for the geometry eval's three arms (refine / random / learned).

Fast: stubbed solvers and a stubbed refine(), so no real descent runs. The real
numbers come from ``scripts/run_geometry_eval.py`` (too heavy for the unit suite)."""
from conftest import load_script

geo = load_script("run_geometry_eval")


class _OracleSolver:
    """Returns the known solution — every problem accepted."""
    def sample(self, problem, k):
        return [list(problem.solution) for _ in range(k)]


class _LearnedStub:
    """Alternates a diverged (None) rollout with the true solution, so the
    None-candidate guard in learned_arm is exercised."""
    def __init__(self):
        self.calls = 0

    def sample(self, problem, k):
        self.calls += 1
        return [None] if self.calls % 2 else [list(problem.solution)]


def _echo_refine(graph, x0, **kw):
    class T:
        x = list(x0)
    return T()


def _payload(monkeypatch, learned=None):
    # random arm's uniform inits never hit the exact checker; keep it instant
    monkeypatch.setattr(geo, "refine", _echo_refine)
    idp = geo.geometry_in_distribution(n=2)
    hop = geo.geometry_held_out(n=2)
    return geo.build_payload(idp, hop, 2, refine_solver=_OracleSolver(),
                             learned_solver=learned, ckpt="stub.pt" if learned else None)


def _check_cell(cell):
    assert 0 <= cell["k"] <= cell["n"]
    lo, hi = cell["ci95"]
    assert 0.0 <= lo <= cell["rate"] <= hi <= 1.0


def test_schema_and_skipped_learned(monkeypatch):
    payload = _payload(monkeypatch)
    # legacy run_split_eval schema intact
    assert payload["solver"] == "refine"
    assert set(payload["splits"]) == {"in_distribution", "held_out_structure"}
    assert "generalization_gap" in payload and "per_problem" in payload
    # arms: refine solves everything (oracle), random solves nothing (echo)
    for split in ("in_distribution", "held_out_structure", "pooled"):
        _check_cell(payload["arms"]["refine"][split])
        _check_cell(payload["arms"]["random"][split])
    pooled = payload["arms"]["refine"]["pooled"]
    assert (pooled["k"], pooled["n"], pooled["rate"]) == (4, 4, 1.0)
    assert payload["arms"]["random"]["pooled"]["k"] == 0
    for arm in ("refine", "random"):
        assert payload["arms"][arm]["wall_ms_total"] >= 0.0
        assert payload["arms"][arm]["wall_ms_mean"] >= 0.0
    assert payload["arms"]["learned"]["status"] == "skipped"
    assert "reason" in payload["arms"]["learned"]
    # a skipped arm burned no compute, so it carries no timing fields
    assert "wall_ms_total" not in payload["arms"]["learned"]
    assert payload["learned_vs_random"] is None


def test_learned_arm_and_ztest(monkeypatch):
    payload = _payload(monkeypatch, learned=_LearnedStub())
    learned = payload["arms"]["learned"]
    assert learned["status"] == "ok"
    assert learned["checkpoint"] == "stub.pt"
    assert learned["wall_ms_total"] >= 0.0
    assert learned["wall_ms_mean"] >= 0.0
    # with K=2 the second (solution) rollout rescues every problem
    assert learned["pooled"]["k"] == learned["pooled"]["n"] == 4
    lv = payload["learned_vs_random"]
    assert lv["z"] > 0
    assert 0.0 <= lv["p_one_sided"] <= 0.5
