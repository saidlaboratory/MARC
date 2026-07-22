"""Shared test helpers: scripts/ isn't a package, so load script modules by path."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


class StubSolver:
    """Scripted learned-arm stand-in; propose(problem, k, call_no) -> candidates."""

    def __init__(self, propose):
        self.propose = propose
        self.calls = 0

    def sample(self, problem, k):
        self.calls += 1
        return self.propose(problem, k, self.calls)


def diverge_then_zeros(problem, k, call):
    # first call diverges (None candidate), the rest propose zeros
    nv = len(problem.graph.variables)
    return [None if call == 1 else [0.0] * nv for _ in range(k)]


def patch_load_solver(monkeypatch, module, stub):
    seen = {}

    def fake_load(name, **kw):
        seen.update(kw, name=name)
        return stub

    monkeypatch.setattr(module, "load_solver", fake_load)
    return seen
