"""Phase-spec wiring checks for the overnight orchestrator (no subprocesses).

build_phases only touches manifest/state inside lazy lambdas, so passing
manifest=None keeps this side-effect free (a real Manifest would rotate
results/overnight/MANIFEST.json on construction)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_overnight",
    Path(__file__).resolve().parent.parent / "scripts" / "run_overnight.py",
)
ov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ov)

CKPT_EVALS = ["eval_hard", "eval_coupled", "eval_dimension_scaling", "eval_crossfamily"]


def _phases(smoke=False):
    return ov.build_phases(smoke, manifest=None, state={"invention_data": None})


def test_phase_names_match_order():
    for smoke in (False, True):
        assert [p["name"] for p in _phases(smoke)] == ov.PHASE_ORDER


def test_differentiating_evals_want_ckpt_but_dont_require_it():
    by_name = {p["name"]: p for p in _phases()}
    for name in CKPT_EVALS:
        spec = by_name[name]
        assert spec.get("wants_ckpt") is True
        # soft: no hard skip when the checkpoint is missing
        assert not spec.get("needs_ckpt")


def test_slow_eval_timeouts_capped():
    for name in ("eval_main_learned", "eval_hard", "eval_coupled",
                 "eval_dimension_scaling"):
        assert ov.TIMEOUTS[name] == 90 * 60
        assert ov.TIMEOUTS[name] < ov.DEFAULT_TIMEOUT
