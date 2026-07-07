"""Tests for the P2 reward interface (marc/train/reward.py).

These use *fake* trajectory dicts (no policy/model needed) plus a real CASEngine and
conservative Checker built from the shipped two-equation example, so they run without
torch_geometric. Solution of {x+y-3=0, x-y-1=0} is (x, y) = (2, 1).
"""

import os

import pytest

from marc.cas.engine import CASEngine
from marc.graph.serialize import load_graph
from marc.cas.checker import Checker
from marc.train.reward import (
    TERMINAL_B,
    compute_reward,
    shaping_reward,
    terminal_reward,
)

_EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "marc", "data", "examples", "two_equations.json",
)
_SOLUTION = [2.0, 1.0]      # satisfies both equations
_WRONG = [0.0, 0.0]         # satisfies neither


@pytest.fixture
def cas():
    return CASEngine(_EXAMPLE, ["x", "y"])


@pytest.fixture
def graph():
    return load_graph(_EXAMPLE)


@pytest.fixture
def checker():
    return Checker()


# --- shaping ---------------------------------------------------------------

def test_energy_decreasing_gives_positive_shaping():
    """Spec: fake trajectory with energy decreasing -> positive shaping reward."""
    traj = {"x_final": _SOLUTION, "energy_trajectory": [8.0, 4.0, 1.0, 0.0]}
    assert shaping_reward(traj) == pytest.approx(8.0)  # E0 - E_final = 8 - 0
    assert shaping_reward(traj) > 0


def test_energy_increasing_gives_negative_shaping():
    traj = {"x_final": _WRONG, "energy_trajectory": [0.0, 2.0, 5.0]}
    assert shaping_reward(traj) < 0


def test_shaping_is_potential_based_first_minus_last():
    """Only endpoints matter (telescoping potential), not the path in between."""
    a = {"x_final": _WRONG, "energy_trajectory": [10.0, 3.0]}
    b = {"x_final": _WRONG, "energy_trajectory": [10.0, 9.0, 1.0, 7.0, 3.0]}
    assert shaping_reward(a) == pytest.approx(shaping_reward(b))  # both 10 - 3 = 7


def test_shaping_fallback_uses_cas_when_no_trajectory(cas):
    traj = {"x_final": _SOLUTION, "energy_trajectory": []}
    # E(solution) == 0 -> fallback shaping -E(x) == 0
    assert shaping_reward(traj, cas=cas) == pytest.approx(0.0)


# --- terminal (conservative checker) --------------------------------------

def test_checker_accept_fires_terminal_reward(graph, checker):
    """Spec: checker accept -> terminal reward fires."""
    traj = {"x_final": _SOLUTION, "energy_trajectory": []}
    assert terminal_reward(traj, graph, checker) == pytest.approx(TERMINAL_B)


def test_checker_reject_gives_zero_terminal(graph, checker):
    traj = {"x_final": _WRONG, "energy_trajectory": []}
    assert terminal_reward(traj, graph, checker) == pytest.approx(0.0)


def test_no_checker_earns_no_terminal_reward(graph):
    traj = {"x_final": _SOLUTION, "energy_trajectory": []}
    assert terminal_reward(traj, graph, None) == pytest.approx(0.0)


# --- compute_reward (terminal + shaping) ----------------------------------

def test_compute_reward_accepted_solution(graph, checker, cas):
    traj = {"x_final": _SOLUTION, "energy_trajectory": [6.0, 0.0]}
    # terminal B + shaping (6 - 0)
    assert compute_reward(traj, graph, checker, cas) == pytest.approx(TERMINAL_B + 6.0)


def test_compute_reward_purist_drops_shaping(graph, checker, cas):
    traj = {"x_final": _SOLUTION, "energy_trajectory": [6.0, 0.0]}
    assert compute_reward(traj, graph, checker, cas, use_shaping=False) == pytest.approx(
        TERMINAL_B
    )


def test_compute_reward_rejected_solution(graph, checker, cas):
    traj = {"x_final": _WRONG, "energy_trajectory": [5.0, 5.0]}
    # no terminal, shaping 5 - 5 = 0
    assert compute_reward(traj, graph, checker, cas) == pytest.approx(0.0)


def test_x_values_read_from_tensor_like(cas):
    """_x_values tolerates a squeeze()/tolist() tensor-like without importing torch."""
    class _FakeTensor:
        def __init__(self, data):
            self._data = data

        def squeeze(self):
            return self

        def tolist(self):
            return self._data

    # Empty energy_trajectory forces the cas fallback, which reads x_final via _x_values.
    traj = {"x_final": _FakeTensor([2.0, 1.0]), "energy_trajectory": []}
    # (2, 1) is the exact solution -> E == 0 -> fallback shaping == -E == 0.
    assert shaping_reward(traj, cas=cas) == pytest.approx(0.0)
