"""CoT baseline pure helpers (marc/eval/baselines/cot_baseline.py).

Only the offline, deterministic pieces are covered — prompt rendering and answer
parsing. The networked ``call_model`` is out of scope for the unit suite (it
lazily constructs the OpenAI client and hits the API).
"""

from marc.eval.baselines.cot_baseline import parse_answer, render_prompt
from marc.graph.serialize import load_graph
from marc.eval.runner import Problem

GRAPH_PATH = "marc/data/examples/two_equations.json"


def _problem():
    G = load_graph(GRAPH_PATH)
    return Problem(id="two_eq", graph=G, solution=[2.0, 1.0])


def test_render_prompt_lists_variables_and_equations():
    prompt = render_prompt(_problem())
    assert "x" in prompt and "y" in prompt
    assert "ANSWER:" in prompt
    # one line per factor in the system
    assert prompt.count("=") >= len(_problem().graph.factors)


def test_parse_answer_extracts_floats():
    assert parse_answer("...\nANSWER: x=2.0, y=1.0", n_vars=2) == [2.0, 1.0]
    assert parse_answer("ANSWER: 3.5, -4", n_vars=2) == [3.5, -4.0]


def test_parse_answer_rejects_wrong_arity_or_missing():
    assert parse_answer("no answer here", n_vars=2) is None
    assert parse_answer("ANSWER: x=1.0", n_vars=2) is None  # too few numbers
