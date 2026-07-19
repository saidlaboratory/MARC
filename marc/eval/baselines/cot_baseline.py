import re


import sympy as sp

def _split_residual(expression: str) -> tuple[str, str]:
    expr = sp.sympify(expression)
    const, var_part = expr.as_coeff_Add()
    rhs = -const
    return str(var_part), str(rhs)

def render_prompt(problem) -> str:
    var_names = [v.id for v in problem.graph.variables]

    equations = []
    for factor in problem.graph.factors:
        lhs, rhs = _split_residual(factor.expression)
        equations.append(f"{lhs} = {rhs}")

    system = "\n".join(equations)
    answer_line = ", ".join(f"{v}=<decimal>" for v in var_names)

    return (
        f"Solve this system of equations for {', '.join(var_names)}:\n\n"
        f"{system}\n\n"
        "Reason step by step, then give your final answer as the very last line "
        "in exactly this format, using decimal numbers (e.g. 3.5, not 7/2):\n"
        f"ANSWER: {answer_line}"
    )
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _build_client(model: str | None):
    """Provider-aware client. Prefers Gemini (free tier) when GEMINI_API_KEY is set,
    reached through its OpenAI-compatible endpoint so we keep the one SDK; otherwise
    falls back to the standard OpenAI client / OPENAI_API_KEY."""
    import os

    import openai

    if os.environ.get("GEMINI_API_KEY"):
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL
        )
        # flash-lite has a much larger free-tier daily request budget than flash.
        return client, (model or os.environ.get("COT_MODEL", "gemini-flash-lite-latest"))
    return openai.OpenAI(), (model or os.environ.get("COT_MODEL", "gpt-5"))


# --- resume cache: persist model answers so a run interrupted by a daily quota
# cap resumes where it left off (problems are regenerated deterministically). -----
import json as _json
import os as _os

_CACHE_PATH = _os.environ.get("COT_CACHE", "results/p2_main/cot_cache.json")
_cache: dict | None = None


def _cache_load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = _json.load(open(_CACHE_PATH))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _cache_store(key: str, value: str) -> None:
    c = _cache_load()
    c[key] = value
    _os.makedirs(_os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    _json.dump(c, open(_CACHE_PATH, "w"))


def call_model(prompt: str, model: str | None = None, *, max_retries: int = 5) -> str:
    """One chat completion with exponential backoff on rate limits (free tiers 429)."""
    import time

    import openai

    client, resolved = _build_client(model)
    cache_key = f"{resolved}::{prompt}"
    cached = _cache_load().get(cache_key)
    if cached is not None:
        return cached

    delay = 4.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=resolved,
                max_completion_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            _cache_store(cache_key, text)
            return text
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")

import re


def parse_answer(text: str, n_vars: int) -> list[float] | None:
    match = re.search(r"ANSWER:\s*(.+)", text)
    if match is None:
        return None

    numbers = re.findall(r"-?\d+\.?\d*", match.group(1))
    if len(numbers) != n_vars:
        return None

    return [float(n) for n in numbers]


class CoTSolver:
    def __init__(self, model: str | None = None):
        # None -> auto-resolve per provider (gemini-2.5-flash on Gemini, gpt-5 on OpenAI)
        self.model = model

    def sample(self, problem, k: int) -> list[list[float]]:
        prompt = render_prompt(problem)
        n_vars = len(problem.solution)

        candidates = []
        for _ in range(k):
            text = call_model(prompt, self.model)
            parsed = parse_answer(text, n_vars)
            if parsed is None:
                parsed = [float("nan")] * n_vars
            candidates.append(parsed)
        return candidates
    
import json
from marc.eval.problems import in_distribution, held_out_structure
from marc.eval.runner import run_split_eval

import json
from marc.eval.problems import in_distribution, held_out_structure
from marc.eval.runner import run_split_eval

if __name__ == "__main__":
    import os

    N = int(os.environ.get("COT_N", "25"))
    PERTURB_DELTA = 0.1
    K = 1

    solver = CoTSolver()  # auto: Gemini (gemini-2.5-flash) if GEMINI_API_KEY else OpenAI
    _, model_used = _build_client(solver.model)
    print(f"CoT baseline provider model: {model_used}  (N={N} per split)")

    metrics = run_split_eval(
        in_distribution(N),
        held_out_structure(N),
        solver=solver,
        perturb_delta=PERTURB_DELTA,
        n_samples=K,
        solver_name="cot_baseline",
    )

    metrics["model"] = model_used
    os.makedirs("results/p2_main", exist_ok=True)
    with open("results/p2_main/cot_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"solve rate (in-dist): {metrics['splits']['in_distribution']['solve_rate']}")
    print(f"solve rate (held-out): {metrics['splits']['held_out_structure']['solve_rate']}")
    print(f"generalization gap: {metrics['generalization_gap']}")