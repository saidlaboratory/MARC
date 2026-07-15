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
import openai

_client = openai.OpenAI()  # reads OPENAI_API_KEY from env automatically


def call_model(prompt: str, model: str = "gpt-5") -> str:
    response = _client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

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
    def __init__(self, model: str = "gpt-5"):
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
    N = 25              # TODO: confirm with Sparsh
    PERTURB_DELTA = 0.1  # TODO: confirm with Sparsh
    K = 1

    solver = CoTSolver(model="gpt-5")

    metrics = run_split_eval(
        in_distribution(N),
        held_out_structure(N),
        solver=solver,
        perturb_delta=PERTURB_DELTA,
        n_samples=K,
        solver_name="cot_baseline",
    )

    import os
    os.makedirs("results/p2_main", exist_ok=True)
    with open("results/p2_main/cot_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"solve rate (in-dist): {metrics['splits']['in_distribution']['solve_rate']}")
    print(f"solve rate (held-out): {metrics['splits']['held_out_structure']['solve_rate']}")
    print(f"generalization gap: {metrics['generalization_gap']}")