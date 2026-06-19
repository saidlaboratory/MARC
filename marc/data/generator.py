"""ProblemGenerator: produce and persist procedurally-generated math problems."""

import json
import os
import random
from typing import List, Tuple

from marc.cas.engine import CASEngine
from marc.graph.serialize import save_graph


class ProblemGenerator:
    """Generate math problems from a list of templates and save them to disk.

    Args:
        templates: Iterable of template objects that expose a ``generate`` method
            and a ``name`` attribute.
        split_ratio: Fraction of instances (per template) assigned to train.
        seed: Master RNG seed for reproducibility.
    """

    def __init__(self, templates, split_ratio: float = 0.8, seed: int = 42):
        self.templates = list(templates)
        self.split_ratio = split_ratio
        self.seed = seed

    def generate(
        self, n_per_template: int, output_dir: str
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Generate *n_per_template* problems per template and persist them.

        For each template, the first ``split_ratio`` fraction of instances go to
        train and the remainder to test.

        For every instance the CASEngine is used to assert the generated solution
        actually satisfies the factor graph, enforcing the key invariant.

        Returns:
            (train_paths, test_paths) where each element is a list of
            (graph_json_path, solution_json_path) tuples.
        """
        os.makedirs(output_dir, exist_ok=True)
        train_paths: List[Tuple[str, str]] = []
        test_paths: List[Tuple[str, str]] = []
        rng = random.Random(self.seed)

        for template in self.templates:
            template_dir = os.path.join(output_dir, template.name)
            os.makedirs(template_dir, exist_ok=True)
            instance_paths: List[Tuple[str, str]] = []

            for i in range(n_per_template):
                graph, solution = template.generate(seed=rng.randint(0, 2**31))

                json_path = os.path.join(template_dir, f"{i}.json")
                save_graph(graph, json_path)

                # Verify invariant: CASEngine must accept the solution.
                symbol_names = " ".join(v.id for v in graph.variables)
                x_vals = [solution[v.id] for v in graph.variables]
                cas = CASEngine(json_path, symbol_names)
                assert cas.accepts(x_vals, tol=1e-4), (
                    f"CASEngine rejected generated solution for "
                    f"{template.name} instance {i}: energy={cas.energy(x_vals)}"
                )

                sol_path = os.path.join(template_dir, f"{i}_solution.json")
                with open(sol_path, "w") as f:
                    json.dump(solution, f)

                instance_paths.append((json_path, sol_path))

            split_idx = int(len(instance_paths) * self.split_ratio)
            train_paths.extend(instance_paths[:split_idx])
            test_paths.extend(instance_paths[split_idx:])

        return train_paths, test_paths
