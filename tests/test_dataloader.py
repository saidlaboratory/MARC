import json
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from marc.data.dataset import MARCDataset
from marc.data.collate import collate_fn

EXAMPLE_GRAPH = "marc/data/examples/two_equations.json"
EXAMPLE_SOLUTION = {"x": 2.0, "y": 1.0}


def make_solution_file():
    """Write solution dict to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(EXAMPLE_SOLUTION, f)
    f.close()
    return f.name


def test_dataset_loads_item():
    sol_path = make_solution_file()
    dataset = MARCDataset([(EXAMPLE_GRAPH, sol_path)])
    assert len(dataset) == 1
    data, x_star = dataset[0]
    # 2 variables, 2 factors, 4 edges
    assert data["variable"].x.shape == (2, 1)
    assert data["factor"].x.shape == (2, 1)
    assert data["variable", "connected_to", "factor"].edge_index.shape == (2, 4)
    assert data["variable", "connected_to", "factor"].edge_attr.shape == (4, 1)
    assert x_star.shape == (2, 1)


def test_dataset_solution_values():
    sol_path = make_solution_file()
    dataset = MARCDataset([(EXAMPLE_GRAPH, sol_path)])
    data, x_star = dataset[0]
    # Solution should be x=2, y=1 in some order
    vals = sorted(x_star.squeeze().tolist())
    assert vals == pytest.approx([1.0, 2.0])


def test_dataloader_batches():
    sol_path = make_solution_file()
    dataset = MARCDataset(
        [(EXAMPLE_GRAPH, sol_path), (EXAMPLE_GRAPH, sol_path)]
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batched_data, solutions = next(iter(loader))
    # 2 graphs batched together: 4 total variable nodes
    assert batched_data["variable"].x.shape[0] == 4
    assert len(solutions) == 2


def test_edge_attr_coefficients():
    sol_path = make_solution_file()
    dataset = MARCDataset([(EXAMPLE_GRAPH, sol_path)])
    data, _ = dataset[0]
    coeffs = data["variable", "connected_to", "factor"].edge_attr.squeeze().tolist()
    # two_equations.json has coefficients [1, 1, 1, -1]
    assert sorted(coeffs) == pytest.approx([-1.0, 1.0, 1.0, 1.0])
