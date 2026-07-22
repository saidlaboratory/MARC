import json
import torch
from torch.utils.data import Dataset

from marc.graph.serialize import load_graph
from marc.graph.pyg import build_heterodata


class MARCDataset(Dataset):
    """Dataset of (HeteroData, solution_tensor) pairs.

    Each item is loaded from a pair of files:
    - graph_path: JSON file with the FactorGraph
    - solution_path: JSON file mapping variable_id -> float value

    The HeteroData has:
    - data["variable"].x: [n_vars, 1] current values (initialized to 0.0)
    - data["factor"].x: [n_factors, 1] zeros
    - data[("variable","connected_to","factor")].edge_index: [2, n_edges]
    - data[("variable","connected_to","factor")].edge_attr: [n_edges, 1] coefficients

    The solution_tensor: [n_vars, 1] with the known solution values (x*)
    """

    def __init__(self, path_pairs):
        """
        Args:
            path_pairs: list of (graph_json_path, solution_json_path) tuples
        """
        self.path_pairs = path_pairs

    def __len__(self):
        return len(self.path_pairs)

    def __getitem__(self, idx):
        graph_path, solution_path = self.path_pairs[idx]
        graph = load_graph(graph_path)
        with open(solution_path) as f:
            solution = json.load(f)

        data = build_heterodata(graph)

        x_star = torch.tensor(
            [[solution[v.id]] for v in graph.variables],
            dtype=torch.float,
        )

        return data, x_star
