from dataclasses import dataclass
from typing import List

import numpy as np

from .schema import VariableNode, FactorNode, Edge


@dataclass
class FactorGraph:
    variables: List[VariableNode]
    factors: List[FactorNode]
    edges: List[Edge]

    def get_values(self):
        return np.array(
            [v.value for v in self.variables],
            dtype=float,
        )

    def set_values(self, x):
        for node, value in zip(self.variables, x):
            node.value = float(value)