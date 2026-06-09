from dataclasses import dataclass


@dataclass
class VariableNode:
    id: str
    value: float = 0.0


@dataclass
class FactorNode:
    id: str
    expression: str


@dataclass
class Edge:
    variable_id: str
    factor_id: str
    coefficient: float = 1.0