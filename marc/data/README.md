# `marc/data` — problem templates & generation

Procedurally-generated problems are the training substrate for MARC. Each problem is a
**constraint graph** (`marc/graph`) plus a known **solution**, and every generated
instance is verified so the stored solution actually satisfies the graph — this is the
invariant the whole training loop relies on.

## The template contract

A template is a small dataclass exposing:

```python
name: str
generate(seed: int | None = None) -> tuple[FactorGraph, dict[str, float]]
```

- **variables** — `VariableNode(id, value=0.0)`; solving fills in the values.
- **factors** — `FactorNode(id, expression)`; `expression` is a SymPy-parseable string
  that must equal **0** at the solution.
- **edges** — `Edge(variable_id, factor_id, coefficient)` wiring variables into the
  factors they appear in (used by the GNN; not by the CAS/checker).

The returned `dict` maps each variable id to its solution value.
`marc/data/generator.py::ProblemGenerator` calls `generate`, saves the graph, and
**asserts the CASEngine accepts the solution** before persisting it.

## Template families

### Linear systems — `LinearSystem2x2Template`, `LinearSystem3x3Template`

Random square linear systems `A·v = b` with a known integer solution; degenerate
(singular `A`) draws are rejected and retried. Factors are linear, e.g. `2*x-3*y-5`.

### Coordinate geometry — `TriangleDistanceTemplate`, `PointSlopeTemplate`  *(P4)*

Extends generation beyond equation systems to **2-D coordinate geometry**. A point is a
pair of variable nodes `(px, py)`; relations become factor expressions:

| Factor kind | Expression (== 0) | Meaning |
|-------------|-------------------|---------|
| pin | `px - a` | anchor a known coordinate |
| distance | `(px-qx)**2 + (py-qy)**2 - d2` | squared distance between two points is `d2` |
| slope | `rise*(px-ax) - run*(py-ay)` | `P` lies on the line through `A` with direction `(run, rise)` |

Coordinates are integers, so every residual is an exact integer and the CAS / exact
checker accept the stored solution with zero energy.

- **`TriangleDistance`** — pin base points `A`, `B`; two distance factors fix vertex `C`
  (6 variables, 6 factors). Non-degenerate: `A, B, C` are distinct and non-collinear.
- **`PointSlope`** — pin anchor `A`; one slope + one distance factor fix `P`
  (4 variables, 4 factors). `P ≠ A` so the direction and distance are well-defined.

Convenience list: `GEOMETRY_TEMPLATES = [TriangleDistanceTemplate(), PointSlopeTemplate()]`.

```python
from marc.data.templates import GEOMETRY_TEMPLATES
graph, solution = GEOMETRY_TEMPLATES[0].generate(seed=0)
```

## Generating a dataset

```python
from marc.data.generator import ProblemGenerator
from marc.data.templates import GEOMETRY_TEMPLATES

gen = ProblemGenerator(GEOMETRY_TEMPLATES, split_ratio=0.8, seed=7)
train, test = gen.generate(n_per_template=20, output_dir="data/geometry")
```

Every instance is CAS-verified during generation; `tests/test_geometry.py` additionally
checks 20 samples per family against both the numeric CASEngine and the conservative
exact `Checker`.
