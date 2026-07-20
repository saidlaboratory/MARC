# MATH benchmark — coverage & scope analysis (reality check)

**What this is:** an honest measurement of how far MARC reaches on real competition math
(a 48-problem MATH-500 sample, `marc/data/math_benchmark/math500_sample.jsonl`,
`scripts/run_math_coverage.py`). It is **not** a claim that MARC solves MATH. MARC is a
numeric constraint solver; general MATH needs reasoning + proof it does not do.

## Headline numbers
| Metric | Value |
|---|---|
| Problems | 48 |
| **Parser coverage** (formalize into a FactorGraph at all) | **0 / 48 = 0.000** |
| Solve accuracy on covered subset | n/a (nothing covered) |

The template parser (`marc/nl/parser.py`) recognizes exactly three sentence shapes
(sum/difference, sum/product, two-distance geometry). None of the 48 real problems match
those exact shapes, so coverage is zero. **The bottleneck is formalization (NL → graph),
not the solver.**

## The honest scope breakdown (why 0%, and what is reachable)
Hand-categorizing the 48 problems by whether they fit MARC's paradigm — *find a variable
assignment satisfying algebraic constraints, with a numeric answer*:

| Category | ~count | In MARC's paradigm? |
|---|---|---|
| **Constraint-shaped** (single/system of equations, coordinate geometry — "solve for x/y") | ~8–10 (~20%) | **Yes** — reachable *if* a formalizer produced the graph; the solver can then handle them |
| **Pure computation** (evaluate/simplify an expression, gcd, divisor count, arithmetic series, base conversion) | ~15 (~31%) | No — this is CAS/SymPy work, not constraint search; solving these would not exercise MARC's contribution |
| **Reasoning / proof / counting / construction** (combinatorics, "find all n", number-theory arguments, extremal problems) | ~23 (~48%) | **No** — different paradigm entirely; out of architectural scope |

So even with a perfect formalizer, MARC's *solver* targets only the ~20% constraint-shaped
slice; ~31% is CAS territory (not our contribution); ~48% is out of scope by architecture.

## Takeaways for the paper
1. Report this as a **scope/reality-check**, not a result: "our template formalizer covers
   0% of a MATH-500 sample; ~20% of the sample is constraint-shaped (MARC's target
   paradigm), and reaching it requires autoformalization we defer to future work."
2. It quantifies, on a standard benchmark, exactly what the system does and does not target —
   which strengthens the paper's honesty and pre-empts "does it work on real problems?".
3. The realistic path to the constraint-shaped slice is a learned/LLM **formalizer** that
   emits FactorGraphs, with MARC as the verified solver (future work; see also the hybrid
   LLM+MARC direction). Building a regex formalizer to inflate this number would misattribute
   CAS/SymPy computation to MARC and is deliberately avoided.

## Reproduce
```
python scripts/run_math_coverage.py   # -> results/p_math/coverage.json
```
Sample pulled from HuggingFaceH4/MATH-500 (test split). Expand the JSONL to run on more.
