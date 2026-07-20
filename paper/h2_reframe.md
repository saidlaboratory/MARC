# H2 (structure invention) — honest reframe

**Fixing-plan item:** A7. This is the framing to use in the paper. H2 is **preliminary
evidence**, not a tested hypothesis — say so plainly.

## What H2 claims
That the system can *invent auxiliary objects* (extra variables / constraints) that make
an otherwise-hard problem tractable — "lemmas and auxiliary quantities" (CONCEPT.md).

## What we actually have
- Evidence: `results/p3_h2/h2_report.md`, three toy families in `marc/eval/structure_eval.py`
  (`sum_product`, `bilinear_product`, `quadratic_link`).
- **Solve-rate benefit: Δ = 0.00** on all three toys (fixed graph vs. augmented graph).
- **Auxiliary-usage rate: 39–100%** — the search *routes through* the auxiliary structure
  when it is present.
- The "structure model" is **energy-guided best-of-k with an oracle**, **not** a trained
  D3PM sampler over `marc/model/structure_head.py` (that head exists but was never trained).

## Why Δ = 0 is not a failure of the model — it's baked into the eval
The toy families are constructed so the augmented graph is **solution-equivalent** to the
fixed graph (the report states this). Both encode the same solution set, so *any* correct
solver reaches the same solve rate on both — Δ = 0 is a **property of the benchmark
design**, not evidence about structure invention. The usage rates are the only real signal:
they show the search *exploits* auxiliary structure when it is available.

## Paper language (drop-in)
> We report H2 as **preliminary**. Auxiliary-usage rates (39–100%) show the solver routes
> through auxiliary structure when present, but solve-rate parity (Δ = 0) is expected because
> our current toy families are solution-equivalent by construction. Trained structure
> diffusion (a D3PM sampler over the slot-type head) and a benchmark family that is
> **unsolvable without the auxiliary** — which would make H2 falsifiable — are left to future
> work.

## What would make H2 real (post-deadline, A7 P2)
1. Train the D3PM sampler over `StructureHead` (currently untrained).
2. Build ≥1 toy family where the fixed graph is **provably unsolvable** without the auxiliary
   variable (current toys keep the solution set identical → Δ = 0 is unavoidable). That family
   is the falsification test H2 needs.

**Do not** present H2 as a validated result. The house style is the H2 report's own honesty.
