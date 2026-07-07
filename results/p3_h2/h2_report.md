# H2 — Auxiliary Object Usage

**Hypothesis (CONCEPT.md H2):** allowing the refinement process to modify graph
structure (adding nodes/edges) enables the introduction of lemmas and auxiliary
quantities.

**Metric:** `auxiliary_usage_rate` (`marc/eval/structure_eval.py`) — the fraction of
*solved* structure-model runs whose winning candidate actually used the auxiliary
slot. Reported alongside solve rate so usage is read against whether it helped.

## Method

Full discrete structure diffusion (D3PM over slot types) is [Frontier] and not yet
trained end-to-end — only its interface (`marc/model/structure_head.py`'s
ABSENT-vs-active categorical head) exists on `main`. Per TECHNICAL_GUIDE §14
("build the simplest thing that exhibits the phenomenon first"), this eval's
**structure model** is an MVP stand-in: energy-guided best-of-`k` search that
splits its restart budget between the *fixed* graph (no auxiliary slot) and an
*augmented* graph carrying one extra auxiliary variable + its defining factor(s),
keeping whichever restart the CAS energy accepts at lowest energy. This substitutes
the exact energy oracle for a trained slot-activation policy — a real, honest
proxy, not a fabricated one, but not the frontier system either.

Three toy families, each a genuine textbook auxiliary-quantity trick:

| Toy | Fixed constraints | Auxiliary invented | Why it's a real lemma |
|---|---|---|---|
| `sum_product` | `x+y=s`, `x*y=p` | `d=x-y`, `d²=s²-4p` | Vieta's difference-of-roots identity |
| `bilinear_product` | `x*y=A`, `y*z=B`, `x*z=C` | `w=x*y*z`, `w²=A*B*C` | product trick — `x=w/B, y=w/C, z=w/A` |
| `quadratic_link` | `x²+y=a`, `x²-y=b` | `z=x²` | substitution that linearises both equations in `(z,y)` |

The auxiliary variable is always *added*, never removed — the augmented graph keeps
every original factor, so its solution set for the original variables is identical
to the fixed graph's. What's measured is whether the extra structure changes what
the *search* finds, and whether the winning run used it.

Run: `python scripts/run_h2_eval.py --n 25 --k 10` → `results/p3_h2/summary.json`,
`results/p3_h2/trajectories.jsonl` (one line per toy × instance × model).

## Results (n=25 instances/toy, k=10 restarts, 5/5 fixed/aux split)

| Toy | Fixed solve rate | Structure solve rate | Δ | Auxiliary usage rate |
|---|---|---|---|---|
| `sum_product` | 0.72 | 0.72 | 0.00 | **0.39** |
| `bilinear_product` | 0.36 | 0.36 | 0.00 | 0.00 |
| `quadratic_link` | 0.28 | 0.28 | 0.00 | **1.00** |
| **Overall** | **0.45** | **0.45** | **0.00** | **0.41** |

## Case study: `sum_product`, seed 1 — `x+y=-5, x*y=6`

**Before (fixed graph, 2 variable nodes, 2 factor nodes):**

```
  (x) --1--> [eq1: x+y-(-5)]
  (y) --1-->/
  (x) --1--> [eq2: x*y-6]
  (y) --1-->/
```

**After (augmented graph, +1 variable node, +2 factor nodes):**

```
  (x) --1--> [eq1: x+y-(-5)]
  (y) --1-->/
  (x) --1--> [eq2: x*y-6]
  (y) --1-->/
  (x) --1--> [eq_def: d-(x-y)]  <-- new
  (y) -(-1)->/
  (d) --1-->/
  (d) --1--> [eq_aux: d**2-((-5)**2-4*6)]   <-- new, i.e. d**2 - 1
```

The known solution is `x=-2, y=-3` (or the symmetric `x=-3, y=-2`). In
`trajectories.jsonl`, this instance's structure-model winner is `used_aux=true`,
converging to `x=-2.0, y=-3.0` at energy `7.9e-31` (tighter than the fixed model's
own accepted run at `9.2e-21` from the same instance) — a real one because the
auxiliary factor pins `d=x-y=1`, and `1² = (-5)² - 4·6 = 25-24 = 1` checks out
independently of `x, y`; the model then only has to satisfy the *linear* `eq_def`
to recover `x, y` from `d`, which converges faster and tighter than solving the
bilinear `eq2` directly.

## Discussion

- **Usage is real and substantial** (39% for `sum_product`, 100% for
  `quadratic_link`) — when the structure model solves a problem, its winning run
  frequently routes through the auxiliary object rather than ignoring it, and the
  case study above shows a concrete instance where doing so reaches a *tighter*
  solution than the fixed graph's own best run.
- **It does not yet lift the headline solve rate** at this toy/non-learned-search
  scale — Δ=0.00 on all three toys. Instance-level inspection (not shown in the
  summary table) found the auxiliary graph's restarts sometimes solve instances the
  fixed restarts alone would miss and vice versa (the extra dimension is a
  double-edged sword for a blind energy-gradient search: it can also introduce its
  own flat/nonconvex regions, e.g. `eq_aux`'s `d²` term is flat at `d=0`). Because
  `solve_structure` always includes fixed-graph restarts *in the same budget*, one
  side's win rarely goes to waste, damping the net effect to ~0 in aggregate.
- **Honest interpretation:** this MVP proxy answers "does the model use the
  auxiliary object when it's available and helpful" (yes, frequently) but not "does
  a *trained* structure-diffusion policy know when to invoke it" — that requires
  the real D3PM sampler over `StructureHead`, which is explicitly [Frontier] scope
  (TECHNICAL_GUIDE §10, §14: "Structure diffusion may not converge in time... the
  paper's defensible core is value diffusion + checker RL"). The natural next step
  once a trained policy exists is to re-run this exact harness swapping
  `solve_structure`'s energy-argmin selection for the policy's own choice, and
  check whether usage rate rises *and* solve rate lifts together — that would be
  the actual H2 confirmation.
- **Aside:** building this harness surfaced a narrow precision edge case in the
  shared `Checker._to_exact` float→rational snapping (can pick a spurious nearby
  fraction instead of the intended simple one when a residual sits within ~1e-10 of
  the tolerance boundary) — flagged separately for a fix; it required tightening
  this eval's `refine()` polish schedule (`_POLISH_STEPS=2500`, `_POLISH_LR=0.1` in
  `structure_eval.py`) well past the shared default to avoid it, since these
  bilinear/quartic toy energies converge less cleanly than the linear systems the
  P1/P2 suites use.

## Reproduce

```bash
python scripts/run_h2_eval.py --n 25 --k 10
```
