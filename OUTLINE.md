# OUTLINE — MARC v0.2: the orchestrator reframe

**Purpose of this document:** everything a team member needs to understand the new framing and start writing — the thesis, the story arc, the evidence and where it lives, section-by-section paper guidance, framing rules, and what is blocked on the overnight run. Read this before touching the `.tex`.

**Deadline context:** AAAI abstract deadline July 27, 2026. The `.tex` does not exist yet. Writing starts now against this outline; final numbers drop in from the clean overnight run.

---

## 1. The thesis in three sentences

Solving a constraint problem involves two kinds of decisions: **continuous** (what values satisfy the equations — owned by classical solvers, which we measured to be near-unbeatable) and **discrete** (what representation makes the problem tractable at all — which auxiliary variable, substitution, or defining relation to introduce — where classical solvers have *nothing* and enumeration grows combinatorially). MARC learns the discrete decision: a structural prior over constraint-graph augmentations, run as a discrete-diffusion policy, whose proposals are completed by classical value solvers and accepted only by an exact symbolic checker. One line: **a neural mathematician's instinct for "introduce `d = x − y`," bolted onto solvers that finish the job and a checker that keeps everyone honest.**

## 2. Why this framing (and why it is credible)

1. **Our own controls chose it.** We bet on value diffusion first and falsified it publicly: learned value proposals tie/lose to random restart on coupled systems (R7); Levenberg–Marquardt with restarts saturates the hard families at 1.000; the GNN could not even overfit `Ax=b` from raw coefficients. Meanwhile the structure-selection policy is the one learned component that beat its controls.
2. **The economics invert.** For values, enumeration/search was cheap → the learned model had no room. For structure, the candidate space is combinatorial → one policy forward vs. exponentially many candidate-solve attempts, an advantage that *grows* with difficulty.
3. **Precedent.** AlphaGeometry (Nature 2024) is the same division of labor — neural auxiliary-construction proposals + symbolic engine — in geometry-specific machinery. MARC generalizes the pattern to arbitrary constraint graphs with an exact checker in the loop. Cite it early and prominently; it converts "weird pivot" into "recognized paradigm, generalized."
4. **The negative results are the motivation, not the embarrassment.** The paper's credibility rests on: we measured where learning does not help, and located where it does.

## 3. The story arc (this is the paper's spine)

1. **Problem:** LLM math reasoning is memorization-prone and unverifiable; we want derive-not-recall on a checkable substrate.
2. **Substrate:** constraint graphs + exact CAS residuals + a two-stage checker (numeric, then symbolic-exact). Training reward comes only from the checker.
3. **First bet, measured and closed:** value diffusion. Present R7 (coupled negative), the LM column, and the random-restart control *as findings* — with CIs. This section buys the trust the main claim spends.
4. **The surviving observation:** entrapment (noise escapes deterministic traps, 0.525 ± 0.086) — stochasticity matters in *search*; it never required a learned denoiser.
5. **The reframe:** the discrete/continuous decision split (§1 above). What classical solvers cannot do at any budget: change the representation.
6. **The system:** aux-required problem families (fixed graph certifiably unsolvable; only the right augmentation fixes it), candidate menus with certified exactly-one-solvable structure and hard negatives, the structure policy (absorbing-D3PM over padded slots, graph-conditioned encoder, value head predicting the defining constant), classical solvers downstream, checker gate.
7. **Results:** invention/solve rates vs. the control battery (random-slot, no-context, always-none, gold-oracle, enumeration), cross-pattern holdout, amortization costs, hard-negative confusion — all under the clean seed protocol, multi-seed, Holm-corrected.
8. **Honest scope:** menu-based selection with predicted defining value = rungs 1–2 of a ladder whose endpoint (free-form generation) is future work. State the ladder explicitly.

## 4. Evidence inventory (what exists, where it lives, what may be cited)

| Evidence | Status | Source of truth |
|---|---|---|
| Entrapment: noise reduces trapping 1.000 → 0.475, reduction 0.525 ± 0.086, N=200 | **Solid, citable** | `paper/RESULTS.md` R2; `results/p1_entrapment/` |
| Coupled negative (R7): learned ties/loses random at every n | **Solid, citable** — it is the motivation | `paper/RESULTS.md` R7; `results/p_coupled/coupled.json` |
| LM/exact classical columns on hard + coupled suites | **Solid** (regenerate rows in the overnight run) | `scripts/run_hard_eval.py`, `run_coupled_eval.py` outputs |
| Hybrid beats cold-start Langevin (but so does random) | Solid, cite with both halves | `paper/RESULTS.md` R3 |
| Cross-family value-transfer (partial, 2/4) | Citable with caveats | `paper/RESULTS.md` R4 |
| Dimension scaling | Only the **unified-v2** methodology run | `results/p_scaling/scaling.json` (`methodology: "unified-v2"`) |
| Structure selection 0.45/0.53 vs random 0.125 | **WITHDRAWN — never cite** (eval seeds == validation seeds; see R8) | `paper/RESULTS.md` R8 |
| Structure selection, clean protocol | **Pending the overnight run** — the paper's headline slot | will land in `results/p5_invention/invention.json` + `invention_heldout.json` |
| MATH coverage 0/48 | Citable as scope reality-check | `results/p_math/coverage.json` |

**Citation law (non-negotiable):**
- Structure-selection numbers only from runs whose JSON has `seed_hygiene.overlap_instances: 0`.
- Every rate: N + Wilson CI. Every comparison: z-test, Holm-corrected within the declared family.
- `refine`/`lm`/`exact`/`random` are always labeled classical baselines, never system results.
- The word **"invents"** is reserved for the ladder's endpoint. Current claims say **"menu-based structure selection (with predicted defining value)."**
- Positive controls are reported with their by-construction caveats (`positive_control.by_construction` in the eval JSON).

## 5. Section-by-section writing guide

Suggested owners in (parentheses) — reassign freely at the meeting.

### §1 Introduction (Quang)
Lead with the discrete/continuous decision split, not with diffusion. The AlphaGeometry sentence appears by paragraph 2. The contribution list: (i) an honest, controlled study closing the learned-value-proposal route on constraint graphs; (ii) a general substrate + certified problem families for studying learned structure augmentation; (iii) a trained structure policy beating controls under a contamination-proof protocol; (iv) the amortization analysis. Do not promise invention.

### §2 Substrate & verification (Davin)
Constraint graphs (variable/factor nodes, expression strings), CAS residuals/energy, two-stage checker, why checker-only reward. Source: `marc/graph/`, `marc/cas/`, TECHNICAL_GUIDE §§3–7. Half a page; this is plumbing, written confidently.

### §3 The value-diffusion study (honest negative) (Sparsh)
The controlled experiments and what killed each claim: random-restart control shrank R5; R7 coupled families closed the route; LM saturates; the `Ax=b` overfit probe. Entrapment as the surviving (pre-registered) positive. Tone: measurement, not apology. Sources: RESULTS.md R1–R7, `paper/learned_solver_fix.md`, `results/p_coupled/`.

### §4 Method: the structure policy (Quang)
Aux-required families + certificates (exact rank for linear; **empirical probes for nonlinear — state plainly they are probabilistic claims at a stated budget**), menu construction with hard negatives and randomized gold support, padded-slot representation, absorbing-D3PM forward/reverse, the graph-conditioned encoder, predicted defining value. The ladder (§8 below) closes this section. Sources: `marc/structure/`, `marc/data/aux_required.py`, TECHNICAL_GUIDE §10/§14.

### §5 Experimental protocol (Sparsh)
The part reviewers will probe hardest — write it proudly: disjoint seed spaces asserted at checkpoint load (contamination is *impossible by protocol*, and we say why the protocol exists — we caught ourselves once); multi-seed pooled Wilson CIs; Holm correction over the declared comparison family; the full control battery (random-slot, no-context, always-none, gold-oracle, enumeration); cross-pattern holdout; hard-negative confusion; amortization measurement. Source: `scripts/run_invention_eval.py`, `paper/REVIEW_ATTACKS.md`.

### §6 Results (Akash, once the overnight lands)
Placeholder tables NOW with the exact JSON keys they will be filled from (`samplers.*.invention_rate`, `arms.enumeration.*`, `comparisons_holm.*`, `invention_heldout.json`). Structure: main table (policy vs. controls, both samplers), cross-pattern table, amortization figure (policy cost vs. enumeration cost as K grows), hard-negative confusion, value-solver context rows.

### §7 Related work (Akash)
AlphaGeometry (closest relative — generalize, don't compete), D3PM/DiGress (formalism), neural algorithmic reasoning (independent evidence for delegating numerics), neural CO/SAT solvers (we learn representation change, not search), RLVR/verifier training, PoT/PAL (computation delegation taken to its end). The README's prior-art table is the skeleton.

### §8 Limitations & the ladder (Quang)
Rungs: menu selection → predicted value → compositional/multi-aux → free-form generation. Current work = rungs 1–2. Also: synthetic families only, MATH coverage 0/48, empirical certificates are budget-relative, single training seed caveat where applicable. Reviewers reward this section — write it first, not last.

## 6. Anticipated reviews and our answers (from `paper/REVIEW_ATTACKS.md`)

| Attack | Our answer |
|---|---|
| "Selection from K candidates is classification, not invention." | Correct — and we say so (the ladder). The claim is amortized structure choice under verification, with a continuous value head (rung 2) already beyond pure classification. |
| "Enumeration solves your task exactly." | Reported as an arm, expected 1.00 — the claim is **cost**: measured policy-vs-enumeration economics, widening with K. |
| "The policy could be matching family signatures, not reading the graph." | Randomized gold support kills the signature; the no-context ablation and hard-negative confusion measure graph-reading directly. |
| "Your earlier numbers were contaminated." | Yes — we caught it, withdrew them (R8), and rebuilt the protocol so the eval refuses contaminated seeds. This is a strength; cite the protocol. |
| "Value diffusion failed; why keep the diffusion formalism at all?" | For structure, discrete diffusion is the natural formalism for ABSENT→active instantiation (D3PM), and the single-shot ablation is always reported alongside — if single-shot wins at scale, the paper says so. |
| "Toy scale." | Conceded in Limitations; the certified-family methodology and the protocol are contributions independent of scale. |

## 7. What is blocked vs. writable today

**Writable now (≈70% of the paper):** §§1, 2, 3, 4, 5, 7, 8 — everything except final result numbers. The negative-result section is fully numbered already.

**Blocked on the overnight run (Sparsh, MacBook M5 — see `RUNBOOK_SPARSH.md`):** §6 tables; the abstract's headline sentence; the amortization figure. The harness produces every number the placeholders need in one command.

**Decision gate after the run:** policy beats controls on nonlinear families with cross-pattern holdout intact → submit AAAI July 27. Signal is weaker → the same paper, honestly hedged, goes to the next cycle (ICLR ~6 weeks out) with rungs 3+ matured; do not force it.

## 8. Glossary (use these terms, exactly)

- **Structure policy** — the learned model proposing augmentations (never "the solver").
- **Menu-based structure selection** — the current capability (rungs 1–2). Not "invention."
- **Aux-required family** — problems whose fixed graph is certifiably unsolvable without the correct augmentation.
- **Certificate** — proof of unsolvability: `exact` (linear rank theorem) or `empirical` (solver-probe at a stated budget; probabilistic).
- **Enumeration arm** — try every candidate, first checker-accept wins; the exact ceiling and the cost baseline.
- **Amortization** — policy inference cost vs. enumeration cost; the economic claim.
- **Seed hygiene** — disjoint train/val/test seed ranges, asserted mechanically; `overlap_instances: 0` or the number does not exist.
- **Classical baselines** — `refine` (Langevin), `lm` (Levenberg–Marquardt), `exact` (linear), `random` (multi-start). Always labeled, never headlines.

---

*Questions on framing → this file + `paper/REVIEW_ATTACKS.md` first, then Quang. Questions on any number → `paper/PROVENANCE.md` has its command, seed, and commit.*
