<div align="center">

# MARC

### **M**athematical **A**I **R**easoning **C**ore

*A learned structural prior over constraint graphs — neural proposals for **what to add** to a problem, classical solvers for the values, an exact checker for the truth.*

<br />

[![Status](https://img.shields.io/badge/status%20v0.3-structural%20repair-blue?style=for-the-badge)](https://github.com/saidlaboratory/MARC)
[![Phase](https://img.shields.io/badge/phase-structure%20invention%20ladder-blue?style=for-the-badge)](#the-invention-ladder)
[![Domain](https://img.shields.io/badge/domain-mathematical%20reasoning-6366f1?style=for-the-badge)](#motivation)
[![License](https://img.shields.io/badge/license-TBD-888?style=for-the-badge)](#)

<br />

**Quang Bui, Sparsh Roy, Akash Gundimeda, Davin Yin** · SAID Laboratory · July 2026

[Overview](#overview) · [What we measured](#what-we-measured-and-what-it-changed) · [The bet](#the-bet-division-of-labor) · [Invention ladder](#the-invention-ladder) · [Evaluation](#evaluation) · [Repo tour](#repo-tour) · [Prior art](#prior-art)

</div>

---

## Overview

> **MARC is an orchestrator, not a solver.**
> It represents a problem as a **constraint graph** and learns the one decision classical methods cannot make: **what structure to add** — the auxiliary variable, substitution, or defining relation that turns an unsolvable graph into a solvable one. Its candidate-conditioned ranker scores the graph produced by each repair; values are then found by classical solvers, and every answer must pass an exact symbolic checker.

The division of labor is the thesis:

| Decision | Who makes it | Why |
|---|---|---|
| **What structure to add** (auxiliary variable `d = x − y`, its defining factor, where it enters) | **Candidate-conditioned graph repair ranker** | No gradient exists over this choice; enumeration is combinatorial; a learned compatibility prior amortizes it |
| **What values satisfy the constraints** | **Classical solvers** (Levenberg–Marquardt, Langevin refinement, exact linear solve) | Near-unbeatable on smooth algebraic systems — we measured it, twice |
| **Is the answer true** | **Exact checker** (numeric gate + symbolic-exact gate) | Verification is the only training signal and the only termination condition |

One sentence for the whole project: *MARC is a neural mathematician's instinct for "introduce `d = x − y`," bolted onto solvers that finish the job and a checker that keeps everyone honest.*

---

## What we measured, and what it changed

MARC v0.1 bet on **value diffusion**: a learned denoiser iteratively refining node *values* toward consistency. We tested that bet with pre-registered controls and it lost — publicly, with confidence intervals. The reframe is not a pivot away from evidence; it is what the evidence chose:

| Finding | Result | Consequence |
|---|---|---|
| **Noise reduces entrapment** (RQ2) | Deterministic descent traps 100% vs. 47.5% with Langevin noise; reduction 0.525 ± 0.086, N=200, CI excludes 0 | Real — but it argues for *stochasticity in search*, not for a learned denoiser |
| **Learned value proposals vs. random restart** (coupled families, R7) | Learned **ties or loses at every dimension** once solutions are coupled; the earlier high-dim win was a separability artifact | The "learned proposal beats classical search" route is **closed** |
| **Classical baseline strength** | Levenberg–Marquardt with restarts saturates the hard nonlinear families at 1.000 | Raw solve rate can never be MARC's claim |
| **GNN numerical capacity** | The denoiser could not overfit `Ax = b` on four fixed systems from raw coefficients | Propagating precise numbers through message passing is a structural limitation, not a tuning problem |
| **Slot-based structure policy (v0.2)** | 0.410 vs 0.200 random in-pattern, but 0.234 vs 0.238 on an unseen pattern | Established the direction, but failed structural transfer |
| **Candidate-conditioned repair (v0.3)** | **Held-out linear pattern 0.565 vs 0.343 candidate-only / 0.283 random; balanced nonlinear 0.889 vs 0.422 / 0.253** | Operator-aware repair scoring replaces slot classification; nonlinear result is stable across three seeds (SD 0.006) |

Full evidence ledger: [`paper/RESULTS.md`](paper/RESULTS.md) · selected v0.3 evidence and invalidated pilots: [`results/p_repair/README.md`](results/p_repair/README.md) · every number's command/seed/commit: [`paper/PROVENANCE.md`](paper/PROVENANCE.md) · standing review-attack checklist: [`paper/notes/REVIEW_ATTACKS.md`](paper/notes/REVIEW_ATTACKS.md).

**Results integrity rules (house law):** classical solvers (`refine`, `lm`, `exact`) are always labeled baselines; every rate carries N and a Wilson CI; paired comparisons use exact McNemar tests; structure-selection numbers are citable only under an explicit, disjoint seed protocol and current data version. Numbers predating the relevant seed/data-protocol fix are withdrawn and must not be cited.

---

## The bet (division of labor)

### Why structure, not values

Solving a constraint system involves two different kinds of decision:

- **Continuous:** *what values satisfy these equations.* Smooth, gradient-rich, and owned by sixty years of numerical analysis. Learning adds nothing here — our controls confirmed it.
- **Discrete:** *what representation makes the problem tractable at all.* Which auxiliary quantity to introduce, which substitution linearizes the system, which lemma bridges the gap. No gradient exists over this space; enumeration grows combinatorially; and classical solvers have **nothing** — a solver cannot decide to invent `d = x − y`.

A learned prior over the discrete space amortizes a cost that *grows* with problem difficulty: one forward pass versus exponentially many candidate-solve attempts. That economics is the opposite of the value-diffusion bet, whose advantage shrank as baselines got the same compute.

The external precedent is strong: **AlphaGeometry** (Nature, 2024) is exactly this architecture — a neural model proposes the auxiliary constructions no deduction engine can derive; a symbolic engine does the rest. MARC builds the same phenomenon in a **general constraint-graph substrate** rather than geometry-specific machinery.

### System at a glance

```mermaid
flowchart LR
    P["Problem"] --> G["Constraint Graph"]
    G --> S["Repair Ranker\n(score each candidate-\naugmented graph)"]
    S -->|"proposed augmentation:\naux var + defining factor"| A["Augmented Graph"]
    A --> V["Classical Value Solver\n(LM / Langevin refine / exact)"]
    V --> C{"Checker\n(numeric + symbolic-exact)"}
    C -->|reject| S
    C -->|accept| OK["Verified Solution"]
```

| Component | Role | Where |
|-----------|------|-------|
| **Constraint graph** | Variables + factor nodes (relations); the shared substrate | `marc/graph/` |
| **Repair ranker (v0.3)** | Operator-aware message passing over each candidate-augmented graph; listwise selection with a matched candidate-only control | `marc/model/repair_ranker.py`, `scripts/run_repair_ranker.py` |
| **Structure policy (v0.2 baseline)** | Absorbing-D3PM reverse process over padded structure slots | `marc/structure/` |
| **Classical value solvers** | Levenberg–Marquardt (`lm`), Langevin refinement (`refine`), exact linear (`exact`) — always the value-finders, always labeled baselines | `marc/refine/`, `marc/eval/solver.py` |
| **Checker** | Two-stage gate: numeric tolerance, then symbolic-exact acceptance; sole reward source and sole termination condition | `marc/cas/checker.py` |
| **CAS** | Exact residuals/energy/gradients (SymPy) | `marc/cas/` |

---

## The invention ladder

We climb from selection toward generation, one falsifiable rung at a time. Each rung keeps the same end-to-end verification: a proposal only counts if the augmented graph **actually solves and passes the checker**.

| Rung | What the policy does | Status |
|:---:|---|---|
| **1 · Menu selection** | Pick the correct augmentation from K procedurally generated candidates (exactly one certified/reachable repair under the stated protocol) | **Supported**: 0.565 on an unseen linear pattern and 0.889 on balanced nonlinear menus |
| **2 · Predicted defining value** | The policy's value head supplies the defining constant itself — the candidate space becomes continuous; the menu only provides insertion structure | **Built** (`predicted_pin`); evaluated as the `policy_value` arm |
| **3 · Compositional / multi-aux** | Choose insertion set and defining relation independently; multiple simultaneous auxiliaries (the padded-slot schema already supports >1 active slot) | Designed, not built |
| **4 · Free-form generation** | Emit the defining expression itself — invention proper | The prize; out of scope until rungs 1–3 hold |

**Naming discipline:** until rung 4, the honest term is **menu-based structure selection (with predicted defining value)** — "invention" appears only in code identifiers and in describing the ladder's endpoint.

**Problem families.** Aux-required families where the fixed graph needs repair: linear patterns (`offset`/`coupled`/`shared`, exact rank certificates) and nonlinear patterns (`vieta`: `u = x − y + δ`; `quad_link`: `u = x² + δ`) with **empirical certificates** (a candidate is "unreachable" iff a 12-restart solver probe fails — recorded per instance as an empirical claim, not a theorem). Gold linear insertion support is randomized; nonlinear candidates are matched in expression family, support, coefficients, and balanced offset prior.

---

## Hypotheses (v0.3 — revised under evidence)

**H-Structure (supported for menu-based repair).** A trained structural prior over candidate-augmented graphs selects the representation change needed for solver success above random and candidate-only controls, transfers across a held-out linear pattern and partially across an unseen nonlinear relation, and reduces enumeration cost. The measured speedup grows from 1.21x at K=4 to 3.91x at K=16.

**H-Value (resolved, negative).** Learned value proposals do not beat random multi-start on coupled systems (R7) and cannot beat LM anywhere we tested. We keep this result in every paper we write — it is the motivation, measured.

**Retained from v0.1:** verification-gated training (the checker is the only reward), procedural generation with *structural* holdout (train patterns ≠ test patterns, not just fresh constants), and derive-not-recall evaluation discipline.

---

## Evaluation

What we measure, per run (`scripts/run_repair_ranker.py`):

| Metric | Question it answers |
|---|---|
| **Invention rate** vs. gold, + solve rate of the applied choice | Does the policy pick structure that *works end-to-end*? |
| **Random-slot / no-context / always-none controls** | Is it better than chance? Does it actually read the graph? |
| **Enumeration arm** (try every candidate, first accept wins) | The exact-solver ceiling — expected 1.00 — and its **cost** (solver calls, wall-clock) |
| **Amortization** (policy forwards + 1 solve vs. ~K/2 solves) | The economics of the bet, measured not asserted |
| **Cross-pattern holdout** (`--exclude-family`) | Generalization beyond memorizing a family's canonical augmentation |
| **Hard-negative confusion** | Is the policy reading constants, or matching insertion topology? |
| **Seed hygiene** (train/val/test ranges disjoint, asserted at load) | The eval refuses to run on contaminated seeds — protocol, not promise |
| Wilson CIs + exact paired McNemar tests + multi-seed repeats | Instance uncertainty, paired significance, and optimization robustness |

Value-solver context rows (never headlines): `refine`, `lm`, `random`, `exact` on the hard/coupled suites, with the entrapment ablation retained as the RQ2 result.

**Reproduce everything:** `python3 scripts/run_overnight.py` (one command, crash-safe, per-phase logs + manifest; see [`RUNBOOK_SPARSH.md`](RUNBOOK_SPARSH.md)).

---

## Repo tour

```
marc/
  graph/       constraint-graph schema, PyG conversion
  cas/         SymPy residuals/energy/gradients + two-stage checker
  data/        procedural templates: linear, hard nonlinear, geometry,
               coupled chains, aux-required families (the invention data)
  structure/   THE CORE BET — padded slots, absorbing-D3PM corruption,
               invention menus + certificates, structure policy + reverse sampler
  refine/      classical Langevin refinement + residual Jacobians
  model/       GNN encoders, operator-aware repair ranker, structure head
  diffusion/   value-diffusion machinery (retained: hybrid + ablations context)
  train/       Stage-A denoising, Stage-B GRPO (corrected), scale trainer (GPU/MPS),
               structure-policy training (CE + optional solve-reward)
  eval/        harness, metrics (Wilson/z/Holm), solver registry incl. lm/exact,
               structure evals
scripts/       repair-ranker training/eval/multi-seed scripts + legacy harnesses
paper/         RESULTS.md, PROVENANCE.md, notes/ (working notes), figures
```

---

## Prior art

| Line of work | Relationship to MARC |
|--------------|---------------------|
| **AlphaGeometry** (neural auxiliary constructions + symbolic engine) | The closest relative and the strongest precedent for the division of labor; MARC generalizes the pattern from geometry-specific machinery to arbitrary constraint graphs, with an exact checker in the loop |
| **Graph / discrete diffusion** (D3PM, DiGress) | The formalism for the structure policy: absorbing corruption over slots, learned reverse process; instantiation = ABSENT → active |
| **Neural algorithmic reasoning** | Independently documents why GNNs struggle with precise numerics — consistent with our `Ax=b` overfit failure and the delegation of values to classical solvers |
| **Neural / GNN constraint & SAT solvers** | Learned components inside search; MARC differs by learning the *representation change*, not the search itself |
| **RLVR & verifier-gated training** (GRPO, DeepSeek-R1) | The training discipline for both stages; reward only from the checker |
| **Tool-augmented computation** (PoT, PAL) | The reasoning/computation split, taken to its logical end: *all* computation is delegated |
| **Neurosymbolic & formal math** (Lean, AlphaProof) | The verifier-centric, derive-not-recall philosophy MARC retains from v0.1 |

---

## Success criteria

**Supported** if the repair ranker, under the clean seed/data protocol, (a) beats random and candidate-only controls with paired significance on nonlinear aux-required families, (b) holds on cross-pattern holdout, and (c) approaches the enumeration ceiling at a measured fraction of its cost—with the cost gap widening as K grows. Data Version 6 supports all three within the stated menu-based scope.

**Falsified** if the no-context ablation matches the full policy (the model isn't reading the graph), or cross-pattern transfer collapses to chance (it memorizes family signatures), or the amortization advantage disappears at realistic K.

Either outcome is publishable. That is the point of the protocol.

---

<div align="center">

<br />

*v0.3 · The founding value-diffusion framing is preserved in [CONCEPT.md](CONCEPT.md); the evidence that forced this reframe is in [paper/RESULTS.md](paper/RESULTS.md).*

**SAID Laboratory** · [saidlaboratory/MARC](https://github.com/saidlaboratory/MARC)

</div>
