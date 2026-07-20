# MARC — Meeting Notes

**Date:** 2026-07-19 · **AAAI deadline:** 2026-07-27 (8 days) · **Branch of latest work:** `sparsh/a1-hard-suite`

Snapshot for the team meeting: what changed, where we stand, decisions needed, and next moves against the deadline. Written to be honest — the caveats are in here on purpose.

---

## 1. TL;DR

- **The learned solver now works.** It went from *diverging / 0% solve* to actually solving. Five real bugs fixed. (`paper/learned_solver_fix.md`)
- **Two genuine results in hand:** (a) noise escapes entrapment where deterministic descent is 100% trapped; (b) on high-dim non-convex problems the learned model does per-instance inference that beats classical refinement **and** a trivial prior.
- **NEW (today): the eval suite is de-saturated** — the #1 paper blocker (A1) is fixed, and the first hard-suite ablation shows the **learned hybrid beats refine** (0.625 vs 0.350 vs 0.000). This answers the "what does the denoiser add?" review attack.
- **Reality check:** these are toy problems (linear systems, bilinear systems, constructed traps). This is **not** competition math (IMO/USAMO) and not SOTA. It's a plausible AAAI submission **if scoped honestly**.
- **Still no paper draft (.tex).** That is the critical path (C1).

---

## 2. What we accomplished this session

### Engineering / correctness
- **Fixed the learned diffusion solver (0% → solves).** Root causes: (1) denoiser never received the noised input `x_t`; (2) equation constants absent from graph tensors; (3) timestep didn't condition variables; (4) inference guidance exploded; (5) a 1-variable `squeeze` bug. Method = diffusion proposal + energy-descent polish.
- **Architectural contribution — per-instance inference.** The model was collapsing to "predict the mean solution." Fix: condition each variable on its **incident constraint constants** + a **direct constant→output skip** (message-passing LayerNorm was washing out the magnitude). Recovers roots at mean-error 0.9 (was 5.4).
- **Test coverage:** 159 → 208 tests, all green. New coverage for structure toys, embeddings, rollout, CoT baseline, dimension-scaling, hard templates.
- **CoT baseline runs on Gemini** (free-tier, OpenAI-compatible endpoint, backoff + resume cache).
- **Merged to `main`:** PR #50 (solver converges + ablations + entrapment), PR #51 (inference + scaling + SUMMARY). Quang's `FIXING_PLAN.md` merged (#52).

### Science results
| Result | Numbers | File |
|---|---|---|
| Learned solver converges (convex) | 0% → 100% in-dist & held-out | `paper/learned_solver_fix.md` |
| Entrapment (RQ2) | deterministic 100% trapped → Langevin 0.475; reduction **0.525 ± 0.086**, N=200 | `results/p1_entrapment/` |
| Dimension scaling | learned beats determ (0) / Langevin (→0) / mean-prior (0); decays 0.68→0.10 over n=1..6 | `paper/dimension_scaling_result.md` |
| **A1 hard suite (new)** | bilinear traps de-saturate: determ 0.000, Langevin 0.175–0.35 | `scripts/run_hard_eval.py` |
| **A8.1 hybrid ablation (new)** | learned_hybrid beats refine on both hard families: BilinearSystem **0.625** vs 0.350/0.000; BilinearProduct **0.725** vs 0.125/0.000 | `results/p_hard/hard_eval.json` |

---

## 3. Fixing-plan status (P0 items)

| ID | Item | Status |
|---|---|---|
| C2 | Merge PR #50 + doc sweep | ✅ merged (#50, #51) |
| **A1** | Eval suite saturated → hard tier | ✅ **done today** (bilinear templates de-saturate) |
| **A8.1** | Hybrid vs refine-only ablation | ✅ **done — learned wins both families** (0.625 vs 0.350; 0.725 vs 0.125) |
| A2 | Headline numbers from learned model, not refine | ✅ hard-suite headline table + figure (`paper/figures/hard_suite_table.md`, `fig_hard_suite.pdf`) |
| A4/A5 | Guidance / purist ablations degenerate | ✅ reframed (`paper/ablation_reframe.md`): replace with A8.1; guidance sweep on hard checkpoint optional (P1) |
| A6 | CoT baseline too thin (N=25, k=1) | ⏳ needs Gemini key + N≥100, k≥4, stronger tier |
| A7 | H2 null result | ✅ reframed as "preliminary" (`paper/h2_reframe.md`) |
| A3 | Geometry training template | ⏳ P1 — not started (Davin) |
| A8.3 | Entrapment on a non-convex family | ⏳ P0 framing — bilinear suite now exists for it |
| C1 | **No paper (.tex)** | ❌ **not started — critical path (Quang)** |
| C3 | Provenance table | ✅ `paper/PROVENANCE.md` started (R1–R12) |

---

## 4. AAAI timeline (from FIXING_PLAN.md)

Two parallel tracks; **the paper track must never wait on experiments.**

- **Now → Jul 21:** hard-suite generator (done) → run learned/refine/CoT grid on it (A1/A2/A6). Start LaTeX scaffold + intro/method/related-work (≈60% of the paper has zero experiment dependency).
- **Jul 22:** finish hybrid ablation (A8.1) + guidance/purist re-runs (A4/A5).
- **Jul 23 — DECISION GATE:** does the learned solver separate from CoT on the hard suite?
  - *Yes (CIs disjoint):* H1 claims section, consider main-track framing.
  - *No / CoT wins:* workshop framing — system + entrapment (RQ2) + hybrid ablation as the contributions; H1/H2 preliminary. **The draft is written to survive this either way.**
- **Jul 24–25:** final numbers into tables; provenance table; internal read.
- **Jul 26–27:** revisions, license, abstract, **submit**.

---

## 5. Decisions to make at the meeting

1. **Main-track vs workshop framing.** Default to workshop-survivable; the Jul 23 gate can only upgrade. Agree we write to the honest floor.
2. **Owners for the paper sections** (per plan): Intro/method (Quang), System (Davin), Entrapment (Sparsh), H1 (Akash), H2 preliminary (Quang). **Who starts the .tex today?**
3. **CoT baseline budget.** A6 needs a stronger model tier + N≥100 → real (small) API spend. Approve? Which model as primary baseline?
4. **Scope honesty in claims.** Agree the house style: `refine` always labeled a classical baseline; every number carries solver + N + CI + provenance; no "formally verified", no unqualified geometry/NL claims.
5. **License** (C4): MIT vs Apache-2.0 — needs all four sign-offs (15 min).
6. **GPU?** If one materializes, launch the D=512/L=8 scale plan immediately (a mid-week checkpoint upgrades every table). If not, scope scale explicitly.

---

## 6. Open risks / honest caveats (say these out loud)

- **Toy problems only.** Biggest solved: 3×3 linear or 2–3 var bilinear. Not natural math, no proofs, **not IMO/USAMO** — a different research program.
- **Checkpoints are toy-scale** (D=128, CPU-minutes). Not a scaling claim.
- **Dimension-scaling degrades at n=6** (learned 0.10) — the honest weak spot; it beats baselines but is not dimension-immune.
- **H2 is a null result** on an untrained proxy; must be reframed as preliminary, not sold.
- **Mechanism novelty is moderate** — amortized learned proposal beating blind search is a known principle; our contribution is the concrete constraint-solving instance + the conditioning architecture.
- **API keys (OpenAI + Gemini) were pasted in a working chat — rotate them.**

---

## 7. Immediate next moves (this week, concrete)

1. **Finish A8.1** (running) → if hybrid > refine on both families, it's a headline ablation. Commit `run_hard_eval.py` + results.
2. **A2:** regenerate `summary_table.md` + figures from the learned checkpoint on the hard suite; label `refine` as a baseline row.
3. **A4/A5:** re-run guidance/purist sweeps on the hard suite (degenerate "w=0, 0.000" numbers cannot ship).
4. **A6:** with a Gemini/OpenAI key, scale CoT to N≥100, k≥4, add Wilson CIs, sweep perturbation Δ.
5. **C1:** scaffold the AAAI LaTeX **today** and port intro/method/related-work from CONCEPT.md / README.
6. **C3:** start `paper/PROVENANCE.md` (figure → script → commit → seed) now, not at the end.
