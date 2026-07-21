# AAAI readiness — honest verdict, reframe, next steps

> **Superseded for writing purposes (2026-07-21):** the project adopted the **orchestrator reframe**
> (learned structure proposals + classical value solvers + checker). Start from [OUTLINE.md](OUTLINE.md)
> (the team writing guide) and [README.md](README.md) (the framing); this file remains the
> point-in-time readiness verdict that motivated the reframe.

**Date:** 2026-07-20 · **Deadline:** 2026-07-27 (7 days) · **Canonical results:** `paper/RESULTS.md`
+ `paper/PROVENANCE.md`. Everything else in `paper/*.md` is working notes (consolidate — see §5).

---

## 1. Is this AAAI-ready? — **Not for the main track as-is. Workshop-viable with honest framing.**

Be blunt: tonight's rigor pass (adding the random-multistart control) was the right thing to do,
and it **shrank the claim**. The honest state:

- **What's solid:** the system works end-to-end; the learned solver converges; entrapment (noise
  escapes deterministic traps) is real with CIs; the amortized-inference **crossover** (learned
  beats random search at n≥3) is genuine and controlled.
- **What's weak for a main-track bar:**
  1. **The learned model has no advantage over *random multi-start* on the hard suite** — the
     "win over classical refinement" is really the *hybrid recipe*, not the denoiser. (A8.1.)
  2. **The one place the denoiser wins (high-dim crossover) is on *synthetic, independent*
     bundled traps.** A reviewer will say: "random search failing in high dimension is the curse
     of dimensionality; a model that learns each variable's marginal root is expected to win —
     demonstrated only on toy problems it was built for." That criticism is fair right now.
  3. **Entrapment = textbook annealed Langevin** (novelty is confirmatory, not new).
  4. **All problems are synthetic**; MATH coverage is 0/48; no real domain.
  5. **No `.tex` exists yet.**

**Verdict:** a defensible **workshop** paper (honest system + entrapment RQ + the crossover as
preliminary evidence). **Main-track needs the §3 experiment to land** — otherwise the novel
contribution is too narrow.

---

## 2. What must be reframed (do NOT ship the old framing)

| Old (wrong) framing | Corrected framing |
|---|---|
| "Learned diffusion solver beats classical refinement." | "A proposal+polish **hybrid** beats cold-start Langevin; the *learned* proposal beats **random** search only in high dimension." |
| Headline = A8.1 hard-suite numbers. | Headline = the **dimension-scaling crossover** (learned > random for n≥3). A8.1 becomes "the hybrid recipe" evidence, with the random control shown. |
| "Solves math problems." | "Numeric constraint solver; 0/48 MATH coverage; targets the constraint-shaped slice." (Reality check, not a result.) |
| Report `refine` numbers as system results. | `refine`/`random` are baselines; always labelled; every number has N + CI/z-test. |
| "Structure invention" as a claim. | "Menu-based structure selection (with predicted defining value)" — built and trained; clean-protocol numbers pending (see §3b, `paper/notes/REVIEW_ATTACKS.md` #2/#3). |

---

## 3. The decisive experiment — RAN IT. Result: the learned advantage does NOT survive coupling.

We built the coupled chained-bilinear family (`x_i+x_{i+1}=s_i, x_i·x_{i+1}=p_i`) and ran the
scaling with the existing controls (R7 in `RESULTS.md`). **Outcome: the learned model ties or
loses to random restart at every dimension (0/5).** The R5 high-dim advantage was an
**independence artifact** — it required per-variable-separable solutions (memorizable marginals)
*and* random restart collapsing (must hit all n basins by chance). Under coupling, neither holds,
and the diffusion model provides **no advantage over random search + refinement**.

**Implication:** the "amortized learned proposal beats classical search" route to a main-track
claim is **closed**. What remains genuinely solid is narrower: the learned solver *converges*
(engineering), entrapment/noise escapes deterministic traps (real but textbook Langevin), and the
hybrid recipe beats cold-start Langevin (but so does random restart). That is a **workshop-level**
contribution.

## 3b. The remaining main-track shots
- **Menu-based structure selection (with predicted defining value).** BUILT: the D3PM
  structure policy is trained on the aux-required families (fixed graph certified
  unsolvable without the auxiliary), and the end-to-end eval harness runs it with
  positive/negative controls. Honest naming: the model selects one of K candidate
  auxiliary structures and predicts its defining value — it does not synthesize
  structure from an open vocabulary, so claim-language is "selection", not
  "invention". **Clean-protocol numbers pending**: the preliminary run is withdrawn
  (test seeds == validation seeds, plus a train/eval data-source mismatch in the
  harness — both now fixed; see `paper/RESULTS.md` R8 and `paper/notes/REVIEW_ATTACKS.md`
  #2/#3). Regenerate under the seed-space v1 protocol before citing anything.
- **A real (narrow) domain.** Get the model training+solving on geometry or a real
  polynomial-system slice — moves it from "synthetic toy" to "real problems." ~2–3 days, medium
  risk. Biggest credibility jump.
- Honest note: a rigorous *"when do learned proposals help constraint solving — and when they
  don't"* study (including the coupled negative) is a legitimate paper, but reads workshop, not
  main-track, at AAAI.

---

## 4. Next steps to the deadline (priority order)

1. **[P0, ~1 day] Coupled high-dim scaling experiment (§3)** — decides the paper's ceiling.
2. **[P0, today] Start the `.tex`** — intro/method/related-work are experiment-independent
   (`paper/notes/related_work.md` is ready). Write to the *corrected* framing (§2).
3. **[P0] Pick the target** — AAAI main vs an AAAI/NeurIPS workshop vs a different venue. Default
   to **workshop** unless §3 lands; decide at the team meeting.
4. **[P1] CoT baseline properly** (needs a fresh, funded key): N≥100, k≥4, stronger model,
   Wilson CIs — only if we want an H1/LLM comparison. Otherwise cut it.
5. **[P1] One honest limitations paragraph** — CircleLine failure, cross-family 2/4, learned
   loses to random at low n, synthetic-only, no proofs. Reviewers reward this.
6. **[P2] Rotate the OpenAI + Gemini keys** (exposed in the working chat).

---

## 5. Doc consolidation (the sprawl you flagged)

There are ~10 `paper/*.md` + root docs. Keep **two canonical**, treat the rest as source notes:

- **Canonical:** `paper/RESULTS.md` (all results, corrected) · `paper/PROVENANCE.md` (every number's
  command/seed/commit).
- **Source notes (fold into the .tex, then archive):** `learned_solver_fix.md`,
  `dimension_scaling_result.md`, `math_coverage.md`, `related_work.md`, `h2_reframe.md`,
  `ablation_reframe.md`, `hard_suite_table.md`.
- **Process docs (not paper):** `FIXING_PLAN.md`, `MEETING_NOTES.md`, `SUMMARY.md`, this file.

Suggest: move source notes under `paper/notes/` so `paper/` top level is just the `.tex` + the two
canonical `.md`s.

---

## Bottom line
Honest, rigorous, working system with one genuine (but narrow) result. **Workshop-ready now;
main-track hinges on the §3 coupled-family experiment.** Don't oversell — the corrected framing is
what makes it survivable. Decide the target at the meeting; start the `.tex` today regardless.
