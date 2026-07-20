# AAAI readiness — honest verdict, reframe, next steps

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
| H2 (structure invention) as a result. | H2 = preliminary (Δ=0 is baked into solution-equivalent toys). |

---

## 3. The ONE experiment that decides main-track vs workshop

**Does the learned advantage survive on a *coupled*, non-trivial high-dimensional family?**

The current crossover uses **independent** bundled traps — so "learn each variable's marginal"
suffices, and a skeptic dismisses it. The decisive test: a family where variables are **coupled**
so the solution is *not* a product of per-variable marginals (e.g. a chained system
`x_i + x_{i+1} = s_i`, `x_i · x_{i+1} = p_i`, or a sparse random non-convex system), scaled in n.

- **If learned still beats random restart at high n on the coupled family** → the model is doing
  real joint amortized inference, not marginal memorization. **That is a main-track result.**
- **If it collapses to random on coupling** → workshop framing; report honestly as a limitation.

This is ~1 day (build the coupled template + calibrate + run the scaling with the controls that
already exist). **Highest-leverage next action.**

---

## 4. Next steps to the deadline (priority order)

1. **[P0, ~1 day] Coupled high-dim scaling experiment (§3)** — decides the paper's ceiling.
2. **[P0, today] Start the `.tex`** — intro/method/related-work are experiment-independent
   (`related_work.md` is ready). Write to the *corrected* framing (§2).
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
