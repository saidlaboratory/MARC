# MARC Fixing Plan

**Written:** July 19, 2026 · **Deadline:** July 27, 2026 (8 days)
**State:** [PR #50](https://github.com/saidlaboratory/MARC/pull/50) open (learned solver now converges); no paper draft exists; eval suite saturated.

Every flaw below has: evidence (file refs), why it matters, the fix, effort, and a suggested owner based on who built the area. Priorities:

- **P0** — paper-blocking; must land before July 27
- **P1** — materially strengthens the paper; do if time allows
- **P2** — post-deadline (v2 / camera-ready / next cycle)

---

## A. Evidence & science flaws

### A1 · The eval suite is saturated — H1 has no solve-rate signal ‖ P0

**Evidence:** `results/p1_baselines/metrics.json`, `paper/figures/summary_table.md`, `results/p2_main/cot_baseline.json` (PR #50). Learned solver, deterministic `refine`, and Gemini CoT all score 1.000 in-dist and 1.000 held-out. Generalization gap = 0.000 for every solver.

**Why it matters:** H1 ("smaller generalization gap than CoT") cannot be tested when everyone is at ceiling. A reviewer reads "gap 0.000 vs gap 0.000" as *no evidence*, and the claim collapses.

**Fix:**
1. Add a hard tier to the procedural generator (`marc/data/templates.py`): larger systems (4x4, 5x5), mixed nonlinear factors (reuse the P3 toy families in `marc/data/` — `sum_product`, `bilinear_product`, `quadratic_link` are already implemented), and tighter checker tolerance.
2. Target a regime where solvers land in the 0.3–0.8 band so curves can separate. Calibrate with `refine` first (cheap), then run learned + CoT.
3. Re-run the full grid: `refine`, learned (PR #50 checkpoint), CoT — in-dist, held-out, perturbed, length-extrapolation.

**Effort:** ~2 days (1 generator, 1 runs). **Owner:** Akash (generator, built P1 dataloader + P3 toys) + Sparsh (runs, owns eval harness).
**Dependency:** PR #50 merged (C2).

### A2 · Headline paper numbers come from the hand-coded solver, not the learned model ‖ P0

**Evidence:** `paper/figures/summary_table.md` — "_Solver: **refine**_" footnote on the main results table; `refine` is the deterministic energy-gradient baseline (`marc/refine/iterative.py`), not the trained denoiser.

**Why it matters:** presenting hand-crafted-solver numbers as the system's headline result is the single fastest way to get rejected — reviewers will (correctly) read it as misattribution. It also undercuts our own contribution: the learned solver *does* now solve the convex suite (PR #50).

**Fix:** regenerate `summary_table.md` and all three figures from the PR #50 learned checkpoint on the new hard suite (A1). Report `refine` explicitly as a *classical baseline* row, never as the headline. Every table in the paper states which solver produced it.

**Effort:** 0.5 day once A1 runs exist (plotting is scripted: `scripts/plot_results.py`). **Owner:** Sparsh.

### A3 · The learned solver is only validated on the convex suite; geometry is untouchable ‖ P1

**Evidence:** `results/p4_scale/roadmap.md` — "cannot handle the geometry domain at all"; no `ProblemGenerator`-compatible geometry template exists (`marc/data/geometry.py` is eval-only), so the model cannot even *train* on geometry.

**Why it matters:** the paper claims a general substrate ("same template extends to geometry") but the learned model has seen exactly one problem family. One extra domain turns "it works once" into "it transfers."

**Fix:**
1. Write `GeometryTemplate` for the training generator mirroring `LinearSystem2x2Template` (coordinates as variables, squared-distance factors — the eval-side encoding in `marc/data/geometry.py` is the spec).
2. Retrain Stage A on mixed algebra+geometry data; evaluate on held-out geometry.
3. If it doesn't converge in time, scope the paper's claims to equation systems and present geometry as an eval-only domain for the classical pipeline (which already works — `scripts/demo_end_to_end.py`).

**Effort:** 1 day template + 1 day train/eval. **Owner:** Davin (built the geometry domain). **Fallback is free:** honest scoping costs a paragraph.

### A4 · Guidance ablation is degenerate ‖ P0

**Evidence:** `paper/figures/summary_table.md` — "best w = 0.0 (solve rate 0.000)". The ablation says guidance is useless *and* nothing solves anything, because the checkpoints are toy-scale (D=128, CPU-minutes, `scripts/train_p2_checkpoints.py`).

**Why it matters:** CAS guidance is the paper's central mechanism. An ablation showing best-guidance-weight-is-zero actively refutes our own architecture. It cannot appear in the paper in this form.

**Fix:** re-run the guidance sweep on the PR #50 converged checkpoint (D=256, the one that actually solves). If w=0 still wins on the hard suite, that is a real finding that must reshape the narrative (the diffusion-proposes/refine-polishes hybrid becomes the story — see A8). If the sweep shows w>0 helping, the mechanism is validated. Either way: run it, don't ship the degenerate version.

**Effort:** 0.5 day (sweep is scripted). **Owner:** Quang (owns training). **Dependency:** A1 hard suite, PR #50.

### A5 · Purist-reward ablation is uninformative ‖ P1

**Evidence:** `summary_table.md` — "shaping gain 0.000 (standard 0.000 vs. purist 0.000)". Zero vs. zero distinguishes nothing.

**Fix:** same as A4 — re-run on the converged checkpoint at a difficulty where solve rates are off the floor. If still indistinguishable, drop the ablation from the paper (one sentence: "reward shaping showed no effect at this scale").

**Effort:** piggybacks on A4's runs. **Owner:** Quang.

### A6 · CoT baseline is too thin to carry the H1 comparison ‖ P0

**Evidence:** `results/p2_main/cot_baseline.json` — N=25 problems, `n_samples: 1`, Gemini flash-lite.

**Why it matters:** this is the paper's *only* external comparison. N=25 with one sample per problem has confidence intervals wide enough to drive a truck through, and "flash-lite" invites a "you compared against a weak model" review.

**The good news buried here:** the perturbation signal is real and currently our best H1 evidence — CoT's solve rate *drops 0.36* under constant perturbation on held-out structure while graph-side solvers drop 0.0 (`perturbation_robustness` = solve-rate drop, `marc/eval/metrics.py:11`). That is exactly the "recall detector" CONCEPT.md promised. It deserves to be a headline figure, not a footnote.

**Fix:**
1. Scale to N≥100 problems on the hard suite, `n_samples≥4` (pass@k needs k>1), with a stronger model tier as the primary baseline (keep flash-lite as a second point; the provider-aware runner with backoff+resume already supports this — `marc/eval/baselines/cot_baseline.py`).
2. Sweep perturbation Δ ∈ {0.1, 0.5, 1.0, 2.0} for both CoT and learned solver → one robustness-curve figure.
3. Report CIs (Wilson intervals) on every solve rate.

**Effort:** 1 day (mostly API wall-clock; resume cache exists). Budget API spend. **Owner:** Akash (built the CoT baseline).

### A7 · H2 (structure invention) is a null result on an untrained proxy ‖ P0 (framing) / P2 (science)

**Evidence:** `results/p3_h2/h2_report.md` — fixed vs. structure solve rate Δ = 0.00 on all three toys; the "structure model" is energy-guided best-of-k with an oracle, not the D3PM head (`marc/model/structure_head.py` exists but was never trained).

**Why it matters:** H2 is half the paper's hypothesis section. As evidence it currently shows: no solve-rate benefit, from a proxy, with an oracle energy. A reviewer will call this "H2 untested." The usage-rate numbers (39–100%) show routing *through* auxiliaries happens, not that it *helps*.

**Fix (deadline-realistic):** reframe, don't oversell. In the paper: H2 gets a "preliminary evidence" subsection — usage rates show the search exploits auxiliary structure when present; solve-rate parity is expected because the augmented graph is solution-equivalent by construction (the report says this). Explicitly list trained structure diffusion as future work.
**Fix (real, P2):** train the D3PM sampler over `StructureHead`, and build at least one toy family where the fixed graph is *unsolvable* without the auxiliary (current toys keep solution sets identical, so Δ=0 is baked in — this is a design flaw in the eval, not just the model). That family is what would make H2 falsifiable.

**Effort now:** 0 (writing only). **Owner:** whoever drafts §H2 (Quang).

### A8 · The entrapment result is real but narrow — and the hybrid solver complicates the story ‖ P0 (framing)

**Evidence:** PR #50 — deterministic descent 100% trapped vs. Langevin 0.475, reduction 0.525 ± 0.086, N=200, 95% CI excludes 0. Also PR #50: pure DDIM cannot reach checker tolerance; final system is "diffusion proposes, `refine()` polishes."

**Why it matters:** two review attacks are guaranteed: (1) "noise escapes local minima is textbook Langevin dynamics" — confirmatory, not novel; (2) "if `refine()` does the final solving, what is the learned denoiser buying you?" The second one is dangerous because right now we have no ablation isolating the denoiser's contribution.

**Fix:**
1. Add the missing ablation: **hybrid vs. `refine`-only from the same initialization**, on the hard suite — steps-to-solve and solve-rate. If diffusion proposals let `refine` succeed where cold-start `refine` fails or is slower, that's the paper's real claim, cleanly demonstrated. This is *the* most important new experiment after A1.
2. Frame entrapment as RQ2 validation (the README literally poses it as a research question with a falsification criterion — "falsified if injected noise does not reduce entrapment"). A pre-registered question answered with CIs is honest science, not a novelty claim.
3. Run entrapment on one non-convex family (the P3 toys have real local minima) to blunt "convex toys only."

**Effort:** 1 day. **Owner:** Sparsh (ran the 200-graph study) + Davin (DDIM path).

---

## B. Scope & infrastructure gaps (be honest, don't fix by July 27)

### B1 · No GPU-scale training ‖ P2
`roadmap.md` calls the D=512/L=8 full Stage-A+B plan (`results/p4_scale/scaling_notes.md`) "the single highest-leverage next step." True, but not in 8 days unless a GPU materializes today. If one does: launch the scripted plan immediately and let it run in the background — a mid-week checkpoint that beats D=256 upgrades every table. Otherwise: paper states scale explicitly and cites the scaling notes as the plan.

### B2 · NL parser covers 3 sentence templates ‖ P2
`marc/nl/parser.py` is real but closed-vocabulary. Paper must say "template-based formalization for three problem shapes; general autoformalization out of scope (CONCEPT.md defers it by design)." Do not demo NL input without that caveat.

### B3 · No formal (Lean) checker gate ‖ P2
Checker is numeric + SymPy-exact only (TECHNICAL_GUIDE §7's Lean gate unbuilt). Paper says "symbolic-exact verification; proof-kernel gate future work." Do not use the word "formally verified."

### B4 · Geometry training template missing ‖ covered by A3.

---

## C. Paper & process flaws

### C1 · There is no paper ‖ P0 — the critical path

**Evidence:** `paper/` contains 3 figures, `summary_table.md`, `learned_solver_fix.md`. No `.tex` anywhere.

**Fix:** scaffold AAAI-format LaTeX **today** (July 19). Section skeleton with owners:

| Section | Content source | Owner |
|---|---|---|
| Intro + method | CONCEPT.md, README (already paper-grade prose) | Quang |
| System | TECHNICAL_GUIDE + PR #50 fix write-up | Davin |
| Entrapment (RQ2) | A8 runs | Sparsh |
| H1: generalization + perturbation | A1/A6 runs | Akash |
| H2: preliminary | A7 reframing | Quang |
| Related work | README prior-art table (already structured) | anyone |

Write around placeholder numbers from day 1; drop in final numbers July 24–25. The intro/method/related-work ~60% of the paper has **zero dependency on new experiments** — nothing blocks writing today.

### C2 · PR #50 unmerged + docs on `main` contradict it ‖ P0

**Evidence:** `results/p4_scale/roadmap.md` on `main` says the learned solver "solves ~0% of the P2 suites"; PR #50 says 100%. Both true at their timestamps, contradictory to any reader now.

**Fix:** review and merge PR #50 (July 19–20), then sweep `roadmap.md`, `README.md` badges, and `scaling_notes.md` for stale claims. One person does the doc sweep in the merge PR.
**Owner:** Quang reviews, Sparsh sweeps docs.

### C3 · Every paper number needs provenance ‖ P0
Rule from today: no number enters the draft without the exact command + seed + commit recorded next to it (a `paper/PROVENANCE.md` table: figure → script → commit → seed). The 0.516-vs-0.525 entrapment discrepancy between `main` and PR #50 is precisely the kind of thing that becomes a rebuttal-period fire drill.

### C4 · License is TBD ‖ P1
README badge says `license-TBD`. AAAI code-release expectations + a public repo need a decision (MIT/Apache-2.0). 15 minutes; needs all four authors' sign-off.

---

## Schedule (July 19 → 27)

Two tracks in parallel. **Paper track never waits on the experiment track.**

| Date | Experiment track | Paper track |
|---|---|---|
| **Jul 19 (Sat)** | Review + merge PR #50 (C2) | Scaffold LaTeX, port intro/method (C1) |
| **Jul 20 (Sun)** | Hard-suite generator (A1) | Related work + system sections |
| **Jul 21 (Mon)** | Calibrate suite; launch learned/refine/CoT grid (A1, A6) | Entrapment section w/ existing numbers (A8.2) |
| **Jul 22 (Tue)** | Hybrid-vs-refine ablation (A8.1); guidance/purist re-run (A4/A5) | H2 preliminary section (A7); doc sweep lands |
| **Jul 23 (Wed)** | **DECISION GATE** — see below | Full draft assembled, placeholder numbers |
| **Jul 24 (Thu)** | Non-convex entrapment (A8.3); geometry stretch (A3) | Final numbers into tables/figures (A2) |
| **Jul 25 (Fri)** | Runs frozen; PROVENANCE.md complete (C3) | Internal full-team read |
| **Jul 26 (Sat)** | — | Revisions; license (C4); abstract polish |
| **Jul 27 (Sun)** | — | **Submit** |

### Decision gate (July 23)

Look at the hard-suite grid:

- **Learned solver separates from CoT** (better gap or robustness, CIs disjoint) → H1 gets a claims section; consider main-track framing.
- **No separation / CoT wins** → workshop framing: system + RQ2 entrapment + hybrid ablation as contributions, H1/H2 as preliminary. This is the default and the draft is written to survive it — the gate can only *upgrade* the paper, never stall it.

### Standing rules until submission

1. No number in the paper without a solver label, N, CI, and provenance entry.
2. `refine` is always labeled a classical baseline.
3. Nothing in the draft claims more than its section's evidence — the H2 report's honesty is the house style.
4. Main is frozen for anything but paper-supporting changes after July 23.

---

## Post-deadline (v2) queue

In leverage order, per `roadmap.md`: (1) GPU-scale Stage-A/B (B1) → re-run every table; (2) train D3PM structure diffusion + an aux-*required* toy family so H2 is falsifiable (A7); (3) geometry training template if it missed the deadline (A3); (4) Lean gate (B3); (5) LLM-assisted autoformalization with checker verification (B2).
