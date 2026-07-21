# MARC — session handoff (for a fresh context window)

**Written:** 2026-07-21 · **AAAI deadline:** 2026-07-27 · **Repo:** `saidlaboratory/MARC`
(you are on the MacBook M5, MPS; torch 2.12; `PYTHONPATH=.` needed to run scripts).

Read this first, then `paper/RESULTS.md` + `paper/PROVENANCE.md` (canonical) and
`AAAI_READINESS.md` (the strategic call). Everything below is honest; do not re-inflate claims
that were rigorously falsified.

---

## 1. What MARC is
A neuro-symbolic **diffusion constraint solver**: math problems → **factor graphs** (variables +
symbolic residual constraints); a **GNN denoiser** proposes real-valued assignments by reverse
diffusion; a **CAS (SymPy)** gives exact residuals/energy/gradient + an exact accept checker.
Classical fallback = energy-gradient / annealed-Langevin refinement (`marc/refine`).

## 2. The single most important thing: the honest verdict
**Workshop-level, NOT main-track.** Over this session we rigorously tested every plausible
main-track angle and **all three are negative**:
1. *Learned proposal beats classical search* → **No.** With the proper **random-multistart +
   polish control**, the learned model **ties/loses to random restart** on the hard (bilinear)
   and coupled families. Its only clean win is on **independent high-dim traps** (dimension
   scaling, learned 0.925 vs random 0.0 at n=4) — but that is amortizing the curse of
   dimensionality on *separable* problems, and it **vanishes under coupling**.
2. *Structure/auxiliary invention* → **No.** Adding the "lemma" auxiliary **hurts** the numeric
   solver (higher-dim search); it only helps symbolic/staged solving = plain CAS. The trained
   structure **policy** beats fixed/no-context (p<1e-4) but **ties random** selection (p=0.28).
3. *LLM (Gemini) + MARC verified solving* → **No.** LLM-direct ≈0.80 vs formalize-then-solve
   ≈0.00 on MATH; formalization is a lossy bottleneck.

**Do not claim "learned solver beats classical."** The honest, defensible framing: a
neuro-symbolic diffusion solver + entrapment analysis + a rigorous *characterization of when
learned proposals help constraint solving (and when they don't)*, controls and negatives
included.

## 3. What IS solid (real, honest)
- **Learned solver converges** (was diverging/0% → 100% on convex; 5 bugs fixed — see
  `paper/learned_solver_fix.md`). Trained at scale this run: Stage-A loss **0.60**.
- **Entrapment (RQ2):** deterministic descent 100% trapped → annealed Langevin 0.475; reduction
  **0.525 ± 0.086** (95% CI excludes 0, N=200). Real but textbook Langevin.
- **Dimension-scaling crossover** (independent traps): random wins n≤2, **learned wins n≥3**
  (curse-of-dimensionality amortization). Honest, but narrow/synthetic.
- Everything has Wilson CIs + 2-proportion z-tests (`marc/eval/metrics.py`).

## 4. Results table (this session, honest)
| Experiment | Learned vs baselines | Verdict |
|---|---|---|
| Convex (p1/main/CoT) | all 1.000, gap 0 | saturated, no signal |
| Hard bilinear | learned = random (0.55/0.68/0.68), fails CircleLine (0.00) | ties random |
| Coupled chained bilinear | learned ≤ random at every n (0.23–0.48) | ties/loses |
| Dimension scaling (independent) | learned 0.95/0.95/0.975/0.925/0.25 vs random 1.0/0.725/0.075/0/0 | **learned ≫ random n≥3** |
| Structure-invention policy | > fixed & no-context (p<1e-4); = random (p=0.28) | ties random |
| Geometry (refine) | 0.56 in-dist | non-saturated real-ish domain (future signal?) |
| CoT (Gemini flash-lite) | 1.0 convex | saturated |

## 5. Overnight run (this session) — outcome & lessons
`scripts/run_overnight.py` (crash-safe manifest). Ran to completion: **17 ok / 4 skipped / 1
failed**.
- **Stage-A trained (D512/L8, loss 0.60, ~30 min on MPS).**
- **Stage-B GRPO DIVERGED** (loss 12.9→134k, reward ~−9e8, ~35min/epoch) — I cut it.
- **`eval_main_learned` killed** after 3h+ stuck (D512 diffusion+guidance loop too slow on MPS
  over the full perturbation/length suite; convex/saturated anyway).
- **CRITICAL harness gap:** the eval scripts (`run_hard_eval`, `run_coupled_eval`,
  `run_dimension_scaling`) **retrain their own small models and do NOT load the D512 checkpoint**
  — so scaled training never reaches the differentiating evals. **Fix before any future scaled
  run:** wire `MARC_CKPT` into those eval scripts. Until then, scaled runs are wasted on the
  convex saturated evals.
- Relaunch trick that worked: `--skip tests,train_stage_a,train_stage_b`; kill a stuck eval
  subprocess and the harness marks it failed and continues.

## 6. Git / repo state (IMPORTANT — unfinished)
- **On branch `main`.** There is a **local commit not yet pushed** (the overnight results in
  `OVERNIGHT_RESULTS.md`); `git push origin HEAD:main` was **rejected — needs `git pull` first**
  (main advanced). **First action for next context: `git pull --no-rebase origin main`, resolve
  any conflict in `OVERNIGHT_RESULTS.md`, then push.**
- Merged this session: PR #75 (stale-test fix + status). PR #56 (coupled negative) and branches
  `sparsh/structure-invention`, `sparsh/aaai-readiness` carry the negatives (may be open).
- Test suite: **green (365 passed)** — fixed a stale `test_invention_eval` Holm-family assertion.
- Commits are authored by **@ImSpxrsh, no Claude co-author** (keep this convention).

## 7. Canonical docs (avoid the sprawl)
- `paper/RESULTS.md` — all results, corrected framing (READ THIS).
- `paper/PROVENANCE.md` — every number → command/seed/commit (R1–R14).
- `AAAI_READINESS.md` — the workshop-vs-main-track call + reframe table.
- `OVERNIGHT_RESULTS.md` — this run's outcome.
- Source notes: `paper/{learned_solver_fix,dimension_scaling_result,math_coverage,related_work,
  ablation_reframe,h2_reframe,structure_invention_negative,llm_verify_negative}.md`.
- Process: `FIXING_PLAN.md`, `MEETING_NOTES.md`, `SUMMARY.md`, `RUNBOOK_SPARSH.md`.

## 8. Next steps (priority)
1. **Push the pending results commit** (git pull → push; §6).
2. **Decide target = workshop** (default) unless a new positive appears. Start the `.tex` — ~60%
   (intro/method/related-work) is experiment-free; `related_work.md` positions vs DIFUSCO /
   Langevin-CO / amortized inference.
3. If still chasing main-track: only remaining shots are **(a) wire the D512 checkpoint into the
   evals and see if scale changes coupled/hard** (cheap, likely still negative), **(b) a real
   domain with a genuine positive** (geometry at 0.56 is the least-saturated lead), or **(c) new
   method**. All are longer than the deadline; be honest about odds.
4. **Consolidate `paper/*.md`** into `paper/notes/`, keep RESULTS + PROVENANCE at top level.

## 9. Security / hygiene
- **Rotate the OpenAI + Gemini API keys** — both were pasted into this working chat (exposed).
  Gemini free tier is quota-limited (~flash 20/day; flash-lite higher). Keys were used only via
  out-of-repo env files, never committed, and scrubbed after use.
- Big checkpoints live in `checkpoints/scale_D512_L8/` (gitignored, ~426 MB each).

## 10. One-line status
Rigorous, honest, working system with one narrow positive (high-dim independent amortization);
**workshop-ready, not main-track**; overnight run confirmed the findings and exposed the
harness/checkpoint wiring gap; a pending results commit needs pull+push.
