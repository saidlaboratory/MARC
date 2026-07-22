# MARC — session handoff (fresh-context, 2026-07-22)

**AAAI abstract deadline:** 2026-07-27. Repo `saidlaboratory/MARC`.
Run scripts with `PYTHONPATH=.` (some also need `:scripts`); MacBook, torch 2.12, MPS.
Canonical docs: `paper/RESULTS.md`, `paper/PROVENANCE.md` (every number → command/seed/branch),
`paper/ABSTRACT.md`, `paper/tex/marc.tex` (the paper), `paper/notes/*.md`.

## Conventions (IMPORTANT)
- Commits authored by **@ImSpxrsh, NO Claude co-author** (git is already configured for this).
- **Do NOT put "Generated with Claude Code" in PR bodies** (team preference).
- Every rate carries N + Wilson CI or z-test. Negatives are reported as primary findings.
- `main` push is blocked by the harness; work on a branch, push, open a PR; the team merges.
- Abstract is em-dash-free (AI-tell hygiene); the paper body uses `---` as house style (fine).
- Before staging: `git checkout -- marc/graph/__pycache__` (tracked pyc noise reappears).

## The paper in one line
A **controlled study / characterization** of *when* learned proposals help continuous algebraic
constraint solving. Two acts: (1) value-diffusion is mostly a boundary/negative unified by a
**factorization law**; (2) relocating learning to the **discrete structural-repair** decision is a
decisive **positive** (the repair ranker). MARC (factor graphs + GNN diffusion denoiser + CAS +
exact checker) is the *instrument*, not the claimed contribution — keep it framed that way (the
diffusion model's novelty is weak; that is fine because we do not claim it as the contribution).

## DONE + merged to main this session
- **Abstract v5** (repair co-headlined + geometry + real-systems validation), em-dash-free, ~265w.
  `paper/tex/marc.tex` + `paper/ABSTRACT.md`. (PRs #113, #115.)
- **Geometry learned arm (R25)** — the law's live prediction, TESTED and refuted, which sharpened
  it: a trained denoiser ties random on the coupled geometry point-chain and collapses with it
  (0.625/0.175/0.025/0.000, 0/4 wins). Corrected the law to **two conditions**: learning helps iff
  (1) reachability collapses AND (2) the solution is per-variable separable. (PR #114.)
- Earlier merged: factorization law R9 (MAE 0.012), repair ranker R20–R24 (0.997 vs 0.236,
  p<1e-70, beats cheap-probe on accuracy+cost, 0.982±0.006 multiseed), fixes #103/#104, R8 regen.

## OPEN PR / branches
- **PR #116 `sparsh/real-systems`** — **External validity (R26)**: eight NAMED real systems
  (robotics IK, trilateration, Rosenbrock/Himmelblau, cyclic-4, circle/conic). Classical **LM
  solves 8/8**; gradient-polish random restart 4/8. No learning-favorable regime (real = low-dim +
  coupled → classical suffices, consistent with the law). Answers **synthetic-only**. Merged main
  in; mergeable. **Action: get it merged.**
- Branches `sparsh/crossover-families` work below is not yet pushed at handoff — see next section.

## WINS LANDED this session (PR #117 open) — both experiments finished
1. **Crossover replication + learned-beats-LM** — the strongest new win.
   `PYTHONPATH=.:scripts python3 scripts/run_crossover_families.py --K 8 --test 40 --epochs 200 --ntrain 200`
   → `results/p_scaling/crossover_families.{json,log}`. Establishes TWO things:
   (a) the R5 amortization crossover **replicates** across 3 structurally different separable
   families (baseline / double_well / wide_roots) — not one designed family;
   (b) the learned proposal beats **LM (the strong classical solver), not just random** — LM ALSO
   collapses ~p^n (measured on baseline: LM 0.825/0.575/0.200/0.100/0.000 at n=1/2/3/4/6). This
   closes the "did you compare to a real solver?" attack on R5.
   Files already created (uncommitted): `scripts/run_crossover_families.py`,
   `tests/test_crossover_families.py` (passing). **DONE — landed as R27 in PR #117.** (RESULTS.md +
   PROVENANCE, add a paper table/paragraph to the R5/scaling section of `marc.tex`, commit to a
   branch `sparsh/crossover-families`, push, PR (NO Claude footer).
2. **3-seed geometry hardening** — `scripts/run_pointchain_learned.py --seeds 3`
   → `results/p_geometry/pointchain_learned_3seed.log`. Hardens R25 (ties reproduce). When done,
   update R25's note to "3 seeds, tie robust".

## Honest WIN inventory (what the paper can claim)
- **STRONG:** repair ranker (R20–R24) beats controls + cheap-probe on accuracy AND cost, multiseed-robust.
- **STRONG (new):** learned beats BOTH random and LM at high-dim separable; crossover replicates.
- **SOLID:** entrapment R2 (0.525±0.086); factorization law R9 (parameter-free MAE 0.012, 3-family
  validation incl. the geometry refutation); structure-selection R16 (0.410 vs 0.200 in-pattern, sig).
- **EXTERNAL VALIDITY (new):** R26 real systems (LM 8/8) — answers synthetic-only.
- **HONEST NEGATIVES:** coupling kills value-learning (R7); geometry (R25); CircleLine; transfer 2/4
  (R4); K=16 repair advantage gone (cost-only, R24).

## Highest-value NEXT wins (priority order)
1. **A real-domain POSITIVE = the single biggest lever for main-track.** The repair positive is on
   a synthetic task construction. The real analog is **geometry auxiliary construction**
   (AlphaGeometry's domain): geometry problems unsolvable without an auxiliary point/line, a menu of
   candidate auxiliaries (exactly one solvable, certified), train the ranker. Hard but decisive —
   moves the paper from "coin-flip" to "strong."
2. **More separable families** for the crossover (extend `run_crossover_families`) — cheap breadth.
3. **Repair ranker: more generalization axes** (held-out patterns, harder negatives).
4. **Tighter CIs** on R5/R25 (more seeds).

## Honest main-track verdict
Borderline / credible-but-not-a-lock (~coin-flip). Strengths: the law (falsifiable, parameter-free,
3-family), the repair positive, the controlled protocol, external validity, learned-beats-LM.
Weaknesses: strongest positive is a synthetic task construction; diffusion model weak/negative
(fine if framed as instrument); fundamentally an analysis paper. Biggest lever: a real-domain
positive (item 1).

## Health
- Full suite `PYTHONPATH=. python3 -m pytest -q` was **404 passing**; +new tests → ~406–408.
- Result JSONs whitelisted in `.gitignore` (results/ is ignored; add `!results/p_.../x.json` per file).
