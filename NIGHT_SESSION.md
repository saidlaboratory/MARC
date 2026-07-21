# Overnight session — the factorization law (2026-07-21)

**Branch:** `sparsh/crossover-theory` (committed locally, **not pushed** — open a PR when ready).
**Starting point:** HANDOFF.md's honest verdict (workshop-level; R5 positive + R7 negative solid).
**What this session added:** the unifying *scientific* result the prior state lacked — a
falsifiable, parameter-free **law** that predicts both R5 and R7 (and now a real domain) from
one measured quantity. This is the main-track-shaped contribution: an analysis/understanding
result with a general principle, not a new SOTA claim. Nothing falsified was re-inflated.

## The result (R9 — see `paper/notes/crossover_law.md`, `paper/RESULTS.md`)
All methods share one polish + one checker. Define single-start reachability `q(n)`. Then
best-of-K random restart is exactly `1-(1-q(n))^K`. The regime is set by how `q(n)` decays:
- **Separable (independent traps):** `q(n)=v^n` provably. Measured slope **−1.03 (R²=0.98)**,
  v=0.27; a single constant reproduces the whole random curve **parameter-free, MAE 0.012**;
  expected restarts explode **3.7→600**. Random collapses ⇒ learning wins (R5).
- **Coupled bilinear:** slope **−0.13 (R²=0.96)**, expected restarts flat **2→4**. Random
  survives ⇒ learning ties, classical LM dominates (R7).
- **Geometry (real domain, new):** syntactically coupled but reachability **collapses**
  (slope −0.77, R²=0.999; q=0.653/0.147/0.027/0.007) — per-point reflection ambiguity +
  spurious basins compound. ⇒ the diagnostic
  is the *measured slope*, not the syntactic label; geometry is flagged **learning-favorable**.

**One-line law:** a learned proposal beats classical search iff single-start reachability
decays with dimension (equivalently, iff the acceptance basins effectively factorize). Steep
slope ⇒ learning helps; flat slope ⇒ it cannot.

## Artifacts produced
- `scripts/run_crossover_theory.py` — measures q(n) with the *identical* generators/refine/checker
  as R5/R7 (600 trials, Wilson CIs), tests the law, predicts the random curve. `--no-geometry`
  to skip the slower real-domain family. Reproduce: `PYTHONPATH=. python3 scripts/run_crossover_theory.py --trials 600 --K 8 --seed 20260721`.
- `marc/data/geometry.py` — `build_point_chain_graph` / `make_point_chain`: scalable coupled
  geometry family (2k vars), integer solutions the checker accepts.
- `paper/main.tex` — full AAAI-2026 draft built on the law (needs the official `aaai2026.sty`).
- `paper/ABSTRACT.md`, `paper/refs.bib`, `paper/notes/crossover_law.md` (derivation + limits).
- `paper/RESULTS.md` R9; `paper/PROVENANCE.md` R17–R19; `paper/figures/fig_crossover_theory.pdf`.
- `results/p_crossover/crossover_theory.json` (tracked via .gitignore whitelist).
- Tests: `tests/test_crossover_theory.py` (law algebra, dichotomy, geometry). **Suite: 370 → passing.**

## Honest status / what's NOT claimed
- No claim a learned solver beats classical search in general — the coupled result is the
  opposite, reported as a primary finding. The contribution is the law that says *when*.
- Geometry: the law *predicts* learning can help there (steep collapse). Training the geometry
  denoiser and running it vs the random-restart control is the flagged next experiment (not done).
- Crossover point is not claimed sharply: the higher-N (600) re-measurement puts random weaker
  at low n than the N=40 R5 table, so the robust claim is the geometric collapse, not a specific n*.

## Next steps (priority)
1. **Train the geometry denoiser** on `make_point_chain` and eval vs random-restart + LM — the
   law's live prediction; a geometry *positive* would materially strengthen the paper.
2. Drop the official `aaai2026.sty`/`.bst` in `paper/`, compile `main.tex`, fill the two ⟨geo⟩
   slots from `crossover_theory.json`.
3. Optionally still wire `MARC_CKPT` into the eval scripts (HANDOFF §5) — but the law already
   explains why scale won't rescue the coupled negative (factorization is a problem property).
4. Push `sparsh/crossover-theory`, open a PR (author @ImSpxrsh, no Claude co-author per repo convention).
