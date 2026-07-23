# AAAI 2026 Reproducibility Checklist — Draft Answers

Draft answers against the actual repo state, for issue #131 item 2. Source of truth for every
number is `paper/PROVENANCE.md` (command + seed + commit per result) and `paper/RESULTS.md`.
Fill in the official AAAI checklist form with these once the submission portal is open; update
this file if any answer changes before the deadline.

## Claims

- **Does this paper make theoretical contributions?** Yes, in a limited sense: Section 4.4's
  factorization law (`P_random(n;K) = 1-(1-q(n))^K`, `paper/tex/marc_aaai.tex` `sec:law`) is a
  derived, falsifiable prediction, not a formal theorem with proof obligations. All other claims
  are empirical.
  - All assumptions are stated: the law assumes a fixed polish operator and acceptance checker
    shared across all arms (stated explicitly in `sec:law`); it does not assume the acceptance
    basins factorize — that is the thing being tested (independent vs. coupled families).
  - Proof sketches / informal arguments are given in prose (the binomial best-of-K derivation is
    exact and one line; the geometric-decay argument for separable families is informal but
    stated); there is no formal proof requiring a complete-proof appendix.

## Experiments

- **Does this paper include computational experiments?** Yes — every number in the paper.
- **Any code required for the experiments included in the supplemental material or a URL?**
  Yes, once issue #129's anonymized supplementary zip lands (not yet packaged as of this
  writing). The live (non-anonymous) repo is `saidlaboratory/MARC` on GitHub; every experiment
  in the paper has a runnable script under `scripts/` (see the "Command" column of
  `paper/PROVENANCE.md` for the exact invocation per result).
- **All train/val/test splits specified?** Yes, per experiment, via disjoint seed ranges rather
  than fixed files (procedurally generated data). Example conventions actually used in the repo:
  `marc/structure/invention_data.build_split` uses a `seed + 100000 * source_index` stride per
  data source; `scripts/run_repair_ranker.py`'s Data Version 8 protocol documents train/val/test
  seed bands directly in its `--help` and in `results/p_repair/README.md`. `seed_hygiene()` in
  `run_repair_ranker.py` computes and reports the actual overlap between splits (checked to be 0
  and recorded per result in PROVENANCE), rather than assuming disjointness by convention.
- **Evaluation metrics specified?** Yes: solve rate / invention accuracy (fraction of held-out
  instances the checker accepts), with 95% Wilson intervals (`marc/eval/metrics.wilson_interval`)
  and one-sided two-proportion z-tests or exact paired McNemar tests for comparisons (both
  implemented in-repo, not hand-computed; see `scripts/run_repair_ranker.py`'s
  `_paired_mcnemar`, exercised by `tests/test_repair_ranker.py`).
- **Are statistical significance tests / confidence intervals reported?** Yes, for essentially
  every headline number in `paper/tex/marc_aaai.tex` and `paper/RESULTS.md` — this is treated as
  a house rule (`README.md`: "every rate carries N and a Wilson CI; paired comparisons use exact
  McNemar tests"), not an afterthought.
- **Description of computing infrastructure?** Single MacBook, Apple Silicon (MPS backend), no
  GPU cluster or distributed training (`HANDOFF.md`: "MacBook, torch 2.12, MPS"). No experiment
  in the paper required more than this; wall-clock costs quoted in Section 4.7 / Appendix D are
  measured on this same machine. Issue #128 (End-to-end wall-clock table) will add a single
  formal per-arm timing table at a matched boundary; until it lands, the wall-clock numbers
  already in the paper (Table in Appendix D, `results/p_repair/*_e2e.json`) are the best
  available and are per-arm, same-machine, same-boundary as of R24/#104's fix.
- **Number of parameters in each model?** Not currently reported anywhere in the paper or repo
  docs. `GraphRepairRanker(D=96, L=3)` and its variants are small bipartite message-passing GNNs
  (see `marc/model/repair_ranker.py`); an exact parameter count is a cheap addition
  (`sum(p.numel() for p in model.parameters())`) but is not yet in `paper/RESULTS.md` or
  `paper/PROVENANCE.md` — flagging as a gap rather than guessing a number.
- **Corresponding validation performance for each reported test result?** Partial. Train/val/test
  are procedurally disjoint by seed (see above) and validation is used for checkpoint selection
  (e.g. `scripts/run_repair_ranker.py --n-val ...`), but validation-set numbers themselves are not
  systematically tabulated alongside test numbers in `paper/RESULTS.md` — only the test-time rate
  is reported per result. Also a gap to flag, not a blocker (the split hygiene is real; the
  reporting of the intermediate validation number is what's missing).
- **Number of training / evaluation runs?** Documented per result via seed lists in
  `paper/PROVENANCE.md` (e.g. R22's three-seed robustness run, R25's 3-seed geometry
  confirmation, R27's per-family reruns). Single-seed results are explicitly labeled as such in
  RESULTS.md rather than implied to be multi-seed.
- **Hyperparameter configurations specified?** Yes, per experiment, via the exact CLI invocation
  recorded in `paper/PROVENANCE.md`'s Command column (D, L, epochs, batch size, lr are all
  explicit flags, not defaults buried in code).
- **Bounds for each hyperparameter?** Not applicable — no hyperparameter search / sweep is
  reported as a paper claim; configurations were chosen once per experiment family and reused
  (visible directly in the repeated D/L/lr values across PROVENANCE rows for the same family).

## Reproducibility tooling (repo-specific, not a standard checklist item but directly relevant)

- **One-command verify/rerun tooling:** not yet present. Issue #127 (One-command reproduction +
  CI smoke) is filed and open, assigned to Gun-Akash — this checklist should be re-run against
  its output once merged, since it will add exactly the `verify`/`rerun` modes this section
  currently has to describe qualitatively.
- **Checkpoint archive with hashes:** not yet present. Issue #128 covers this; PROVENANCE commits
  and script commands exist today (so numbers are regenerable from source), but a
  SHA-256-manifested archive of the citable checkpoints themselves is not yet committed.
- **requirements-lock.txt:** present and pinned (`matplotlib==3.10.3`, `numpy==2.4.6`,
  `torch==2.9.1`, `torch-geometric==2.8.0`, `sympy==1.13.3`, `scipy==1.15.2`, `pytest==9.1.0`,
  `PyYAML==6.0.2`, plus `openai==2.41.1` for the CoT baseline only).
- **Test suite:** 57 test files under `tests/`, exercising the CAS checker, solvers, data
  generators, and the repair ranker's own statistical-test helpers.

## Known gaps to close before submission (cross-referenced to open issues)

1. No LICENSE file yet (`README.md` badge reads "license-TBD") — issue #131 item 1, needs a lab
   decision before a checklist answer of "code is released under license X" is honest.
2. No parameter-count or validation-vs-test table yet (see above) — not tracked by an existing
   issue; worth a follow-up if reviewers press on it, but not blocking.
3. Issues #127 (repro script + CI) and #128 (wall-clock table + checkpoint hashes) are open and
   directly upgrade two of the answers above from "qualitative, spread across PROVENANCE.md" to
   "one command, one table."
