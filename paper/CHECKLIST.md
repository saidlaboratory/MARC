# AAAI-26 Reproducibility Checklist — draft answers

Filled against the actual repo so submission day is copy-paste. Update the two figures
marked TODO once the final compute settles (#122 freeze).

## This paper

- **Includes a conceptual outline and/or pseudocode description of AI methods introduced** — Yes.
  Method section (`\section{Method}`); the propose–polish–check loop and the repair ranker are
  described in full, with the energy, guidance, and acceptance gate stated as equations.
- **Clearly delineates statements that are opinions, hypotheses, and speculation from
  objective facts and results** — Yes. Hypotheses are pre-registered (entrapment, entrapment
  trap); the Limitations section is explicit about what is measured vs believed.
- **Provides well-marked pedagogical references for less-familiar readers** — Yes (Related Work:
  multistart, amortized optimization, learning-to-branch, distance geometry, algorithm selection).

## Theoretical contributions

- **All assumptions and restrictions are stated clearly and formally** — Yes. The factorization
  law states its assumption (shared polish + checker; instance-averaged q) and its Jensen-gap
  caveat explicitly.
- **All novel claims are stated formally** — Yes. Eq. (best-of-K) is the formal claim; validated
  against a self-measured curve (MAE 0.012), not assumed.
- **Proofs of all novel claims are included** — N/A / partial. The law is an identity plus a
  measured validation, not a theorem with a proof obligation; "exactly one solvable option" is a
  CAS certificate (real-root nonexistence), machine-checked per instance rather than proved on
  paper.
- **Proof sketches or intuitions are given for complex results** — Yes.

## Datasets

- **All datasets are procedurally generated** — Yes. No external dataset dependency; every family
  is generated from a seed (`marc/data/`, `marc/structure/`). MATH-500 is referenced only as a
  scope measurement (48-problem coverage count), not trained or tested on.
- **All novel datasets are described and will be released** — Yes. Generators ship in the code
  artifact; the citable result JSONs ship under `results/`.

## Computational experiments

- **Code released** — Yes (this repository; MIT LICENSE).
- **All source code required for the reported experiments is included** — Yes. Every PROVENANCE
  row names the exact command; `results/*.json` are committed.
- **Dependencies specified** — Yes. `requirements.txt` (loose) and `requirements-lock.txt`
  (pinned: torch 2.9.1, torch-geometric 2.8.0, scipy 1.15.2, sympy 1.13.3, numpy 2.4.6,
  matplotlib 3.10.3, pytest 9.1.0, PyYAML 6.0.2). Python 3.10+.
- **Reported results can be reproduced** — Yes. `scripts/repro_all.sh verify` recomputes every
  cited table from the committed JSONs and diffs against the paper; `rerun` documents the full
  compute path. CI runs the fast test subset plus verify on every push.
- **Number of algorithm runs / seeds** — Reported per result. Headline nonlinear repair and
  dimension-scaling law use 3 optimization seeds (11/29/47) with independent per-seed eval draws;
  entrapment uses 5 seeds; R28 geometry uses 3 seeds; R30 real-systems repair is measured on
  N=200 instances per class under two-stream failure selection.
- **Central tendency and variation reported** — Yes. Every rate carries N and a 95% Wilson
  interval or a two-proportion z-test; multi-seed results carry mean ± population SD.
- **Statistical significance** — Yes. Exact paired McNemar on common-stream outcomes for
  paired arm comparisons; Holm step-down within declared comparison families.
- **Compute resources described** — Yes (see below). No cluster or accelerator required.
- **Assets have licenses / are cited** — Yes. MATH-500 cited (\citep{hendrycks2021math});
  AlphaGeometry and the multistart/learning-to-branch/distance-geometry references cited.

## Compute used

Every experiment in the paper runs on a single machine (Apple-silicon MacBook, CPU; PyTorch
MPS optional). No GPU cluster. Representative wall times (one machine, `--solve-e2e` boundary):

- Repair-ranker training (nonlinear, 3 seeds): TODO minutes/seed — fill from #128 wall-clock table.
- R28 geometry v3 (per seed, warm dataset cache): ~7.5 h/seed at protocol scale (n=1250/400/600,
  60 epochs); dataset build parallelized over the process pool.
- R30 real-systems repair (N=200 × 4 classes): ~1.5 min total.
- Dimension-scaling law (3 seeds × 5 dimensions): TODO — fill from `scaling_3seed.json`.
- Fast test subset (CI): ~2 minutes.
