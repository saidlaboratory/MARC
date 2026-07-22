# Review attacks — discussion-phase arsenal (living doc)

The paper is borderline; it wins or loses in reviewer discussion. Each section below is
one anticipated attack: the attack in the reviewer's own words, a two-sentence defusal
you can paste into a response, and the evidence pointer (RESULTS.md row + file). Sourced
from the five-persona review board pass (2026-07-22) plus the earlier hostile-review
pass; pre-v8 attacks that the v8 protocol killed are deleted.

House rule: claim language stays "menu-based structure selection / structural repair
(with predicted defining value)" — never "invention," never "no classical baseline."

## Numbers discipline for rebuttals — read first

- **Never cite 0.565, 0.445, or 0.889** (v6/v7 repair, withdrawn). The trajectory
  0.565→0.445→0.380 may be cited *only* as shortcut-closing evidence.
- **Never quote the old dimension-scaling rows** (0.550-vs-0.050 at n=3, "random wins
  at n≤2"). Cite unified-v2: learned 0.95/0.95/0.975/0.925/0.25 vs random
  1.0/0.725/0.075/0/0 at n=1/2/3/4/6 (`results/p_scaling/scaling.json`, PROVENANCE R15).
- The withdrawn v0.2 numbers (0.45/0.53) stay withdrawn; R8 is motivation only.

## The attacks

### #1 · "You built all your own benchmarks"
> "Every family is a synthetic construction by the authors; the characterization may be
> an artifact of families designed to produce it."

**Defusal:** We ran the characterization on eight recognized external systems —
circle/conic intersection, GPS trilateration, Rosenbrock and Himmelblau stationary
points, 2R/3R inverse kinematics, cyclic-4 — and it held: LM solves 8/8, no
learning-favorable regime appears, exactly as the law predicts for low-dimensional
coupled systems. The synthetic families are controls, not benchmarks: they isolate the
two conditions (reachability collapse, separability) that the real systems then test.
**Evidence:** R26 · `results/p_real/real_systems.json` · `paper/notes/real_systems.md` ·
marc.tex `sec:real`.

### #2 · "0.997 means the task is too easy"
> "A ranker at 0.997 on a 4-way menu is a saturated benchmark; the task is trivial."

**Defusal:** The same menus hold the candidate-only control at 0.333, random at 0.236,
and the strongest cheap probe (300-step LM on every candidate) at 0.881 — the task is
not solvable by recipe, chance, or probing, only by reading the candidate-augmented
graph. 0.997 is high because v8 menus carry theorem-grade "exactly one solvable option"
semantics (exact CAS no-real-roots certificates, 356/360), which is the claim: where
operator identity decides solvability, the learned ranker reads it and the controls
cannot. **Evidence:** R10 · `results/p_repair/nonlinear_balanced_full.json`,
`results/p_repair/probe_nonlinear.json` · marc.tex `sec:repair`.

### #3 · "This is just algorithm selection"
> "The repair ranker is per-instance algorithm selection with a verifier — Rice 1976,
> SATzilla, learn-to-branch. Picking from a menu is K-way classification."

**Defusal:** Conceded and cited — the paper frames it as verifier-gated selection in
that lineage, and our own claim language is "menu-based structure selection," not
invention. What that literature does not run is what we claim as new: certificate-grade
menu semantics (distractor unsolvability proven at any budget/seed), candidate-conditioned
encoding of each augmented graph, and the budget-matched restart and probe controls.
**Evidence:** R10 protocol bullets · marc.tex `sec:related` (post-repositioning, board
novelty fix 1) · `paper/notes/repair_ranker.md`.

### #4 · "The law is classical basin counting"
> "q(n)=v^n and best-of-K = 1−(1−q)^K is textbook multistart theory (Rinnooy Kan &
> Timmer); calling it a law overclaims."

**Defusal:** Conceded — the components are textbook and cited as such; the contribution
is the measured-slope diagnostic (one measured constant reproduces the entire best-of-8
restart curve, MAE 0.012, no free parameters) plus the two-condition dissection, which
the geometry family falsified and sharpened live (collapse alone is not enough;
separability is required). R29 then makes the second condition causal and
architecture-free: even the *true* product-of-marginals proposal ties random under
coupling. **Evidence:** R9, R29 · `results/p_crossover/crossover_theory.json`,
`results/p_scaling/oracle_marginal.json` · `paper/notes/crossover_law.md`.

### #5 · "Why should AAAI care about a negative result"
> "A carefully-run negative about diffusion value proposals is a workshop paper."

**Defusal:** It is not a bare negative: the paper delivers a falsifiable predictive law
(parameter-free curve reproduction, MAE 0.012, validated on eight real systems), a
decisive positive exactly where the law says learning can win (R10, p<1e-70 against
matched controls), and a mechanism for why learned-proposal wins in this literature can
vanish under a matched multistart control — an actionable indictment of a common
evaluation practice. The negative is what makes the positive credible; all five review
personas independently named the control discipline the paper's main asset.
**Evidence:** R7, R9, R10, R26 · board `novelty.strengths` 1–3 · marc.tex abstract arc.

### #6 · "Small models — would a bigger denoiser cross the coupling ceiling?"
> "Your denoisers are tiny; perhaps a larger model would learn the joint structure and
> the coupled null would disappear."

**Defusal:** R29 closes this without training anything: sampling each coordinate from
the family's *true* per-variable marginal — the ceiling for any marginal-amortizing
learner at any capacity — still ties random restart at every n≥3 (0/4 significant, same
polish, CRN seeds). The coupled null is a ceiling of the proposal class, not a capacity
artifact; a bigger denoiser could only approach the oracle-marginal arm, which already
fails. **Evidence:** R29 · `results/p_scaling/oracle_marginal.json` ·
`scripts/run_oracle_marginal.py`.

### #7 · "Saving 1.6 solver calls is not a contribution"
> "Enumeration at K=4 is already perfect at ~2.5 calls; your ranker saves 1.6 solver
> calls. So what?"

**Defusal:** On nonlinear menus the ranker beats probing on *accuracy*, not just cost —
0.939 at one call vs 0.881 at 4.73 calls — because certified-rootless distractors are
unsolvable at any budget while short-budget probes miss golds, an error mode
graph-reading avoids. Where enumeration is genuinely cheap and perfect (linear K=4) the
paper says so plainly and labels those rows mechanism-not-deployment; the cost argument
is made only where it holds (4.65× wall-clock at K=16, labeled a cost win only).
**Evidence:** R10 probe + K-scaling paragraphs · `results/p_repair/probe_nonlinear.json`,
`results/p_repair/linear_K16_e2e.json` · marc.tex `sec:repair`.

### #8 · "K=16 accuracy collapse"
> "Your accuracy advantage shrinks with K and is gone at K=16, and direct K=16 training
> sits at chance — the method does not scale in menu size."

**Defusal:** Reported by us as a limitation, with the honest split: at K=16 the
zero-shot checkpoint keeps only the cost win (policy+solve stays 3.8–4.9 ms while
enumeration grows to 22.5 ms), and the direct-training-at-chance negative stays in the
record. Large-K accurate ranking is stated as open; R28's growing construction
vocabulary is the designed next test. **Evidence:** R10 K-scaling ·
`results/p_repair/linear_K16.json`, `linear_K16_trained.json`, `linear_K16_e2e.json`.

### #9 · "The linear result is unstable across seeds"
> "Linear full 0.317 ± 0.069 with one seed at random level — the linear edge is noise."

**Defusal:** Stated as a limitation in the paper by us, alongside the pooled paired
tests (full>control p=3.5e-13, full>random p=7.8e-07 over N=1,200): the linear rows are
a controlled mechanism result, not a deployment claim, and the seed instability is
printed next to them. The deployable claim rests entirely on nonlinear, which is
optimization-robust: 0.982 ± 0.006 across seeds 11/29/47. **Evidence:** R10 multiseed
paragraph · `results/p_repair/linear_multiseed.json`, `nonlinear_multiseed.json` ·
marc.tex `sec:repair` + Limitations.

### #10 · "Single-seed value-proposal cells"
> "The dimension-scaling and crossover cells are one training seed at N=40 per cell;
> why should I trust them?"

**Defusal:** The decision-critical results are multi-seed or learner-free: the ranker
has optimization-seed repeats (R10), the R29 oracle arm has no learner at all and its
recomputed random column reproduces R7 digit-for-digit, and R27 replicates the crossover
on three structurally different families with effect sizes (1.000 vs 0.000 at n=6) that
no seed noise produces. Ns and single-seed status are disclosed per caption; remaining
compute went to decision-critical cells, not blanket repeats. **Evidence:** R27, R29 ·
`results/p_scaling/scaling.json` (`seeds: 1` disclosed), `crossover_families.json`,
`oracle_marginal.json`.

### #11 · "Same structures at train and test — holdout only varies constants"
> "Your 'generalization' is the training patterns with fresh constants."

**Defusal:** The transfer cells are held-out *patterns* and held-out *relations*,
rotated: linear held-out-pattern rotation gives offset 0.407 / coupled 0.450 / shared
0.380 vs ~0.23–0.29 random (N=400 per cell), and nonlinear relation transfer is
bidirectional (vieta→quad_link 0.420, quad_link→vieta 0.393, vs ~0.2 random). Transfer
is partial and reported as such; in-distribution and transfer numbers are never
conflated. **Evidence:** R10 transfer breadth ·
`results/p_repair/random_support_holdout_{offset,coupled,shared}.json`,
`nonlinear_holdout_vieta.json`, `nonlinear_holdout_quad.json`.

### #12 · "Your candidate-only control is above chance — a leak by your own standard"
> "Candidate-only scores 0.333 on balanced nonlinear, CI excluding the 0.25 floor —
> the same magnitude you used to withdraw v6."

**Defusal:** The residual is recipe-intrinsic, not a label leak: with one-sided
templates (u = a·x²+δ), some (a, δ) sign combinations are rootless regardless of the
problem, so the recipe alone carries partial information — the claim is worded as "the
problem graph is required to exceed the recipe ceiling," and the graph-reading effect is
the 0.997-vs-0.333 gap. The v6 withdrawal was different in kind: gold pins were drawn
from a narrower prior than distractor pins, a generation artifact that v8's shared
support/prior eliminates. **Evidence:** R10 v8 protocol bullets ·
`results/p_repair/nonlinear_balanced_full.json` · `results/p_repair/README.md`.

### #13 · "Your numbers moved three times — why trust v8?"
> "The linear headline went 0.565 → 0.445 → 0.380 across data versions; what breaks
> next?"

**Defusal:** Each drop is a named, closed shortcut (pin-prior leak, probe-artifact
certification), disclosed in the paper as an audit trail — the movement is evidence the
controls work, and it moved *against* us on linear while the theorem-grade nonlinear
number moved up for a stated reason. v8 hygiene is computed, not asserted: seed overlap
0 in every citable JSON, exact CAS certificates, one test-enforced reference solver
across certification/grading/training/e2e, and checkpoint-only replays reproduce both
headline evals exactly. **Evidence:** R10 "honest movement" paragraph ·
`results/p_repair/README.md`, `nonlinear_balanced_full_paired.json` · PROVENANCE
withdrawn rows.

### #14 · "The 'operator-aware encoding' mechanism claim is unsupported"
> "Your v0.2→v0.3 comparison confounds architecture, data version, and features; you
> never ablated the operator features."

**Defusal:** We ran the mask ablation and corrected the attribution in the paper:
masking the operator-identity features and retraining leaves the ranker essentially
intact (nonlinear 0.981 vs 0.997, linear 0.379 vs 0.339), so the v0.2→v0.3 gain is
attributed to candidate conditioning — encoding each candidate-augmented graph — with
the compatibility signal carried jointly by constants, coefficients, and incidence. The
earlier operator-feature attribution is explicitly retracted in the text. **Evidence:**
R10 opmask paragraph · `results/p_repair/nonlinear_opmask_ablation.json`,
`linear_opmask_ablation.json`.

### #15 · "Best-of-8 is candidate parity, not wall-clock parity"
> "One guided diffusion rollout costs far more than one uniform draw plus polish; random
> restart at matched wall-clock might erase the n=3–4 advantage."

**Defusal:** The claim is scoped to candidate-budget amortization and stated as such,
with wall-clock reported where deployment is claimed (repair: ms-level per arm). The law
quantifies the exposure and closes it in dimension: expected restarts grow geometrically
(3.7 → 600 by n=6), so any fixed wall-clock budget is exhausted while the learned
proposal's cost is flat — the regime where the claim matters is exactly where wall-clock
cannot rescue random search. **Evidence:** R9 expected-restart rows ·
`results/p_crossover/crossover_theory.json` · board meta `do_not_do` 4.

### #16 · "Why no 3D / molecular DDGP?"
> "The geometry domain is 2D toy chains with exact rational coordinates; real distance
> geometry is 3D molecular DMDGP."

**Defusal:** Scope, disclosed: R28's 2D/exact-rational/integer-coordinate
simplifications are stated verbatim in the paper, its construction vocabulary is
positioned against DMDGP branch-and-prune symmetry theory (Lavor/Liberti/Mucherino),
and 3D/molecular is named as open in one sentence. The claim is that the relocation
thesis reaches a real geometric domain at all — the same pruned chains where a trained
value denoiser exactly tied random restart (R9 geometry arm) — not that we solve
molecular DDGP. **Evidence:** `marc/structure/geo_repair.py` docstring (disclosed
simplifications) · R9 geometry bullet · `results/p_geometry/pointchain_learned.json`.

## If R28 comes back mixed — honest framing (AC do-not-spin rule)

Do not hold the submission for a better R28 outcome, and do not spin the one that
lands. The v8 menu result stands on its own either way.

- **Ranker ties best_fixed / all_cos:** the finding is "constructions repair; a fixed
  recipe suffices at this vocabulary size." Print the fixed recipe as the honest
  ceiling; if the ranker adds anything, reframe as instance-conditional selection
  *beyond* the fixed recipe — otherwise move R28 to the appendix as a real-domain
  construction result with selection open.
- **restart_plus16/32 closes the gap:** the result is "buy more restarts," not "learn
  the construction." Report the scaling curve and say so.
- **Language:** R28 labels are budget-relative measured labels under two restart
  streams, NOT CAS certificates — never describe R28 menus with R10's certificate
  language.
- **Discussion answer if asked:** a mixed R28 bounds the transfer of the relocation
  thesis to that domain; it does not touch the certificate-grade menu-repair claim
  (R10), which is the paper's positive.

## Conceded strengths — protect these

Every persona conceded these; do not trade them away while defending:

1. **Honesty architecture** — the matched-budget random-multistart control that shrank
   our own headline, the disclosed 0.565→0.445→0.380 audit trail, withdrawn numbers
   flagged in three places, negatives (CircleLine 0.000, coupled 0/5, double_well,
   K=16, linear seed instability) all in the text with Ns and CIs.
2. **The falsification loop** — the geometry family was a live test that refuted the
   one-condition law, and the paper reports the sharpened two-condition law with the
   refuting data; R29 then made the mechanism causal.
3. **The v8 protocol** — CAS certificates, one test-enforced reference solver,
   computed seed hygiene, checkpoint-only paired replays, exact McNemar.
4. **Reproducible engineering** — per-row provenance (command, seed, commit), committed
   evidence JSONs that match the paper's numbers exactly, 414 green tests.
