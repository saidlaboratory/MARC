# Review attacks — ranked defusal checklist (living doc)

A hostile-review pass produced this ranked list of the ways a reviewer kills the paper.
Each attack gets: the attack as the reviewer would write it, the defusal, who owns it,
and a status box. Update the boxes as units land; nothing ships while a P0 box is open.

House rule reminder: "invention" in code identifiers stays, but **claim language in
docs and the paper is "menu-based structure selection (with predicted defining
value)"** — the model picks from a K-candidate menu and predicts the defining value;
it does not synthesize structure from an open vocabulary. Attack #2 and #9 die the
moment we stop overclaiming.

## The attacks

### #1 · "Tasks are exactly solvable — there is no amortization story"
Every family has a closed-form/enumerable solution; a reviewer asks why anyone would
learn a sampler for a menu a for-loop can enumerate.
**Defusal:** enumeration arm with wall-clock timing (show what exhaustive menu+solve
costs as K grows) + nonlinear families where per-candidate solving is expensive.
Owners: W1/W2 (in flight).
- [ ] enumeration arm + timing in the eval
- [ ] nonlinear families landed

### #2 · "This is classification, not invention — and single-shot beats diffusion"
The preliminary numbers had single-shot 0.53 vs reverse-diffusion 0.45: the ablation
beats the system, and picking from a menu is K-way classification.
**Defusal:** (a) predicted-pin generation arm — the model must produce the defining
value, not just the slot (W3/W1); (b) honest renaming everywhere to "menu-based
structure selection" (this doc + W6). Do not argue with the reviewer; concede the
menu framing and sell what it actually is.
- [x] naming corrected across owned docs (W6, this PR)
- [ ] predicted-pin generation arm

### #3 · "val==test contamination + train/eval data-source mismatch"
The 0.45/0.53 runs evaluated on the same seed range used for validation checkpoint
selection, AND the harness trained the policy on `aux_required` while eval'ing
`toys`. Both are fatal to the numbers.
**Defusal:** W1 seed-space v1 protocol (disjoint train/val/test seed ranges, a
`seed_hygiene` block in every results JSON with `overlap_instances: 0`) + the W6
harness fix (eval `--data` now tracks what training actually used).
**DO NOT CITE 0.45/0.53 anywhere.** They are withdrawn (paper/RESULTS.md R8).
- [x] harness data-source match + `--eval-seeds` plumbing (W6, this PR)
- [ ] seed-space v1 protocol in run_invention_eval (W1)
- [ ] clean regenerated numbers

### #4 · "Holdout only varies constants — same structures at train and test"
If test instances are the training patterns with fresh constants, "generalization"
is memorizing three templates.
**Defusal:** `--exclude-family shared` is now the default training protocol; the
harness evals the excluded pattern separately (`eval_invention_heldout`) — that is
the cross-pattern generalization number. Owners: W6 (harness default, this PR) + W7.
- [x] held-out-pattern protocol default in the overnight harness
- [ ] held-out number generated and reported

### #5 · "Feature leakage / support shortcut"
The gold candidate may be identifiable from shallow features (menu position, support
size, coefficient signature) without reading the graph.
**Defusal:** no-context ablation (mask the graph, keep the menu — if accuracy holds,
the menu leaks) W3/W1 + support randomization W2.
- [ ] no-context ablation
- [ ] support randomization

### #6 · "Baseline poverty"
Random-slot and always-none are the only non-oracle baselines; a reviewer asks for
anything that tries.
**Defusal:** exact-solver baseline (try every candidate, keep the first that
verifies) + LM baseline. Owner: W5.
- [ ] exact enumeration solver baseline
- [ ] LM baseline

### #7 · "The value head is inert"
If the chosen candidate's predicted defining value never affects solve outcomes, the
"with predicted defining value" clause is decorative.
**Defusal:** policy_value arm — solve with the predicted value pinned vs. the gold
value vs. no pin. Owners: W1 + W3.
- [ ] policy_value arm

### #8 · "Stats hygiene"
Single seed, no multiple-comparison correction, cherry-picked comparisons.
**Defusal:** multi-seed eval (`--eval-seeds 5` is now default in the harness) +
Holm correction (W1); keep reporting the positive control (gold_oracle ≥ 0.95)
honestly — a failed positive control voids the table, and we say so in the output.
- [x] multi-seed flag plumbed through the harness (W6, this PR)
- [ ] Holm-corrected comparisons (W1)

### #9 · "Provenance / naming"
Numbers without commands+seeds+commits, and doc language ("invents") that the code's
own docstrings contradict ("menu-based").
**Defusal:** this unit. PROVENANCE.md rows for the invention eval (command template,
pending clean run); RESULTS.md R8 withdraws the contaminated numbers; naming honesty
sweep across RUNBOOK / AAAI_READINESS / FIXING_PLAN.
- [x] done in this PR (W6)

## Conceded strengths — protect these

The hostile pass conceded three things. Don't lose them while fixing the above:

1. **Honesty culture** — PROVENANCE.md, reported negatives (R7 coupled null,
   CircleLine 0.000, cross-family 2/4), positive controls that can fail loudly.
   Reviewers reward this; keep every negative in the paper.
2. **The entrapment result** — pre-registered RQ, CIs exclude zero, N=200. The one
   number nobody attacked.
3. **Reproducible engineering** — one-command overnight harness, crash-safe
   manifest, per-phase logs, seeds in JSON. Mention it in the reproducibility
   statement; it is a genuine differentiator.
