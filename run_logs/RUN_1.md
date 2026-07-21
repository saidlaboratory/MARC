# RUN 1 — full-scale overnight harness (2026-07-20/21)

Driver: `scripts/run_overnight.py` on MacBook M5 (MPS, torch 2.12). Outcome: 22 phases,
17 ok / 4 skipped / 1 failed. Canonical numbers: `paper/RESULTS.md` + `paper/PROVENANCE.md`;
raw outcome: `OVERNIGHT_RESULTS.md`.

## Summary of results

The run confirms the honest pattern from the small-scale experiments and sharpens it: the
learned model beats naive baselines (fixed / cold-start / deterministic) everywhere, but ties
or loses to *random* selection/restart everywhere except one regime — independent
high-dimensional traps, where random restart collapses under the curse of dimensionality and
the learned proposal holds.

| Experiment | Key numbers | Read |
|---|---|---|
| Convex (p1, main, CoT) | all 1.000, gap 0 | saturated, no signal |
| Hard bilinear | learned = random on 3/4 (0.55/0.68/0.68); CircleLine 0.00 | ties random |
| Coupled chained bilinear | learned ≤ random at every n (0.23–0.48 vs 0.37–0.60) | ties/loses, 0/5 wins |
| Dimension scaling (independent) | learned 0.95/0.95/0.975/0.925/0.25 vs random 1.0/0.725/0.075/0/0 | learned ≫ random for n≥3 |
| Structure-selection policy | > fixed and > no-context (p<1e-4); = random (p=0.28) | ties random selection |
| Geometry (refine) | 0.56 in-dist | non-saturated; best future lead |
| LLM (Gemini) + MARC | direct ≈0.80 vs formalize-then-solve ≈0.00 on MATH | formalization is the bottleneck |

Training: Stage-A D512/L8 converged cleanly (DSM loss 0.60, ~30 min at ~280 ex/s on MPS).
Stage-B GRPO diverged (loss 12.9 → 134,892; reward ~−9e8) and was cut. `eval_main_learned`
was killed after 3h+ stuck (D512 diffusion+guidance too slow on MPS over the full
perturbation suite; that eval is convex/saturated anyway).

One caveat that limits what this run can claim: the differentiating eval scripts
(`run_hard_eval`, `run_coupled_eval`, `run_dimension_scaling`) retrain their own small models
and never load the D512 checkpoint. So the scaled model was only ever tested on the saturated
convex evals. The negatives above are established for the small models; "scale doesn't change
them" is expected but not yet measured.

## What we have that's good

1. A working end-to-end neuro-symbolic system: factor graphs + GNN diffusion denoiser + exact
   SymPy residuals/energy/gradients + exact accept checker + classical Langevin fallback.
2. The learned solver converges (was diverging at 0%; five real bugs found and fixed —
   `paper/learned_solver_fix.md`). First real-scale trained checkpoint (D512/L8, loss 0.60).
3. A genuine, controlled positive: the dimension-scaling crossover. Random restart wins n≤2,
   learned wins n≥3 (0.925 vs 0.000 at n=4). Clean amortized-inference story.
4. The entrapment result (RQ2): deterministic descent 100% trapped → annealed Langevin 0.475;
   reduction 0.525 ± 0.086, 95% CI excludes 0, N=200, pre-registered.
5. Partial cross-family transfer: the hybrid solves 2/4 held-out families it never trained on
   at 0.683 (p<1e-4), plus the finding that a pathological training family disrupts transfer.
6. An architectural finding: variables must be conditioned directly on incident constraint
   constants (LayerNorm washes magnitude out); the constant→output skip cut mean|err| 5.4 → 0.9.
7. Statistical hygiene throughout: every solve rate has N + Wilson CI or z-test; the
   random-multistart control most papers skip; provenance for every number (R1–R14).
8. Rigorous negatives, properly controlled: the coupled family (R7), structure selection vs
   random (R8 protocol), LLM-verify vs LLM-direct. These are informative, not just failures.
9. Crash-safe overnight harness with manifest/skip/relaunch, 365-test green suite, MATH
   coverage reality check (0/48, ~20% constraint-shaped).
10. Geometry at 0.56 in-dist: a non-saturated, real-ish domain where signal is still possible.

## Every flaw

1. **No main-track positive.** All three angles are negative: learned proposal ties/loses to
   random restart on hard and coupled families; structure selection ties random; LLM+MARC
   formalization scores ~0 vs ~0.80 direct.
2. **The one positive is narrow and synthetic.** The n≥3 crossover needs per-variable-separable
   solutions *and* random restart collapsing; it vanishes under coupling. A reviewer reads it
   as "learned marginals on toy problems built for the model."
3. **The R7 diagnosis is structural:** the model effectively learns per-variable marginals, so
   any coupled (joint) solution space kills the advantage. This is an architecture limit, not a
   tuning problem.
4. **Harness gap: the differentiating evals never load the trained D512 checkpoint** — they
   retrain small models. The scaled run was largely wasted on saturated convex evals.
5. **Stage-B GRPO diverges at D512** (unbounded shaping reward — a raw energy delta that can
   reach ~1e16 within the ±1e4 rollout clamp — plus unnormalized inputs; reward ~−9e8),
   ~35 min/epoch. No usable RL stage. Gradient clipping is already in place (`grad_clip: 1.0`
   in `scale.yaml`, applied in both stages) and doesn't save it.
6. **CircleLine fails completely** (0.00 in-dist and cross-family), and cross-family transfer
   fails on 2/4; adding CircleLine to the training mix collapses BilinearSystem transfer
   (0.70 → 0.00).
7. **Learned degrades at n=6 even on independent traps** (0.25/0.10) and loses to random at n≤2.
8. **R8 structure-selection numbers are withdrawn** (test seeds == validation seeds, plus a
   train/eval data-source mismatch); clean seed-space-v1 numbers not yet regenerated. Even the
   valid rerun ties random (p=0.28).
9. **All problems are synthetic.** MATH coverage 0/48; no real domain result yet; geometry only
   has the classical-refine number (0.56), no learned-vs-random comparison there.
10. **Entrapment is confirmatory, not novel** — textbook annealed Langevin.
11. **MPS is too slow for D512 diffusion+guidance at eval scale** (the 3h+ hang); Stage-A loss
    is volatile (spikes to ~390 in epoch 1) despite the `grad_clip: 1.0` already in the config.
12. **No `.tex` exists** with 6 days to deadline; `paper/*.md` sprawl (~10 files) not yet
    consolidated.
13. Hygiene debt: OpenAI + Gemini keys exposed in a working chat (need rotation); a pending
    results commit may still need pull+push; Gemini free tier quota caps the CoT baseline
    (small N, weak model).

## Solutions

1. **(Flaw 4, cheapest, do first)** Wire `MARC_CKPT` into `run_hard_eval`, `run_coupled_eval`,
   `run_dimension_scaling` so they load the D512 checkpoint. Rerun those three with
   `--skip tests,train_stage_a,train_stage_b`. Either scale changes a conclusion (new result)
   or every claim upgrades from "small model ties random" to "scale doesn't rescue it."
2. **(Flaws 1–3, the only real path to a positive)** Attack the joint-distribution limit
   directly: condition the denoiser across the coupling structure (message passing that
   preserves constraint-constant magnitude along chains, or an autoregressive/joint proposal
   head) and re-test on the coupled family. Until the proposal is joint, R7 will reproduce.
3. **(Flaw 9)** Run learned-vs-random on geometry. It is the one non-saturated, non-separable,
   real-ish domain; a win there is worth more than all the synthetic tables combined. ~2–3 days.
4. **(Flaw 5)** Before any Stage-B retry: input normalization, reward scaling/clipping (the
   shaping reward is a raw, unbounded energy delta; a −9e8 reward guarantees divergence and
   gradient clipping — already applied — can't rescue it), lower LR. Or drop Stage-B for this
   paper — Stage-A is the checkpoint that matters.
5. **(Flaw 6)** Treat CircleLine as a diagnostic: characterize why (solution manifold geometry
   vs training distribution) and report it as the failure-mode analysis section. Curriculum or
   family-weighting for the transfer-collapse effect.
6. **(Flaw 8)** Regenerate R8 under seed-space v1 (disjoint ranges, `overlap_instances: 0` in
   the results JSON) via the fixed harness; cite nothing until then.
7. **(Flaw 7)** Report the n=6 dropoff and n≤2 loss openly and scope the claim to the crossover
   window. Optionally probe whether more restarts (best-of-K scaling curve) shifts the crossover.
8. **(Flaws 10, 12)** Write the paper to the corrected framing now: thesis = "learned proposals
   amortize *independent* search, not *joint* search — a controlled characterization of when
   learning helps constraint solving." Entrapment becomes supporting evidence, not a claim of
   novelty. Start the `.tex` today (intro/method/related-work are experiment-free;
   `related_work.md` is ready); consolidate `paper/*.md` into `paper/notes/`.
9. **(Flaw 11)** Cap eval-suite sizes for D512 on MPS and add a per-phase wall-clock budget to
   the harness (`scale.yaml` already carries `grad_clip: 1.0`).
10. **(Flaw 13)** Rotate both API keys; `git pull --no-rebase origin main` then push the pending
    results commit; fund a real key if the CoT baseline stays in the paper (N≥100, k≥4,
    stronger model) — otherwise cut it.

## AAAI main-track confidence

*Reviewed as by a harsh AAAI PC member before scoring:*

> The central claim reduces to: a diffusion model beats random multi-start only when solutions
> are per-variable separable and dimension makes random search infeasible — and the authors'
> own controls show the advantage disappears the moment variables couple. That is a negative
> result about the method they built. The one positive (R5) is on synthetic trap families
> constructed to exhibit exactly the property the model exploits; no real domain, 0/48 on MATH,
> and the entrapment result is standard annealed Langevin. The structure-"selection" component
> ties random choice, and the first version of that experiment was seed-contaminated. Transfer
> fails on 2/4 families including a total collapse. The statistical rigor and the honesty of the
> controls are genuinely above average — I would say this is the best-executed negative result
> I've reviewed this cycle — but AAAI main track wants a contribution, not a well-documented
> boundary. As a "when does learning help" characterization study it lacks the breadth (one
> problem class, one architecture) that would make the negative itself the contribution. Weak
> reject; would be a strong workshop paper.

**Main-track acceptance confidence: 8/100.**

The rigor floor is real (controls, CIs, provenance) and prevents a desk-reject-quality score,
but with no positive on coupled/realistic problems, no real domain, and 6 days to deadline with
no `.tex`, acceptance would require a lenient committee, not a good draft. Contingencies: if
the D512-checkpoint rerun flips a coupled result, ~20/100; if geometry produces a controlled
learned-vs-random win, ~30–35/100. Workshop acceptance with the honest framing: ~85/100 —
that is the venue this project is currently built for.
