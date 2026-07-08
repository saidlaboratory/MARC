# CoT baseline

Token-level chain-of-thought baseline for H1 comparison against MARC's value-diffusion solver.

## Model

`gpt-5` via OpenAI API. Fixed internal temperature — this model does not support
temperature overrides (raises `BadRequestError` if passed). `max_completion_tokens=4096`;
reasoning tokens count against this budget, and problems that exhaust it before producing
a visible answer are scored as unparseable (see Known limitations).

## Prompt

Equations are rendered from `problem.graph.factors` (not the `description` string) so
the model sees exactly what the checker verifies. Answers are constrained to decimal
format (`ANSWER: x=<decimal>, y=<decimal>`) to keep parsing unambiguous — fractions
(e.g. `7/2`) are explicitly disallowed in the instruction.

## Run config

- N = 25 problems per split (in_distribution, held_out_structure)
- perturb_delta = 0.1
- n_samples (k) = 1 (pass@1)

## Running

    python -m marc.eval.baselines.cot_baseline

Output: results/p2_main/cot_baseline.json

## Result summary (N=25)

- solve_rate: 1.0 on both splits (generalization_gap = 0.0)
- perturbation_robustness: 0.0 (in-distribution) vs 0.12 (held-out structure)

Raw solve rate saturates at this problem size/difficulty, so it does not separate
the two splits — but perturbation_robustness does show a gap, suggesting this suite
may still be too easy for solve_rate itself to be an informative H1 signal.
Flagged to Sparsh/Davin: consider a harder suite (e.g. length_extrapolation with
higher n_vars) if solve_rate saturation persists once MARC's model is compared.

## Known limitations

- gpt-5's reasoning-token usage varies per problem; if it exhausts max_completion_tokens
  on internal reasoning before emitting an ANSWER line, parse_answer returns None and
  the Checker correctly scores it as a failure. This is a legitimate baseline failure
  mode (a token-budget design choice), not a parsing bug.
- Output JSON can contain NaN/Infinity for unparseable answers — not valid strict JSON;
  confirm downstream consumers (e.g. summary table merge) handle this or coerce to null.
