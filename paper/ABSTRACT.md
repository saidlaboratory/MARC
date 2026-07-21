# Abstract & title

> Draft for AAAI main track. Numbers marked `⟨run⟩` are refreshed from
> `results/p_crossover/crossover_theory.json` (600-trial run) before submission.
> Framing is the honest one: a predictive law that explains both the positive
> (R5) and the negative (R7) under one mechanism. No claim is made that a learned
> solver beats classical search in general — the opposite is a central finding.

## Title (primary)
**When Do Learned Proposals Beat Classical Search in Continuous Constraint Solving? A Predictive Factorization Law**

### Alternates
- A Factorization Law for Amortized Inference in Constraint Solving
- Learning Helps Constraint Solving Exactly When Basins Factorize

---

## Abstract (primary, ~210 words)

Amortized neural inference — training a model to propose solutions that a cheap
classical routine then polishes — is an increasingly common recipe for
combinatorial and continuous optimization. Yet evaluations rarely isolate what the
*learned* proposal contributes, comparing hybrids against weak cold-start baselines
instead of the obvious control: random multi-start with the same polish budget. We
revisit amortized inference for continuous algebraic constraint solving with a
neuro-symbolic diffusion solver — problems compile to factor graphs, a graph
denoiser proposes assignments by reverse diffusion, and an exact computer-algebra
checker gates every acceptance — and add the controls the recipe usually skips:
random multi-start + polish, and a Levenberg–Marquardt solver with the analytic
Jacobian.

Under these controls we find that the learned proposal's advantage is neither
universal nor illusory but *governed by a measurable geometric property*: whether
the problem's acceptance basins factorize across variables. Best-of-$K$ random
restart succeeds with probability $1-(1-q(n))^K$, where $q(n)$ is the single-start
reachability; when constraints are variable-separable, $q(n)=v^n$ exactly, so the
restart budget grows as $v^{-n}$ and search collapses in high dimension while a
learned proposal that reproduces each marginal stays flat and must eventually win.
Measuring the single constant $v=0.27$ predicts the full random-restart curve with
no free parameters (MAE $0.012$) and the expected-restart budget's $v^{-n}$
explosion (from 4 to 600 restarts over dimensions 1–6). Breaking separability with a
coupled chained-bilinear family, the law predicts — and we confirm — that $q(n)$
stops decaying ($\log$-slope $-0.13$ vs $-1.03$ when separable, both $R^2>0.95$),
random search never collapses ($\approx4$ restarts at every dimension), the learned
proposal ties it at every dimension, and the classical solver dominates. Amortization thus pays off
precisely, and only, when basins factorize and dimension is high. We report the
negative regimes as primary findings with Wilson intervals and $z$-tests
throughout, alongside a controlled entrapment result (annealed noise cuts
deterministic trapping by $0.525\pm0.086$, $N=200$). The law converts scattered
"helps here, not there" observations into a falsifiable account of when amortized
inference earns its training cost — and a reproducible protocol for evaluating it.

---

## Positioning (for the intro, not the abstract)

- **Contribution type:** an *analysis / understanding* paper with a system as the
  instrument. The novelty is the predictive law + the controlled protocol, not a
  new SOTA solver. AAAI accepts this category; the bar is generality, rigor, and a
  falsifiable claim that holds — which the parameter-free prediction delivers.
- **Why it is not oversold:** the headline is a law that *bounds* where learning
  helps; the negative results (coupling kills the advantage; LM dominates) are the
  evidence, not hidden caveats. This is the framing the team's own controls forced
  (`AAAI_READINESS.md`, `HANDOFF.md`).
- **Related work hook:** DIFUSCO / Langevin-CO / amortized-inference papers report
  hybrid wins without the random-restart control; AlphaGeometry-style
  neural-proposal + symbolic-checker division of labor. We supply the missing
  control and the law that explains the results. (`paper/related_work.md`.)
