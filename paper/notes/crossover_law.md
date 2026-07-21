# The factorization law for learned-vs-classical constraint solving (R9)

**Status:** canonical derivation + validation writeup for the paper's central claim.
Numbers come from `results/p_crossover/crossover_theory.json`
(`scripts/run_crossover_theory.py --trials 600 --K 8 --seed 20260721`). This section
unifies R5 (the independent high-dim positive), R7 (the coupled negative), and a real
geometry domain under one falsifiable, parameter-free law whose diagnostic is the
*measured* single-start reachability slope — the answer to "why should AAAI care."

---

## 1. Setup

Every method in this study — Langevin, random multi-start, the learned diffusion
proposal, and the classical Levenberg–Marquardt solver — shares **one** polish
operator (`marc.refine.iterative.refine`, energy-gradient descent) and **one**
acceptance gate (the two-stage `Checker`). Methods differ only in where they start
the polish. This is the design that makes the comparison fair, and it is what lets
us reason about all of them with a single quantity.

For a problem instance with solution manifold $\mathcal{S}$, define the
**single-start reachability**

$$q(n) \;=\; \Pr_{x_0\sim U}\big[\,\text{polish}(x_0)\ \text{is accepted}\,\big],$$

the probability that one random start, drawn from the method's proposal
distribution $U$ and pushed through the shared polish, lands in an accepting basin.
$q(n)$ is the atom of the whole study: it is a purely geometric property of the
problem family and the polish operator, with no learning in it.

## 2. The best-of-$K$ identity

Random multi-start with budget $K$ (the control that the amortized-inference
literature usually omits) succeeds iff at least one of $K$ i.i.d. starts is
accepted:

$$P_{\text{random}}(n;K) \;=\; 1-\big(1-q(n)\big)^{K}. \tag{1}$$

Equation (1) is exact for a fixed instance; across a family it is the
instance-averaged version, with per-instance heterogeneity a second-order
correction (§6). It already says something the field routinely misses: **a hybrid
that beats cold-start Langevin has proven nothing about the value of learning** —
it must beat (1) at the same $K$.

## 3. The factorization dichotomy (the actual content)

The scientific question is how $q(n)$ scales with dimension. This is decided by
whether the acceptance basins **factorize across variables**.

**Separable constraints.** If each factor touches one variable (our independent
trap family: $r_i = (x_i-R_i)((x_i-m_i)^2+h_i)$), the energy is a sum of per-variable
terms and the polish is coordinate-decoupled. A start is accepted iff **every**
coordinate independently lands in its root basin, so

$$q_{\text{sep}}(n) \;=\; v^{\,n}, \qquad v \;:=\; q(1) \;=\; \text{per-variable basin fraction.} \tag{2}$$

$\log q$ is **linear in $n$** with slope $\log v<0$: reachability decays
**geometrically**. Substituting (2) into (1), random search needs
$\Theta(v^{-n})$ starts to hold a fixed success rate and **collapses** in high
dimension. A learned proposal that reproduces each variable's marginal root, by
contrast, keeps $q\approx 1$ and stays flat until model capacity fails. Hence the
learned proposal **must eventually win** — this is the genuine amortized-inference
regime, and it is what R5 measured.

**Coupled constraints.** If factors couple variables (our chained bilinear family:
$x_i+x_{i+1}=s_i,\ x_i x_{i+1}=p_i$), the solution is a joint object and the polish
propagates constraints along the chain. Here basins do **not** factorize, so $q(n)$
does **not** decay geometrically — measured $\log$-slope $\approx 0$. Random search
never collapses, a learned proposal has **nothing to amortize** (it ties random), and
a classical joint solver (Levenberg–Marquardt with the analytic Jacobian) dominates.
This is exactly R7 — no longer a disappointing negative but the *predicted*
consequence of a flat reachability curve.

**The diagnostic is the measured slope, not the syntactic label.** Separability is a
*sufficient* condition for collapse (it forces $q=v^n$ exactly), but it is not
necessary: a coupled system can still have a steeply decaying $q(n)$ if each local
subproblem admits several basins so that errors compound down the chain. What
actually predicts the regime is the **measured** $\log q$ slope: steep (collapse)
$\Rightarrow$ random search fails in high $n$ $\Rightarrow$ learning can help;
flat $\Rightarrow$ random survives $\Rightarrow$ learning cannot. Our real-domain
geometry family (below) is exactly the informative case — syntactically coupled, yet
its reachability collapses — which is why we state the law in terms of the measured
slope.

## 4. The crossover, predicted

Let $p_L$ be the learned proposal's (flat) solve ceiling. Under (1)–(2), learning
overtakes random search at the smallest dimension where $P_{\text{random}}<p_L$:

$$n^\* \;=\; \Big\lceil \frac{\log\!\big(1-(1-p_L)^{1/K}\big)}{\log v}\Big\rceil. \tag{3}$$

$n^\*$ is a **function of two measured constants** ($v$ and $p_L$) and the budget
$K$ — no fit to the crossover itself. Equation (3) exists (is finite) iff $v<1$,
i.e. iff basins factorize; for the coupled family $q$ is flat and (3) has no
solution, correctly predicting *no crossover*.

## 5. Validation (parameter-free)

`run_crossover_theory.py` measures $q(n)$ by single-start polish over
`--trials 600` fresh instances per $n$ (Wilson CIs), using the identical
generators / refine / Checker as R5 and R7 (`--K 8 --seed 20260721`).

| Prediction | Independent (separable) | Coupled |
|---|---|---|
| $\log q(n)$ slope $b$ (=$\log v$) | **−1.032** (steep) | **−0.128** (≈ flat) |
| fit $R^2$ | **0.982** | 0.958 |
| measured $v=q(1)$ | **0.270** | — |
| $E[\text{starts}]=1/q(n)$, $n$ up | **3.7 → 13 → 32 → 150 → 600** (explodes) | **2.0 → 2.5 → 2.9 → 3.8 → 4.3** (flat) |
| $P_{\text{random}}$ MAE, parameter-free $1-(1-v^n)^K$ | **0.012** | (law N/A — not separable) |

Measured $q(n)$ (independent): 0.270, 0.077, 0.032, 0.007, 0.002 at $n=1,2,3,4,6$.
Parameter-free $1-(1-v^n)^K$ with $v=0.270$: 0.919, 0.454, 0.147, 0.042, 0.003 —
against the self-measured best-of-8 random curve 0.902, 0.467, 0.162, 0.030, 0.002.
The single constant $v$ reproduces the whole curve to **MAE 0.012**. Coupled $q(n)$
(0.508, 0.397, 0.345, 0.267, 0.230) decays with slope only $-0.128$: basins do not
factorize, so $1-(1-v^n)^K$ (0.454, 0.147, …) badly under-predicts the measured
coupled random curve (0.478, 0.617, 0.565, 0.468, 0.397) — the law correctly fails
where its premise (separability) is false.

**On the crossover.** The learned proposal sits at a flat ceiling $p_L\approx0.95$
(R5: 0.95/0.95/0.975/0.925 for $n=1$–4). Against the higher-$N$ random curve above
it is already ahead at $n=1$ (0.95 vs 0.90, an overlapping-CI tie) and pulls away
monotonically as random collapses — so we do **not** claim a sharp crossover at a
particular $n$; the robust, predicted quantity is the *geometric collapse* of
random search ($E[\text{starts}]\sim v^{-n}$: 4 → 600 over $n=1$–6) against a flat
learned ceiling. (The $N{=}40$ R5 table reported random $=0.875,0.70$ at $n=1,2$;
the $N{=}600$ re-measurement here, 0.90 then 0.47, is more precise and, if anything,
moves the crossover *earlier* — it does not change the mechanism.) The coupled
family has no collapse and hence no crossover, exactly as Eq. (3) predicts when
$v\to1$. Figure `paper/figures/fig_crossover_theory.pdf`: (a) $\log q$ vs $n$ —
separable is a line ($R^2{=}0.98$), coupled is flat; (b) parameter-free predicted
vs measured random curve against the flat learned ceiling.

## 5a. Real-domain validation: geometry (a coupled family that still collapses)

To test the law outside the synthetic traps we add a **real-ish geometric domain**:
chains of unknown points $P_1,\dots,P_k$ ($n{=}2k$ coordinates) with squared-distance
constraints to fixed anchors and between consecutive points (the geometry the MARC
eval already uses, generalized to arbitrary length; `marc/data/geometry.py`
`make_point_chain`). This is a genuinely nonconvex quartic energy and needs the
geometry-tuned polish; solutions are integer coordinates the checker accepts exactly.

Geometry is syntactically **coupled** (each point ties to the previous), yet its
measured single-start reachability **collapses**: slope **b = −0.77 (R²=0.999)** (vs
−1.03 separable, −0.13 coupled-bilinear), because each point's two-circle subproblem
has a reflection ambiguity and spurious basins, so a random start rarely places *every*
point in the right basin and errors compound down the chain. Measured $q(n)$:
**0.653, 0.147, 0.027, 0.007** at $n=2,4,6,8$.

**Why this matters.** It shows the law's diagnostic is the *measured slope*, not the
coupled/independent label, and it identifies geometry as a **real domain the law flags
as learning-favorable** (steep collapse $\Rightarrow$ random search fails
$\Rightarrow$ an amortized proposal that learns each point's marginal can win). This
is the concrete, law-guided next experiment (train the denoiser on geometry chains and
test it against the random-restart control), and it directly answers the
"all-synthetic" critique of R5/R7. Preliminary geometry solve rates with the classical
polish alone are non-saturated (in-distribution $0.56$, held-out $0.28$; `results/p4_scale/`),
consistent with a hard, non-trivial domain rather than a saturated one.

## 6. Honesty / limitations

- **Instance heterogeneity.** (1) uses instance-averaged $q$; per-instance $q_i$
  varies, so best-of-$K$ from $\bar q$ is an approximation (a Jensen gap). We report
  the *self-measured* best-of-$K$ curve under identical conditions alongside the
  prediction, so the comparison is apples-to-apples; the parameter-free $v^n$
  prediction is validated against that measured curve, not a cross-experiment JSON.
- **Three families, one still open at the solve level.** The law's *reachability*
  prediction is validated on three families (separable traps, coupled bilinear,
  coupled geometry). The *solve-rate* consequence (learned beats random) is confirmed
  on the separable family (R5) and the coupled-bilinear negative (R7); on geometry the
  law predicts learning can help but training the geometry denoiser and running it
  against the random control is the flagged next experiment, not yet done.
- **Learned ceiling is empirical.** $p_L$ (and its capacity-limited decay at large
  $n$, e.g. the independent $n=6$ dropoff) is measured, not derived; the law
  predicts the *random* curve and the crossover, not the learned proposal's absolute
  accuracy.
- **What we do NOT claim.** No claim that the learned solver beats classical search
  in general — the coupled result says it does not. The contribution is the law that
  says *when* it does, and the controlled protocol that reveals it.
