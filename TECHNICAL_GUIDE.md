# MARC — Technical Design & Implementation Guide

**Mathematical AI Reasoning Core**
**Document type:** deep technical specification + build guide
**Status:** Draft v0.1 · companion to the research brief
**Audience:** the MARC engineering/research team (read top-to-bottom before P0)

---

## How to read this document

This is the *engineering* counterpart to the research brief. The brief says **why** MARC exists and **what** the bet is; this document says **how to build it**, in enough detail to start writing code. It moves from formal definitions → the diffusion/denoising mathematics → the neural architecture → the training procedures (with pseudocode) → inference → evaluation → repo layout.

Notation is collected in §12. Anything marked **[MVP]** is the minimum viable path you should build first; **[Frontier]** marks the harder, higher-risk extensions. When in doubt, build the **[MVP]** path end-to-end before touching **[Frontier]** pieces.

A one-paragraph mental model to carry throughout:

> A math problem is a **graph of constraints**. We want to sample an assignment of values (and possibly extra structure) that satisfies all of them. We do this by **denoising**: start from noise, and repeatedly refine the whole graph toward consistency, where a **learned prior** (a graph neural network) proposes refinements and an **exact calculator** (a computer-algebra system) tells us how wrong we still are. A **checker** decides when we are done. Training rewards only what the checker accepts, so the model must *derive*, not recall.

---

## 1. Conceptual foundation (condensed)

Chain-of-thought (CoT) reasoning has three weaknesses we want to engineer away: it can **recall** memorized solutions (inflating benchmarks), its verbalized steps can be **unfaithful** to the true computation, and it **fumbles arithmetic** while spending capacity tracking digits. CoT is *not* mere retrieval — its tokens add serial compute — but the failure modes are real.

MARC's response is to **relocate** the computation into a structured, checkable object (a constraint graph), **offload** arithmetic to an exact engine, and **train only against a verifier**. Mathematics is the ideal first domain precisely because verification is cheap and objective.

The core technical bet is the fusion of two paradigms:

- **Constraint relaxation** — represent the problem as a factor graph; solving = driving all constraints to mutual consistency.
- **Denoising diffusion** — refine a corrupted estimate of the whole solution toward validity; injected noise lets the relaxation escape locally-consistent-but-globally-wrong fixed points.

Fused, these become **denoising diffusion over a constraint graph**.

---

## 2. Formal problem definition

### 2.1 The constraint (factor) graph

A **problem instance** is a bipartite factor graph

$$G = (V, F, E), \qquad V = \{v_1,\dots,v_n\},\quad F = \{f_1,\dots,f_m\}$$

- **Variable nodes** $V$: the quantities of the problem (knowns and unknowns). Each carries a state $x_v \in \mathbb{R}^d$ (for **[MVP]** scalar problems, $d=1$; embeddings of integers/rationals use $d>1$, see §6.3).
- **Factor nodes** $F$: the constraints. Factor $f_i$ has a **scope** $S_i \subseteq V$ (the variables it relates) and a **residual function** $r_i$.
- **Edges** $E$: connect each factor to the variables in its scope. Each edge $(v,f)$ carries **edge features** $e_{vf}$ encoding the *role* of $v$ in $f$ (e.g. its coefficient, exponent, or operator slot). **Edge features are not optional** — they are how the network knows that $f$ is "$2x - y + z = 6$" and not just "some constraint over $\{x,y,z\}$".

The assignment is $x = (x_{v_1},\dots,x_{v_n})$. Each factor produces a **residual**

$$r_i(x) = f_i\!\left(x_{S_i}\right) \in \mathbb{R},$$

defined so that the constraint is satisfied iff $r_i = 0$ (equalities) or $r_i \le 0$ (inequalities, via a hinge — see §2.3).

### 2.2 The consistency energy

Define the scalar **energy**

$$E(x) = \tfrac{1}{2}\sum_{i=1}^{m} w_i\, \rho\!\big(r_i(x)\big),$$

where $\rho$ is a penalty (squared, $\rho(r)=r^2$, for equalities) and $w_i>0$ are per-factor weights. A **solution** is any $x^\star$ with $E(x^\star)=0$, i.e. all residuals vanish. The solution set may be a point, a manifold, or empty.

This single function does a lot of work:
- it is the **target** the denoiser is pushing down,
- its gradient $\nabla_x E$ is the **guidance signal** the calculator supplies,
- it provides **reward shaping** in RL (§8.2),
- $E=0$ (up to tolerance) is the **checker's** numerical gate (§7).

### 2.3 Constraint types

| Type | Residual $r_i$ | Penalty $\rho$ |
|---|---|---|
| Equality $g(x)=0$ | $g(x)$ | $r^2$ |
| Inequality $g(x)\le 0$ | $\max(0, g(x))$ | $r^2$ (one-sided) |
| Definition $v := h(\cdot)$ | $x_v - h(\cdot)$ | $r^2$ |
| Rule applicability (Boolean) | soft indicator $\in[0,1]$ | $-\log(\cdot)$ |

For **[Frontier]** symbolic reasoning, a factor can also be a *rewrite rule* whose residual measures whether a claimed transformation is valid; this is where structure diffusion (§10) enters.

---

## 3. Reasoning as denoising diffusion over the graph

We treat the solution $x$ (and, in the frontier case, the graph structure) as the object to be **generated by denoising**. Two regimes:

- **[MVP] Value diffusion** — $G$ is fixed; only node values $x$ are noised/denoised. Continuous Gaussian diffusion.
- **[Frontier] Structure diffusion** — the graph itself ($V,F,E$) is noised/denoised. Discrete diffusion (§10).

### 3.1 The probabilistic target

We want to sample assignments that satisfy the constraints. Cast as a posterior:

$$p(x \mid G) \;\propto\; \underbrace{p_{\text{prior}}(x)}_{\text{learned valid-solution manifold}} \cdot \underbrace{\exp\!\big(-E(x)\big)}_{\text{constraint likelihood}}.$$

The **prior** captures what valid math solutions look like in general (learned from data); the **energy term** is exact and instance-specific (computed by the CAS). The score of the posterior decomposes:

$$\nabla_x \log p(x\mid G) = \underbrace{\nabla_x \log p_{\text{prior}}(x)}_{\text{learned score}} \;-\; \nabla_x E(x).$$

This decomposition is the heart of MARC: **the network learns the prior; the calculator supplies the exact constraint gradient.** Sampling combines them.

### 3.2 Forward (corruption) process — value diffusion **[MVP]**

Standard variance-preserving Gaussian diffusion (DDPM). With schedule $\{\beta_t\}_{t=1}^{T}$ and $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).$$

$x_0$ is a *valid* solution (from the generator, §8.3); $x_T$ is ~pure noise. The graph $G$ and edge features are **not** corrupted in the MVP — only the values on variable nodes.

> **Schedule choice.** Start with a cosine schedule, $T=1000$ for training. For inference you will use far fewer steps (DDIM-style, 20–50). Tune $T$ and the schedule early; it strongly affects convergence.

### 3.3 Reverse (denoising) process

A network $\epsilon_\theta(x_t, t, G)$ predicts the noise (equivalently an $x_0$-prediction). Its score estimate is

$$s_\theta(x_t, t, G) = -\frac{\epsilon_\theta(x_t,t,G)}{\sqrt{1-\bar\alpha_t}} \;\approx\; \nabla_{x_t}\log q_t(x_t).$$

The **guided** reverse step injects the constraint gradient (energy / classifier guidance):

$$\tilde{s}(x_t,t) = s_\theta(x_t,t,G) - \lambda_t \,\nabla_x E(x_t),$$

with a guidance weight $\lambda_t$ (annealed; stronger at low noise). One ancestral sampling step:

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\Big(x_t + \beta_t\,\tilde{s}(x_t,t)\Big) + \sqrt{\beta_t}\, z, \quad z\sim\mathcal N(0,I).$$

The $\nabla_x E$ term is computed exactly by the CAS (§6): $\nabla_x E = \sum_i w_i\, r_i(x)\, \nabla_x r_i(x)$.

### 3.4 The simpler MVP you should actually build first **[MVP]**

The full SDE machinery is often overkill. The **load-bearing ingredient is noise for exploration, not the formalism.** Build *learned iterative refinement with injected noise* first:

$$x^{(k+1)} = x^{(k)} + g_\theta\!\big(x^{(k)}, r^{(k)}, G, k\big) + \sigma_k\,\xi, \qquad \xi\sim\mathcal N(0,I),$$

where $g_\theta$ is the GNN (§5) predicting a refinement direction, $r^{(k)} = (r_1(x^{(k)}),\dots)$ are the current residuals (from the CAS), and $\sigma_k$ is an annealed noise scale. This is a learned, stochastic generalization of gradient descent / a Newton step on $E$. It is trainable by either of:

1. **Denoising regression** (§8.1): corrupt $x_0$, train $g_\theta$ to recover it in one or a few steps.
2. **RL** (§8.2): reward energy reduction + checker acceptance.

Decide empirically whether you ever need full diffusion. **Validate the iterative-refinement path before committing to the SDE; it can save weeks.**

---

## 4. Why the fusion works (and what each half fixes)

| Failure of pure constraint relaxation | How diffusion fixes it |
|---|---|
| Deterministic message passing stalls at locally-consistent fixed points | Injected noise + a learned denoiser escape and explore |
| Hand-designed update rules don't generalize | The update is *learned* end-to-end |

| Failure of pure (token) diffusion | How the constraint graph fixes it |
|---|---|
| No structured object to denoise | The factor graph *is* the object |
| No exact correctness signal | The CAS residual / energy is exact and per-constraint |
| Hard to verify | The checker gates on $E=0$ |

Message passing and denoising are the **same operation** — local updates propagated over a structure and iterated to convergence — so implementing one as the other is natural, not forced.

---

## 5. The denoiser: a residual-conditioned graph network

### 5.1 Requirements

- Operates on the **bipartite factor graph** (variable nodes ↔ factor nodes).
- **Permutation-equivariant** over variables and factors (no canonical ordering).
- Conditioned on the **noise level / step** $t$ and on the **current residuals** $r_i$ (the CAS guidance, fed as factor features).
- Handles **variable graph sizes** (different problems have different $n,m$) — this falls out of message passing.
- Decodes a per-variable output: predicted noise $\hat\epsilon_v$ (diffusion) or refinement $\Delta x_v$ (iterative).

### 5.2 Message-passing equations

Maintain variable embeddings $h_v\in\mathbb{R}^{D}$ and factor embeddings $h_f\in\mathbb{R}^{D}$. Initialize from raw features:

$$h_v^{(0)} = \mathrm{MLP}_v\big([\,x_v;\, \mathrm{type}(v)\,]\big),\qquad
h_f^{(0)} = \mathrm{MLP}_f\big([\,\mathrm{type}(f);\, r_f;\, \tau(t)\,]\big),$$

where $\tau(t)$ is a sinusoidal/step embedding and $r_f$ is the factor's current residual from the CAS. Then run $L$ rounds of bipartite message passing (each round = one "denoise sub-step"):

**Factor update** (gather from its variables):

$$h_f^{(\ell+1)} = \phi_f\Big(h_f^{(\ell)},\ \textstyle\bigoplus_{v\in S_f}\psi_{v\to f}\big(h_v^{(\ell)}, h_f^{(\ell)}, e_{vf}\big)\Big)$$

**Variable update** (gather from its factors):

$$h_v^{(\ell+1)} = \phi_v\Big(h_v^{(\ell)},\ \textstyle\bigoplus_{f\ni v}\psi_{f\to v}\big(h_f^{(\ell+1)}, h_v^{(\ell)}, e_{vf}\big)\Big)$$

$\bigoplus$ is a permutation-invariant aggregator (sum, mean, or attention; **start with sum/mean, move to attention if needed**). $\phi,\psi$ are MLPs (optionally gated/residual + LayerNorm). $e_{vf}$ (edge features: coefficient/role) are injected into every message.

**Decode** after $L$ rounds:

$$\hat\epsilon_v = \mathrm{MLP}_{\text{out}}\big(h_v^{(L)}\big)\quad\text{(diffusion)}\qquad\text{or}\qquad \Delta x_v = \mathrm{MLP}_{\text{out}}\big(h_v^{(L)}\big)\quad\text{(iterative)}.$$

### 5.3 Forward-pass pseudocode

```python
def denoiser(G, x_t, t):
    # G: factor graph (edges, scopes, types, edge_features)
    # x_t: current variable values  [n, d]
    # t:  noise level / step index
    r = cas.residuals(G, x_t)              # [m]  exact, from the calculator
    h_v = mlp_v(cat([x_t, type_emb(G.var_types)]))           # [n, D]
    h_f = mlp_f(cat([type_emb(G.fac_types), r[:,None], step_emb(t)]))  # [m, D]

    for _ in range(L):
        # factor <- variables
        m_vf = psi_v2f(h_v[G.src], h_f[G.dst], G.edge_feat)  # per-edge messages
        agg_f = scatter_sum(m_vf, G.dst, dim_size=m)
        h_f = phi_f(cat([h_f, agg_f]))
        # variable <- factors
        m_fv = psi_f2v(h_f[G.dst], h_v[G.src], G.edge_feat)
        agg_v = scatter_sum(m_fv, G.src, dim_size=n)
        h_v = phi_v(cat([h_v, agg_v]))

    return mlp_out(h_v)        # predicted noise eps_hat  [n, d]
```

> **Library:** PyTorch Geometric (`torch_geometric`) gives you `scatter`, bipartite message passing, and batching of variable-size graphs for free. Use a `HeteroData` graph with `variable` and `factor` node types.

### 5.4 Capacity & depth

The number of message-passing rounds $L$ sets how far information propagates per network call. For a graph of diameter $\Delta$, you need $L \gtrsim \Delta$ to let constraints "see" each other in one denoise step; smaller $L$ is fine because you iterate the whole denoiser over many steps. Start with $L=4$–$8$, $D=256$.

---

## 6. The calculator (CAS) interface

The calculator is a **first-class component**, not a utility. It computes residuals and (for guidance) residual gradients exactly.

### 6.1 API

```python
class CAS:
    def residuals(self, G, x) -> Tensor:      # r_i(x) for each factor  [m]
        ...
    def energy(self, G, x) -> Tensor:         # E(x) = 0.5 * sum w_i rho(r_i)
        ...
    def energy_grad(self, G, x) -> Tensor:    # ∇_x E(x)  [n, d]   (guidance)
        ...
    def accepts(self, G, x, tol=1e-6) -> bool:# all |r_i| <= tol  (numeric gate)
        ...
```

### 6.2 Implementation notes

- **MVP**: factors are typed algebraic expressions; residuals evaluate the expression at $x$. Use **SymPy** to parse/define factor templates and to get **analytic Jacobians** $\partial r_i/\partial x_v$ once per template (lambdified for speed). Analytic gradients are exact and fast; prefer them over autodiff through the CAS.
- **Batching**: residual evaluation must be vectorized over the batch and over factors. Precompile each factor *template* (e.g. "linear: $a^\top x - b$") to a fast closure; instances differ only in coefficients (edge features) and $b$.
- **Guidance gradient**: $\nabla_x E = \sum_i w_i\,\rho'(r_i)\,\nabla_x r_i$. For squared penalty, $\rho'(r)=r$, so $\nabla_x E = \sum_i w_i\, r_i\, \nabla_x r_i$. For linear factors $\nabla_x r_i$ is just the coefficient row — cheap.
- **Differentiability of training**: in Stage A you do **not** backprop through the CAS (residuals are inputs/targets). In guided sampling you only need $\nabla_x E$ as a numeric vector, not a differentiable graph.

### 6.3 Number representation

How a value enters a node matters (this is the lowest-level lever). Options, increasing in cost:

1. **[MVP]** scalar float per node ($d=1$). Simplest; fine for early algebra/equation systems.
2. Fixed-point / digit-vector embedding for exact integer/rational reasoning.
3. Learned magnitude-aware embeddings (xVal / Fourier-feature style) if float precision becomes the bottleneck.

Keep $d=1$ until experiments show representation is limiting; then upgrade only the node encoder/decoder.

---

## 7. The checker

The checker is the **terminal gate** at inference and the **reward source** in training.

- **Numeric gate** (always available): accept iff $\max_i |r_i(x)| \le \texttt{tol}$. Cheap, used every step for early stopping.
- **Symbolic gate** (**[MVP]** for algebra): use SymPy to verify the candidate satisfies the original equations exactly (rational arithmetic), guarding against floating-point false accepts.
- **Formal gate** (**[Frontier]**): translate the derivation into Lean/Isabelle and require the proof kernel to accept. This is the gold standard for "derive, not recall" (you cannot pass without a machine-checkable derivation) but requires autoformalization (P4). Use it only when value diffusion is stable.

> The checker must be **conservative**: a false accept poisons RL training (rewards a wrong solution). Prefer symbolic/exact acceptance over loose numeric tolerance once you leave the float-only MVP.

---

## 8. Training methodology

Two stages: **(A)** denoising pretraining to get a competent refiner, then **(B)** checker-based RL to enforce derive-not-recall.

### 8.1 Stage A — denoising / score-matching pretraining **[MVP]**

Train $\epsilon_\theta$ (or $g_\theta$) to reverse the corruption on *valid* graphs from the generator.

**Loss (diffusion form):**

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x_0\sim\mathcal D,\ t\sim\mathcal U(1,T),\ \epsilon}\Big[\,\big\|\epsilon - \epsilon_\theta(x_t, t, G)\big\|^2\,\Big].$$

**Loss (iterative-refinement form):** predict the clean target (or the residual-reducing step) from the corrupted state:

$$\mathcal{L}_{\text{IR}} = \mathbb{E}\Big[\big\| (x_0 - x^{(k)}) - g_\theta(x^{(k)}, r^{(k)}, G, k)\big\|^2\Big] \;+\; \gamma\,\mathbb{E}\big[E(x^{(k)}+g_\theta(\cdot))\big].$$

The second term ties the learned step to *actual energy reduction* (the CAS supplies $E$).

**Training step pseudocode:**

```python
def train_step_A(batch):
    G, x0 = batch.graph, batch.solution      # x0 is a VALID assignment
    t   = randint(1, T, (B,))
    eps = randn_like(x0)
    abar = alpha_bar[t]
    x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps    # forward corruption
    eps_hat = denoiser(G, x_t, t)             # network (calls CAS for residuals)
    loss = mse(eps_hat, eps)
    loss.backward(); opt.step(); opt.zero_grad()
    return loss
```

### 8.2 Stage B — checker-based RL (the denoising MDP) **[MVP→core]**

Treat the reverse process as a finite-horizon **MDP**: state $=(x_k, G)$, action $=$ the sampled denoising step, transition $=$ apply the step, terminal reward $=$ checker. Because each step is a Gaussian sample, its log-probability is tractable, so policy-gradient methods apply (the "diffusion-as-policy" view).

**Reward** (terminal + potential-based shaping that preserves the optimum):

$$R = \underbrace{B\cdot\mathbb{1}[\text{checker accepts } x_K]}_{\text{terminal}} \;+\; \sum_{k} \underbrace{\big(E(x_{k-1}) - E(x_k)\big)}_{\text{energy reduction per step}}.$$

**Optimizer: GRPO.** For a problem $q$, sample a **group** of $N$ rollouts $\{o_1,\dots,o_N\}$ with returns $\{R_1,\dots,R_N\}$. Use the group as its own baseline:

$$A_i = \frac{R_i - \operatorname{mean}(R_{1:N})}{\operatorname{std}(R_{1:N}) + \varepsilon}.$$

Maximize the clipped surrogate with a KL leash to a reference policy $\pi_{\text{ref}}$ (the Stage-A model):

$$\mathcal{J}_{\text{GRPO}} = \mathbb{E}\Big[\ \min\big(\rho_i A_i,\ \operatorname{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon) A_i\big) - \beta\,\mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)\ \Big],\qquad \rho_i = \frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}.$$

GRPO drops the value network (cheaper than PPO) — well-suited here since the group baseline is natural.

**The purist variant (truest to "derive, not recall"):** set $B$ as the *only* reward and use **no reference solutions at all** — the model is optimized solely to produce checker-accepted derivations. Start with shaping (faster), then ablate it away to test whether the purist signal suffices.

```python
def train_step_B(q):                      # q: a problem (graph, no solution given)
    rollouts = [sample_trajectory(policy, q) for _ in range(N)]  # stochastic denoising
    R = [reward(o, q) for o in rollouts]   # checker (+ energy shaping) via CAS
    A = (R - mean(R)) / (std(R) + 1e-8)    # group-relative advantage
    logp = [policy.logprob(o) for o in rollouts]   # sum of Gaussian step log-probs
    loss = -(clip_surrogate(logp, logp_old, A)).mean() + beta*kl(policy, ref)
    loss.backward(); opt.step(); opt.zero_grad()
```

### 8.3 Data generation & curriculum

You generate problems **with known solutions** by construction, then formalize them into graphs:

```python
def generate_instance(structure_template, difficulty):
    x_star = sample_solution(structure_template, difficulty)   # pick the answer first
    factors = instantiate_factors(structure_template, x_star)  # built to be satisfied by x_star
    G = build_factor_graph(structure_template, factors)
    assert cas.accepts(G, x_star)                              # invariant: residuals vanish at x_star
    return G, x_star
```

- **Anti-memorization is structural, not numeric.** Partition *structure templates* into train/test so the test set has shapes never seen in training. Numeric perturbation alone is too weak.
- **Curriculum:** schedule `difficulty` (number of variables, factor density, coefficient range, nonlinearity). Empirically, training on the *hardest* solvable tier transfers best — bias the schedule toward harder instances once basics are learned.
- **Length/compositional generalization:** train on $\le k$ variables, **evaluate on $> k$**. This is your primary derive-not-recall probe.

---

## 9. Inference / sampling

```python
def solve(G, steps=40, N=8, guidance=lambda t: 1.0):
    best = None
    for _ in range(N):                       # best-of-N under the checker
        x = randn(n, d)                      # start from noise
        for t in reversed(range(steps)):
            eps_hat = denoiser(G, x, t)
            score   = -eps_hat / sqrt(1 - alpha_bar[t])
            score  -= guidance(t) * cas.energy_grad(G, x)   # exact constraint guidance
            x = ddim_step(x, score, t)        # one (guided) reverse step
            if cas.accepts(G, x):             # early stop the moment it's consistent
                return x
        if best is None or cas.energy(G, x) < cas.energy(G, best):
            best = x
    return best                               # checker verifies before returning
```

Key behaviors: **early-stop** as soon as the checker accepts; **best-of-N** with the energy as tiebreaker; **resample on failure** is safe because verification is cheap. Use **few-step** sampling (DDIM, 20–50 steps) at inference even if trained with $T=1000$.

---

## 10. Structure diffusion (the frontier) **[Frontier]**

Everything above fixes the graph and denoises only values. The frontier is to **denoise the graph itself**, so the model can *invent intermediate objects* — an auxiliary variable, a lemma, a substitution — that a static graph cannot express. This is what lets MARC eventually do open-ended derivation rather than only "solve for a consistent state."

**Representation.** Pad to a maximum slot count: a fixed pool of $N_{\max}$ variable slots and $M_{\max}$ factor slots, each with a categorical **type** including a special `ABSENT` type. Edges are categorical too. The graph state is now $(c_V, c_F, c_E)$ — categorical node/edge types — plus values $x$.

**Discrete diffusion (D3PM / absorbing).** The forward process corrupts categories toward an absorbing `[MASK]`/`ABSENT` state with a categorical transition matrix $Q_t$:

$$q(c_t \mid c_{t-1}) = \operatorname{Cat}\!\big(c_t;\ p = c_{t-1} Q_t\big).$$

The reverse model predicts $p_\theta(c_0\mid c_t)$ per slot/edge. **Instantiating a new node** = a slot transitioning from `ABSENT` to a concrete type during denoising; **removing one** = the reverse. Run **value diffusion and structure diffusion jointly** (continuous + categorical) in one denoiser.

**Why it's hard (and scoped to preliminary results for the paper):**
- Defining a corruption process and noise schedule over *symbolic structure* is not well understood.
- Variable node count, permutation invariance, and the joint continuous+discrete process compound.
- RL over structure-changing actions is high-variance.

Build only after **[MVP]** value diffusion + checker RL is solid. For AAAI, target *early, promising* structure-diffusion results, not a finished system.

---

## 11. Evaluation protocol

Report capability **and**, more importantly, the metrics that test the hypotheses.

| Metric | Definition | Tests |
|---|---|---|
| Solve rate (pass@1, pass@k) | fraction of problems the checker accepts | capability |
| **Generalization gap** | (in-distribution solve rate) − (held-out-structure solve rate) | H1 / derive-not-recall |
| **Length extrapolation** | solve rate vs. #variables, trained on $\le k$, tested $>k$ | H1 |
| **Perturbation robustness** | solve-rate drop when constants are perturbed | recall detector |
| **Entrapment rate** | fraction of runs stalling at $E>\texttt{tol}$ fixed points, **with vs. without injected noise** | RQ2 (noise value) |
| Derivation-verifiability | fraction of *accepted* solutions whose full derivation the checker accepts | faithfulness |
| Intermediate-object usage | (frontier) do auxiliary nodes appear, and do they correlate with solving harder problems? | H2 |

**Baselines:** a token CoT model of comparable scale; deterministic constraint relaxation (no learned denoiser); the model with guidance off; value-only vs. value+structure.
**Ablations:** noise on/off (the key one), guidance weight $\lambda$, number of message rounds $L$, sampling steps, shaping vs. purist reward.
**Reporting:** multiple seeds, confidence intervals, and explicit train/test structural splits documented per experiment.

The central empirical claim is supported if MARC shows a **smaller generalization gap and greater perturbation robustness than the CoT baseline at equal scale**, with high derivation-verifiability. It is falsified if noise doesn't reduce entrapment, or the generalization gap is no better than CoT.

---

## 12. Notation

| Symbol | Meaning |
|---|---|
| $G=(V,F,E)$ | bipartite factor graph: variables, factors, edges |
| $x_v\in\mathbb R^d$ | state/value on variable node $v$ |
| $S_i$ | scope (variables) of factor $f_i$ |
| $r_i(x)$ | residual of factor $i$ (0 ⇒ satisfied) |
| $e_{vf}$ | edge feature (role/coefficient of $v$ in $f$) |
| $E(x)$ | consistency energy $\tfrac12\sum w_i\rho(r_i)$ |
| $\beta_t,\bar\alpha_t$ | diffusion noise schedule |
| $\epsilon_\theta,g_\theta$ | denoiser network (noise-pred / refinement) |
| $s_\theta$ | learned score; $\tilde s$ guided score |
| $\lambda_t$ | guidance weight |
| $A_i,\rho_i$ | GRPO group advantage, importance ratio |
| $B$ | terminal checker reward magnitude |
| $N_{\max},M_{\max}$ | max variable/factor slots (structure diffusion) |

---

## 13. Implementation plan & stack

**Stack**

- **PyTorch** + **PyTorch Geometric** — the GNN denoiser, heterogeneous bipartite graphs, batched scatter ops.
- **SymPy** — factor templates, residuals, analytic Jacobians, symbolic checker (MVP).
- **(Frontier) Lean 4 + a proof-interaction layer** — formal checker; autoformalization (P4).
- **Hydra / OmegaConf** — config; **Weights & Biases** — experiment tracking.
- **NumPy/SciPy** — reference solvers and sanity baselines.

**Suggested repo layout**

```
marc/
  graph/         # factor-graph schema, serialization, HeteroData builders
  cas/           # residuals, energy, energy_grad, symbolic checker  (SymPy)
  data/          # problem generators, structure templates, curriculum
  model/         # GNN denoiser, time/step embeddings, decoders
  diffusion/     # schedules, forward corruption, samplers (DDPM/DDIM), guidance
  refine/        # [MVP] learned iterative refinement variant
  train/         # stage_a (DSM/IR), stage_b (GRPO denoising-MDP)
  eval/          # metrics, generalization splits, ablation runners
  structure/     # [Frontier] discrete graph diffusion (D3PM)
  configs/       # hydra configs per experiment
```

**Build order (maps to the timeline)**

1. `graph` + `cas` + `data` (P0) — get a generator that emits graphs whose residuals vanish at the known solution; assert the invariant.
2. `model` + `diffusion`/`refine` + `train/stage_a` (P1) — value diffusion that converges on in-distribution instances; first length-gen curve.
3. `train/stage_b` + `eval` (P2) — checker RL; the derive-not-recall results that anchor the paper.
4. `structure` (P3) — preliminary discrete structure diffusion.
5. autoformalizer + scaling (P4) — future work.

---

## 14. Design decisions, risks, and honest caveats

- **Build the simplest thing that exhibits the phenomenon first.** Learned iterative refinement with noise before full diffusion; value diffusion before structure diffusion; numeric+symbolic checker before formal. Each step de-risks the next.
- **The noise is the point.** If ablating noise does not reduce the entrapment rate, the core hypothesis is in trouble — run that ablation early.
- **The calculator fixes arithmetic, not strategy.** Most hard-problem failures are reasoning errors; the CAS removes a bounded error class. Don't expect it to do the reasoning.
- **A false-accepting checker is the most dangerous bug** in RL — it rewards wrong answers. Prefer exact/symbolic acceptance; treat numeric tolerance as a fast pre-filter only.
- **Structure diffusion may not converge in time.** It is correctly scoped as preliminary; the paper's defensible core is value diffusion + checker RL + the derive-not-recall evaluation.
- **Specialize, don't lobotomize.** A math-only-from-scratch model loses the ability to parse problems; favor heavy specialization with the rules exposed as the CAS engine.
- **Multiple valid solutions** are fine — diffusion samples one; evaluation must account for it (verify acceptance, not equality to a single reference).

---

## 15. Reading list (method lineages, for grounding)

These are the *components* MARC fuses; the contribution is the combination as a math-reasoning engine plus structure-denoising for object invention. Read for mechanism, not to copy.

- **Diffusion / score-based generation** — DDPM; score matching; DDIM fast sampling; classifier / energy guidance.
- **Diffusion as policy** — denoising-diffusion policy optimization (RL fine-tuning of samplers).
- **Discrete & graph diffusion** — D3PM (categorical/absorbing diffusion); graph diffusion (DiGress-style) for structure.
- **Verifiable-reward RL** — GRPO; process reward models; "verify step by step."
- **Graph neural networks / factor graphs** — message passing; neural constraint & SAT solvers; diffusion for combinatorial optimization.
- **Latent & structured reasoning** — continuous chain-of-thought; looped/recurrent-depth transformers.
- **Formal & tool-augmented math** — Lean/Mathlib; AlphaProof / proof-search systems; Program-of-Thoughts / PAL.
- **Number representation** — Abacus embeddings; xVal; Fourier number embeddings.

---

*End of guide. The two things to lock before P0: (1) the constraint-graph schema in `graph/`, and (2) the Stage-A vs Stage-B training contract (what the CAS returns, and the exact reward definition). Everything else can iterate.*