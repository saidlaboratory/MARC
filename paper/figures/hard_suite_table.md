# Hard-suite results (non-convex families)

_Best-of-8, 60 held-out problems/family, trained 250 epochs. 95% Wilson CIs. `refine` and `random restart` are classical baselines; **random restart + polish is the control that isolates the learned proposal**._

| Family | refine cold | refine+Langevin | **random restart+polish (control)** | learned hybrid | learned>random? |
|---|---|---|---|---|---|
| BilinearSystem | 0.000 | 0.300 | **0.550 [0.42, 0.67]** | 0.550 [0.42, 0.67] | tie/no (p=0.50) |
| BilinearProduct | 0.000 | 0.100 | **0.717 [0.59, 0.81]** | 0.683 [0.56, 0.79] | tie/no (p=0.65) |
| QuadraticSystem | 0.000 | 0.300 | **0.683 [0.56, 0.79]** | 0.683 [0.56, 0.79] | tie/no (p=0.50) |
| CircleLine | 0.000 | 0.033 | **0.200 [0.12, 0.32]** | 0.000 [0.00, 0.06] | tie/no (p=1.00) |

**learned_hybrid beats refine+Langevin on 3/4 families (p<0.05), but beats the random-restart control on 0/4.**

**Honest reading:** the hybrid recipe (a good proposal + energy-descent polish) beats cold-start Langevin — but a *random* init + the same polish does just as well as the *learned* proposal on these small-solution families (learned ties on 2, loses on 2). So the contribution here is the **hybrid recipe**, not the learned denoiser; the learned proposal's advantage appears only in high dimension where random restart fails (see the dimension-scaling result). CircleLine is an outright failure (0.000).