# LLM + MARC verified solving (Lever B) — feasibility: negative

**Hypothesis:** LLM formalizes a problem → MARC/CAS solves it exactly → fixes the LLM's
computational errors, beating the LLM solving directly. A "verified reasoning" main-track angle.

## Result (Gemini flash-lite, MATH-500 sample, symbolic answer-equality)
| pipeline | accuracy |
|---|---|
| LLM direct (CoT) | ~0.80 |
| LLM formalize → SymPy solve | ~0.00 |
| formalization coverage | 1.00 (produces equations) but solving is wrong/empty |

Formalize-then-solve **does not help — it collapses**. Failure modes: evaluation problems don't
map to equation systems (returns nothing); when it does formalize, the solve is often wrong
(e.g. 13536 vs gold 13535; 0 vs gold 5). The LLM is already strong on these problems, and the
formalization step is a **brittle, lossy bottleneck** that loses far more than exact solving
recovers.

## Implication
The verified-reasoning route (LLM + MARC/CAS) does not beat the LLM alone on MATH. Combined with
R7 (learned ties random on coupled systems) and the structure-invention negative (auxiliary hurts
the numeric solver), **all three main-track routes testable in-week are negative.**

## Honest conclusion
There is no main-track-worthy positive result achievable with this system in the 7-day window.
The genuine, defensible contribution is **workshop-level**: a working neuro-symbolic diffusion
constraint solver + the entrapment result + a rigorous, well-controlled study (including these
negatives, which are honest and reviewer-respected). A real main-track effort would require a
different, larger investment: GPU-scale training on a real problem family with a genuine positive,
or a new method — beyond this deadline.
