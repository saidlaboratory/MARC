# MARC: Mathematical AI Reasoning Core

## Denoising diffusion over constraint graphs for verifiable mathematical reasoning

**Status:** Draft v0.1 — **superseded by the v0.2 orchestrator reframe (July 2026)** | **Date:** June 2026  
**Scope:** MVP framing and a phased research roadmap  
**Repo:** https://github.com/saidlaboratory/MARC

> **Reframing note (July 21, 2026).** This document is the founding v0.1 framing and is preserved unchanged as the historical record. The central bet below — a learned denoiser refining *values* toward consistency — was tested with pre-registered controls and lost (learned proposals tie/lose to random restart on coupled systems; Levenberg–Marquardt saturates the hard families; see `paper/RESULTS.md` R7 and the LM columns). MARC v0.2 keeps this document's substrate (constraint graphs), verification discipline (checker-only reward), and evaluation philosophy (derive-not-recall, structural holdout), but relocates the learned component to the one decision classical solvers cannot make: **what structure to add** (auxiliary variables, defining relations). Values are delegated to classical solvers. The current framing lives in [README.md](README.md); the team writing guide is [OUTLINE.md](OUTLINE.md); the evidence ledger is [paper/RESULTS.md](paper/RESULTS.md).

# 1\. Executive Summary

MARC is a novel mathematical reasoning system that avoids traditional left-to-right chain-of-thought token generation. Instead, it represents problems as **constraint graphs** and solves them by iteratively denoising the graph toward a globally consistent state. This system offloads exact computation to a computer algebra system (CAS) and utilizes a formal symbolic checker as the sole source of training reward.  
The design intent is a model that *derives* answers under verification rather than recalling them from training data. By focusing learned capacity on mathematical structure rather than arithmetic, MARC aims to achieve higher reliability and better generalization.

# 2\. Motivation and Problem Statement

Current Large Language Model (LLM) reasoning, primarily based on Chain-of-Thought (CoT) and Reinforcement Learning from Verifiable Rewards (RLVR), faces three critical limitations:

1. **Memorization:** Benchmark performance often reflects exposure to solutions rather than genuine derivation.  
2. **Faithfulness:** Verbalized chains may not reflect the actual internal computation, leading to post-hoc rationalization.  
3. **Arithmetic Brittleness:** Mechanical tracking of digits consumes model capacity better spent on structural reasoning.

The goal of MARC is to relocate computation into a structured substrate and offload mechanical arithmetic to an exact engine.

# 3\. Hypotheses and Research Questions

**Primary Hypothesis (H1):** Refinement of a structured constraint graph toward checker-verified consistency generalizes better than token-level CoT by preventing fallback on recalled sequences.  
**Secondary Hypothesis (H2):** Allowing the refinement process to modify graph structure (adding nodes/edges) enables the introduction of lemmas and auxiliary quantities.

## Key Research Questions

1. Can noise-driven iterative refinement reliably converge to checker-accepted solutions?  
2. Does injected noise reduce entrapment in inconsistent local fixed points?  
3. Does training only against a checker yield "derive-not-recall" behavior?

# 4\. Proposed Approach

## 4.1 Representation: The Constraint Graph

The problem is encoded as a hypergraph with two node types:

* **Variable Nodes:** Quantities carrying noisy values or embeddings.  
* **Factor Nodes:** Relations (equations, identities) that the solution must satisfy.

## 4.2 The Denoiser and Guidance

A Graph Neural Network (GNN) acts as a refiner, trained as a denoiser to move the graph toward a consistent configuration. At each step, a **Computer Algebra System (CAS)** computes exact residuals for every factor, providing a guidance signal similar to classifier guidance in diffusion models.

# 5\. Training Methodology

**Data Generation:** Procedural generation of problems with known solutions is used to hold out structural patterns rather than just specific numbers.  
**Training Stages:**

* **Stage A (Denoising):** Bootstrap a competent refiner via corruption-and-reverse training.  
* **Stage B (Checker Fine-Tuning):** RLVR fine-tuning rewarding only checker-accepted derivations, enforcing derive-not-recall.

# 6\. Evaluation Plan

Beyond raw solve rates, success is measured by:

* **Generalization Gap:** Performance on held-out structures and larger problem sizes.  
* **Robustness:** Resistance to errors when constants are perturbed.  
* **Verifiability Rate:** Fraction of solutions where the *entire derivation* is accepted by the checker.

# 7\. Phased Roadmap

* **P0:** Infrastructure (Schema, Generator, CAS Interface).  
* **P1:** Value-diffusion MVP on fixed structures (algebra systems).  
* **P2:** Checker fine-tuning with RLVR.  
* **P3:** Structure diffusion (graph-growing for complex objects).

# 8\. Risks and Open Questions

* **Discrete Graph Diffusion Complexity:** Noise schedules for symbolic structures are risky; prioritize value diffusion.  
* **Autoformalization Bottleneck:** Converting natural language to graphs is a separate hard problem.  
* **Strategy vs. Arithmetic:** The CAS solves arithmetic, but high-level strategy remains the denoiser's challenge.

# 9\. Positioning and Prior Art

MARC combines existing paradigms into a unique reasoning engine:

* **Graph Diffusion (DiGress):** Adapted for mathematical substrates.  
* **Latent Reasoning (Coconut):** MARC provides a *structured, checkable* latent.  
* **Neurosymbolic (AlphaProof):** Shared verifier-centric philosophy.

# 10\. Success Criteria and Falsification

The program is successful if MARC shows smaller generalization gaps and higher robustness than CoT baselines. It is **falsified** if injected noise fails to reduce entrapment in local minima or if the generalization gap remains equivalent to traditional token-level models.  
