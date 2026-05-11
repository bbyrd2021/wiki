---
type: paper
title: "Pearl 2009 — Causality: Models, Reasoning, and Inference"
aliases: ["Pearl Causality", "Pearl 2009", "SCM", "do-calculus", "causal DAG"]
created: 2026-04-08
updated: 2026-04-08
sources: ["https://archive.illc.uva.nl/cil/uploaded_files/inlineitem/Pearl_2009_Causality.pdf"]
tags: [paper, causality, causal-graph, scm, do-calculus, dsdag, theory]
status: complete
venue: "Cambridge University Press, 2nd edition"
authors: "Pearl"
year: 2009
relevance: "Referenced in MCAM (Cheng et al. ICCV 2025) — theoretical foundation for DSDAG module in [[methods/multimodal-causal-driving|MCDM Approach 3]]"
---

# Pearl 2009 — Causality: Models, Reasoning, and Inference

The foundational mathematical treatment of causality. Cited in MCAM (ICCV 2025) as the basis for the DSDAG causal reasoning module used in [[methods/multimodal-causal-driving|MCDM Approach 3]].

---

## The Central Distinction: Seeing vs. Doing

Pearl's core argument: **probabilistic/statistical models cannot represent causality**. Correlation encodes what we observe; causality encodes what would happen under intervention. These are fundamentally different questions requiring different mathematics.

```
P(Y | X=x)      ← seeing: what is Y when we observe X=x?
P(Y | do(X=x))  ← doing:  what is Y when we SET X=x by intervention?
```

A barometer reading predicts rain (correlation), but breaking the barometer doesn't cause rain (intervention). Statistical models conflate these; causal models distinguish them.

---

## The Ladder of Causation (Three Rungs)

| Rung | Question | Operation | Example |
|------|----------|-----------|---------|
| 1 — Association | What is? | P(Y\|X) — conditioning | "Pedestrian is looking — will they cross?" |
| 2 — Intervention | What if I do? | P(Y\|do(X)) — surgery | "If I turn the signal red, will they stop?" |
| 3 — Counterfactual | What if it had been? | P(Y_x \| X=x', Y=y') | "Would they have crossed if they'd seen the car?" |

Statistical methods operate only at Rung 1. Causal Bayesian networks reach Rung 2. Functional/structural causal models (SCMs) reach Rung 3.

---

## Core Formalism

### Directed Acyclic Graphs (DAGs)

A DAG G has nodes (variables) and directed edges (direct causal relationships). No cycles — effects cannot cause their own causes.

**Key terminology:**
- **Parents** PA_i: nodes with direct arrows into X_i (immediate causes)
- **Ancestors**: all upstream nodes
- **Descendants**: all downstream nodes
- **d-separation**: a graphical criterion for conditional independence

### Bayesian Networks

A DAG G + conditional probability tables P(X_i | PA_i). The joint distribution factorizes:

```
P(x1,...,xn) = ∏ P(xi | pai)         (Markov factorization)
```

**Parental Markov Condition (Theorem 1.2.7):** Each variable X_i is independent of all its non-descendants given its parents PA_i.

### d-Separation (Definition 1.2.3)

The graphical criterion for reading off conditional independencies. A path p is **blocked** (d-separated) by a set Z if:

1. **Chain** i→m→j or **Fork** i←m→j: middle node m **is** in Z (conditioning blocks the flow)
2. **Collider** i→m←j: middle node m is **not** in Z and no descendant of m is in Z (collider opens when conditioned on!)

If Z d-separates X from Y, then X⊥Y | Z in every distribution compatible with G.

**Collider trap:** conditioning on a common effect of two independent causes makes those causes dependent ("explaining away" / Berkson's paradox). Critical for avoiding spurious correlations in models.

---

## Causal Bayesian Networks

A Bayesian network where each parent-child relationship represents a **stable, autonomous physical mechanism** — one that can be changed by intervention without disturbing other mechanisms. This modularity is the key property.

**Definition 1.3.1 (Causal Bayesian Network):** G is a causal Bayesian network if for any intervention do(X=x), the resulting distribution is:

```
P_x(v) = ∏_{i | Vi ∉ X} P(vi | pai)    (truncated factorization)
```

The intervention "surgically" removes the incoming arrows to X and fixes its value. All other mechanisms remain intact.

**Seeing vs. doing, formally:**
```
Observing X=x:    P(y, x, s) / P(x, s)              ← conditioning
Intervening do(X=x):  delete link X1→X3, set X3=x   ← surgery on graph
```

These give different answers whenever there is a backdoor path from X to Y.

---

## Functional Causal Models (Structural Causal Models — SCMs)

The most general formulation. Each variable is determined by a **structural equation**:

```
xi = fi(PAi, Ui),    i = 1,...,n
```

Where:
- PA_i = direct observed causes (parents in the graph)
- U_i = error/disturbance term — **all unmeasured factors** affecting X_i
- f_i = a deterministic function (the "mechanism")

The U_i are assumed mutually independent (no unobserved confounders between U_i terms) in the simplest (Markovian) case.

**Three query types this enables (Section 1.4.2):**
- **Predictions** (Rung 1): P(Y | X=x) — observe
- **Interventions** (Rung 2): P(Y | do(X=x)) — surgery, delete equation for X
- **Counterfactuals** (Rung 3): P(Y_{x'} | X=x, Y=y) — "what would Y have been had X been x' instead of x, given we observed X=x, Y=y"

Counterfactuals **cannot be answered from purely stochastic (probabilistic) models** — they require the functional form f_i to be specified or inferred.

---

## Key Tools

### do-Calculus (Chapter 3)

Three inference rules for manipulating expressions involving do(·) operators, enabling computation of interventional distributions from observational data under certain graphical conditions.

### Backdoor Criterion (Section 3.3.1)

A set Z satisfies the backdoor criterion for estimating the causal effect of X on Y if:
1. No node in Z is a descendant of X
2. Z blocks all backdoor paths from X to Y (paths with arrows into X)

If Z satisfies backdoor criterion: `P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)`

This is the formal justification for controlling for confounders.

### Frontdoor Criterion (Section 3.3.2)

Alternative when backdoor adjustment fails (no valid adjustment set exists). Uses a mediating variable M between X and Y.

### Stability of Causal Relationships (Section 1.3.2)

Causal relationships are **stable** — invariant to changes in other mechanisms. Probabilistic relationships are **epistemic** (change when we learn new things). This stability is why causal models are preferable for prediction under distribution shift.

---

## Counterfactuals in Depth (Chapter 7)

Counterfactuals require the twin network method or abduction-action-prediction:
1. **Abduction**: use evidence to update beliefs about U variables
2. **Action**: modify the model to reflect the intervention
3. **Prediction**: compute the distribution of the counterfactual outcome

Example: "Would the vehicle have stopped had there been a child instead of a plastic bag?" This is a Rung 3 question — requires knowing which U variables govern the stop decision, then asking what outcome f_stop(plastic bag, U) vs. f_stop(child, U) would produce, fixing U at its abducted value.

---

## Connection to DSDAG

The DSDAG (Cheng et al., MCAM) is a learned approximation of a Pearl-style SCM applied to video:

| Pearl Concept | DSDAG Realization |
|---|---|
| Structural equation x_i = f_i(PA_i, U_i) | Learned neural functions over clip features |
| PA_i (observed parents) | Start state Xs, Environment Z |
| U_i (unmeasured factors) | Hidden danger state **W** |
| do(X=x) intervention | Implicit — modeled via state transitions |
| Counterfactual | "What W would make the same action occur for different reasons?" |

**W is Pearl's U — the error/disturbance term**, representing everything that caused the action beyond the observed parent variables. Pearl's framework makes explicit why W is necessary: without it, two instances with identical (Xs, Z, Y) but different causal origins are **indistinguishable** in a pure probabilistic model. W is what makes the distinction representable.

---

## Custom Causal Graph for ROAD-Waymo (Design Thought)

The DSDAG uses a generic Xs→Y→Xe + Z structure. Pearl's framework suggests a more task-tailored graph could be specified for driving scenarios:

```
Road Context (C) ────────────────────────────→ Action (A)
     │                                              ↑
     ↓                    Intent (I) ───────────────┘
Agent State (S) ──────────→ │
     │                      ↓
Social Context (Sc) ──→  Decision (D) ──→ Observable Behavior (B)
```

Where:
- **C** (road type, signals, traffic density) = Pearl's exogenous/context variable
- **S** (position, velocity, gaze) = observed agent state
- **I** (latent crossing intent) = the hidden variable we want — Pearl's U
- **D** (decision point) = mediating variable
- **B** (crossing/not-crossing) = the observed outcome

**Why this might be better:** Pearl's backdoor criterion tells us that to estimate the causal effect of S on B, we must control for C (road context confounders). The DSDAG doesn't distinguish these roles explicitly — it treats all inputs as feeding into the start state. A more structured graph would let us apply do-calculus to reason about *why* an agent acts, not just *what* they do.

**The practical constraint:** Pearl-style graphs require explicit causal structure to be specified or discovered. DSDAG sidesteps this by learning a neural approximation of the structural equations from data. Whether a hand-designed graph would improve on the learned structure is an empirical question — but Pearl provides the theoretical framework for designing and validating such a graph.

---

## Chapter Map

| Chapter | Content | Relevance |
|---------|---------|-----------|
| 1 | Probability, DAGs, Bayesian networks, d-separation, SCMs, interventions, counterfactuals | Core formalism |
| 2 | Causal discovery from data | Less relevant — we specify structure |
| 3 | do-Calculus, backdoor/frontdoor criteria, identification | Intervention theory |
| 4 | Actions, plans, direct vs. indirect effects | Policy analysis |
| 5 | Structural equation models in social science | SEM connections |
| 6 | Simpson's paradox, confounding | Confounding theory |
| 7 | Counterfactuals — formal theory | Rung 3 reasoning |
| 8–10 | Bounding, imperfect experiments, probability of causation | Advanced topics |

---

## Related

- [[methods/multimodal-causal-driving|MCDM Architecture]] — DSDAG is the neural implementation of a Pearl-style SCM
- [[papers/cheng-2025-mcam|MCAM (Cheng et al. ICCV 2025)]] — cites Pearl as the theoretical basis for DSDAG
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — t-norm constraints operate at Rung 1 (associational); causal head operates at Rung 2
- [[directions/constrained-vlm-reasoning|Approach 3 Overview]]
