---
type: concept
title: "Neuro-Symbolic Constraints (T-norm Loss)"
aliases: ["neuro-symbolic", "t-norm", "constraint loss", "ROAD-R"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "ROAD_Reason/docs/APPROACHES.md"
  - "ROAD_Reason/docs/CLAUDE.md"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [concept, neuro-symbolic, t-norm, road-plusplus, logic-constraints]
status: complete
---

# Neuro-Symbolic Constraints (T-norm Loss)

A training approach that enforces known logical requirements on model predictions by adding a constraint violation penalty to the standard loss. Makes predictions verifiably consistent with domain knowledge.

## Core Idea

Logical requirements (e.g., "a large vehicle cannot cross a pedestrian crossing") are encoded as propositional constraints. These constraints are made differentiable using **t-norms** — algebraic operations that map Boolean logic to [0,1].

## Łukasiewicz T-norm Example

Constraint: "LarVeh cannot perform Xing"

```python
violation = max(0, p(LarVeh) + p(Xing) - 1)
```

- If both probabilities are high → large violation → large penalty
- If either is low → no violation → no penalty
- Differentiable everywhere except at the hinge

**Training loss:** `L_total = L_detection + λ · L_constraints`

## ROAD++ Constraint Set

- **49 valid duplexes** (agent+action combinations): encoded as "if agent=A then action=B is invalid" constraints
- **86 valid triplets** (agent+action+location): encoded similarly
- ROAD-R (Marconato et al. 2022) extends this to **243 propositional-logic requirements**

## Sources

- **ROAD-R paper** (Marconato et al., arXiv:2210.01597): Original formalization of constraint-compliant training on ROAD dataset
- **ROAD-Waymo paper** (Salmank et al., arXiv:2411.01683): Applies t-norm constraint loss as a neuro-symbolic baseline on ROAD++
- **ROAD_Reason Approach 2**: Replication of this neuro-symbolic baseline
- **ROAD_Reason Approach 3**: Primary novel contribution that extends t-norm loss with causal reasoning (DSDAG)

## Application to JAAD/PIE

Direct application is limited (JAAD/PIE have no compositional label constraints). However, the methodology applies:
- JAAD action × look combinations have implicit constraints (certain combinations are logically impossible)
- See [[directions/logic-constrained-intent|Logic-Constrained Intent]] for proposed extension

## Related

- [[concepts/compositional-labels|Compositional Labels]] | [[datasets/road-plusplus|ROAD++]]
- [[directions/constrained-vlm-reasoning|Constrained VLM Reasoning]]
- [[directions/logic-constrained-intent|Logic-Constrained Intent (on JAAD/PIE)]]
- [[projects/road-reason|ROAD_Reason]]
