---
type: paper
title: "ROAD-R: The Autonomous Driving Dataset with Logical Requirements"
authors: "Marconato et al."
year: 2022
venue: "Machine Learning 2023 / arXiv 2022"
arxiv: "2210.01597"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, neuro-symbolic, t-norm, logic-constraints]
status: complete
datasets_used: [road-plusplus]
---

# ROAD-R: The Autonomous Driving Dataset with Logical Requirements

**Marconato et al. | Machine Learning 2023 / arXiv:2210.01597**

Extends ROAD with 243 propositional-logic requirements and proposes requirement-compliant training via t-norm constraint losses.

## Contribution

- Adds 243 logical requirements to ROAD dataset (e.g., "an agent cannot be both Car and Pedestrian"; "a Ped-Xing must be at a junction or crossing zone")
- Proposes **requirement-compliant training**: adds loss term penalizing predictions that violate known constraints
- Shows improved requirement compliance without significant accuracy drop
- Enables verifiable AV decision-making — predictions are provably consistent with domain knowledge

## T-norm Constraint Loss

```python
# Łukasiewicz t-norm: "LarVeh cannot perform Xing"
violation = max(0, p(LarVeh) + p(Xing) - 1)
# Loss term: detection_loss + λ * sum(violations)
```

## Significance

Directly applicable to [[datasets/road-plusplus|ROAD++]] — the same constraint set applies. This is the methodological foundation for [[projects/road-reason|ROAD_Reason]] Approach 2 (replication) and Approach 3 (extension with causal reasoning).

## Related

- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] | [[datasets/road-plusplus|ROAD++]]
- [[directions/constrained-vlm-reasoning|Constrained VLM Reasoning]]
- [[directions/logic-constrained-intent|Logic-Constrained Intent]]
