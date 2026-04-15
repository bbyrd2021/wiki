---
type: concept
title: "Compositional Labels (ROAD++)"
aliases: ["compositional labels", "duplex", "triplet", "event labels"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/ROAD_plusplus/ROAD_plusplus_summary.md"
tags: [concept, road-plusplus, compositional-labels, structured-prediction]
status: complete
---

# Compositional Labels

ROAD++ uses a five-level label hierarchy where each bounding box receives simultaneous labels at multiple abstraction levels. Higher levels are compositional — derived from combinations of lower levels.

## Five Levels

| Level | Classes | Multi-label? | Example |
|-------|---------|-------------|---------|
| Agent | 10 | No (one per box) | `Ped` |
| Action | 22 | Yes (multiple) | `Wait2X` |
| Location | 16 | Yes (multiple) | `RhtPav` |
| **Duplex** | **49 task** | Derived | `Ped-Wait2X` |
| **Triplet** | **86 task** | Derived | `Ped-Wait2X-RhtPav` |

Duplexes and triplets are **not independently annotated** — they are derived from the agent+action+location combination.

## Validity Constraints

Not all combinations are valid:
- 49 valid duplexes from 220 possible agent+action combinations
- 86 valid triplets from 3,520 possible combinations

Stored in `duplex_childs` and `triplet_childs` in `road_waymo_trainval_v1.0.json`. These validity mappings are the basis for [[concepts/neuro-symbolic-constraints|neuro-symbolic constraint losses]].

## Concrete Example

```
Agent: Car (id=1)    →  action: MovAway (id=3)    →  loc: OutgoLane (id=1)
Duplex: Car-MovAway (id=9)
Triplet: Car-MovAway-OutgoLane (id=109)
```

## Crossing-Related Labels

| Action | Instances | Semantics |
|--------|----------|-----------|
| `Wait2X` | 54,928 | At kerb, waiting to cross |
| `XingFmLft` | 72,395 | Mid-crossing from left |
| `XingFmRht` | 75,591 | Mid-crossing from right |
| `Xing` | 41,659 | Crossing (unspecified direction) |

These provide richer crossing semantics than JAAD/PIE binary labels — direction of approach disambiguates conflict geometry with the AV.

## Why Compositional Labels Matter

- A model trained on triplet labels learns `Ped-Wait2X-RhtPav` as a single semantic token — "pedestrian waiting on right pavement" — rather than requiring three independent predictions to compose
- Constraint violations become explicitly detectable (a model predicting `LarVeh-Xing` violates the duplex constraint that large vehicles cannot cross pedestrian crossings)
- Enables situation-aware scene understanding rather than object-by-object labeling

## Related

- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] | [[datasets/road-plusplus|ROAD++]]
- [[projects/road-reason|ROAD_Reason]] | [[directions/logic-constrained-intent|Logic-Constrained Intent]]
