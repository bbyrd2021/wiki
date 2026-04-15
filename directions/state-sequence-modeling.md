---
type: direction
title: "State-Sequence Modeling for Crossing Intent"
aliases: ["state sequence modeling", "discrete state sequence", "behavioral state machine"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [direction, jaad, pie, road-plusplus, transformer, sequence-modeling]
status: complete
novelty: true
feasibility: workstation
datasets_required: [jaad, pie, road-plusplus]
---

# State-Sequence Modeling for Crossing Intent

**Direction 5** from SYNTHESIS.md Part 7.

## Motivation

The cross-tabulation shows that **sequence matters**, not just instantaneous state:
- standing → walking → looking is different from always-stationary
- In ROAD++, Stop → Wait2X → XingFmLft is a canonical crossing event pattern detectable early (at Wait2X)

Most papers regress on raw frame features. **Discrete state sequence modeling is underexplored.**

## Approach

1. Represent N-frame window as a sequence of categorical events: (action, look, [cross]) at each frame
2. Map to one of K discrete states (e.g., 4 JAAD states: walking-looking, walking-not-looking, standing-looking, standing-not-looking)
3. Train a Transformer over the sequence of state tokens to classify which trajectory leads to crossing
4. For ROAD++: predict Wait2X → Xing transition (54,928 Wait2X instances provide ample training data)

## Why Novel

Unlike raw feature regression (visual, pose, trajectory), this approach:
- Is explicitly interpretable — the model reasons over human-meaningful behavioral states
- Can incorporate explicit constraint transitions (a standing pedestrian cannot directly become XingFmLft)
- Provides natural uncertainty — multi-step sequences accumulate uncertainty in a principled way

## Datasets

- [[datasets/road-plusplus|ROAD++]] — largest corpus for Wait2X→Xing transition learning
- [[datasets/jaad|JAAD]] — action+look state sequences, decision_point labels
- [[datasets/pie|PIE]] — action+look at critical_point, intention_prob as regression target

## Related

- [[concepts/action-look-crosstab|Action × Look Cross-Tabulation]]
- [[directions/logic-constrained-intent|Logic-Constrained Intent]] (constraints on state transitions)
