---
type: paper
title: "Context-Aware Multi-task Learning for Pedestrian Intent and Trajectory Prediction"
authors: "(multiple authors)"
year: 2025
venue: "Transportation Research Part C"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, multitask, trajectory, context]
status: complete
datasets_used: [jaad, pie]
---

# Context-Aware Multi-task Learning for Pedestrian Intent and Trajectory

**Transportation Research Part C 2025**

Joint training on JAAD+PIE for binary intent classification + future trajectory regression with scene context encoding.

## Contribution

- Shared GRU encoder with multi-task heads (intent + trajectory)
- Scene context: pedestrian crossing zones, traffic lights from JAAD's traffic annotation layer
- Improves intent F1 by ~3% over single-task baselines on JAAD

## Key Contribution

First paper to explicitly encode JAAD's traffic layer (scene-level annotations) as auxiliary input for intent prediction. Traffic light state and crossing zone type provide context that resolves ambiguous pedestrian behavior.

## Related

- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
