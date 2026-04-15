---
type: method
title: "ISTA-Net"
aliases: ["ISTA-Net"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [method, pie, graph, transformer, optical-flow]
status: complete
datasets_evaluated: [pie]
best_accuracy_pie: 0.8951
---

# ISTA-Net

Spatio-Temporal Attention Graph for pedestrian intent prediction on PIE (ICRA 2022).

## Architecture

- Spatio-temporal graph over pedestrian bboxes, optical flow, and ego OBD
- Graph Attention Network (GAT) for spatial context
- Transformer for temporal long-range dependencies

## Results (PIE)

| Metric | Score |
|--------|-------|
| Accuracy | **89.51%** |
| F1 | **88.32%** |

~3pp improvement over PCPA on PIE.

## Related

- [[papers/he-2022-ista-net|ISTA-Net Paper]] | [[datasets/pie|PIE]]
