---
type: method
title: "GTransPDM"
aliases: ["GTransPDM", "Graph Transformer PDM"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [method, jaad, pie, graph-transformer, pose, efficient]
status: complete
datasets_evaluated: [jaad, pie]
best_accuracy_pie: 0.92
best_accuracy_jaad: 0.87
---

# GTransPDM

Graph-embedded Transformer with Positional Decoupling for pedestrian crossing intent prediction. 0.05ms inference (2025 SOTA).

## Key Innovation

**Positional decoupling module**: decomposes lateral (depth-cue) pedestrian motion from the image plane perspective. Crossing pedestrians move across the camera frame in a way that conflates lateral motion with depth changes — decoupling these signals improves the model's understanding of approach direction.

## Inputs

- Pose keypoints (skeleton graph)
- Bounding box sequences
- Ego-vehicle speed

## Results

| Dataset | Accuracy | Inference |
|---------|---------|-----------|
| PIE | **92%** | **0.05ms** |
| JAAD | **87%** | 0.05ms |

## Related

- [[papers/xu-2025-gtranspdm|GTransPDM Paper]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
