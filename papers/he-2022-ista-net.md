---
type: paper
title: "ISTA-Net: Spatio-Temporal Attention Graph for Pedestrian Intent Prediction"
authors: "He et al."
year: 2022
venue: "ICRA"
arxiv: "2204.03931"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, graph, transformer, optical-flow, intent-prediction]
status: complete
datasets_used: [pie]
---

# ISTA-Net

**He et al. | ICRA 2022 | arXiv:2204.03931**

Spatio-temporal attention graph for pedestrian intent prediction on PIE.

## Contribution

- Builds spatio-temporal graph over pedestrian bboxes, optical flow, and ego OBD data
- GAT (Graph Attention Network) for spatial context
- Transformer for temporal long-range dependencies

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **89.51%** |
| F1 | **88.32%** |

Outperforms PCPA by ~3 percentage points on PIE.

## Related

- [[methods/ista-net|ISTA-Net Method]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
