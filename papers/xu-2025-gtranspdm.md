---
type: paper
title: "GTransPDM: Graph-embedded Transformer with Positional Decoupling for Pedestrian Crossing Intention"
authors: "Xu et al."
year: 2025
venue: "IEEE Signal Processing Letters"
arxiv: "2409.20223"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, graph-transformer, pose, intent-prediction]
status: complete
datasets_used: [jaad, pie]
---

# GTransPDM

**Xu et al. | IEEE Signal Processing Letters 2025 | arXiv:2409.20223**

Graph-embedded Transformer with positional decoupling for pedestrian crossing intention prediction on JAAD and PIE.

## Contribution

- **Positional decoupling module**: separates lateral (depth-cue) pedestrian motion from the image plane perspective — handles the challenge that crossing pedestrians move across the camera frame
- Graph Transformer over human pose skeletons
- Inputs: pose keypoints, bounding box sequences, ego-vehicle speed
- **Inference: 0.05ms** — extremely fast, deployment-ready

## Results

| Dataset | Accuracy |
|---------|---------|
| PIE | **92%** |
| JAAD | **87%** |

## Related

- [[methods/gtranspdm|GTransPDM Method]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
