---
type: paper
title: "Pedestrian Group Crossing Intention Prediction Integrating Spatiotemporal Features"
authors: "(multiple authors)"
year: 2025
venue: "Scientific Reports"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, group-behavior, graph, pose]
status: complete
datasets_used: [jaad, pie]
---

# Pedestrian Group Crossing Intention

**Scientific Reports 2025**

Extends JAAD/PIE evaluation to group crossing by modeling multiple pedestrians as a graph.

## Contribution

- Models multiple pedestrians as a spatial graph with inter-pedestrian edges
- Combines pose keypoints, 2D trajectories, and group membership labels
- Outperforms individual-pedestrian baselines on JAADbeh and JAADall subsets

## Key Insight

Pedestrian groups exhibit social force dynamics — one person crossing increases the probability that adjacent pedestrians also cross. Modeling group context provides information unavailable to single-pedestrian models.

## Related

- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
