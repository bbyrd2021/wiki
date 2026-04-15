---
type: paper
title: "Pedestrian Crossing Prediction in Obstructed Scenarios Using Graph-Based Feature Reconstruction"
authors: "(multiple authors)"
year: 2025
venue: "Engineering Applications of Artificial Intelligence"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, occlusion, graph-autoencoder]
status: complete
datasets_used: [jaad, pie]
---

# Occlusion-Robust Crossing Prediction via Graph Feature Reconstruction

**Engineering Applications of Artificial Intelligence 2025**

Addresses partial occlusion in JAAD/PIE using a graph autoencoder to reconstruct missing features.

## Contribution

- **Graph autoencoder** reconstructs missing pedestrian features when partially occluded
- Multi-scale self-attention encoder for crossing prediction on reconstructed features
- +6% accuracy on JAAD, +3% on PIE over SOTA under partial-occlusion scenarios

## Significance

First paper to directly use JAAD's `occlusion` annotation layer as a training signal for occlusion-specific evaluation. Establishes that occlusion-aware models can significantly outperform generic models in realistic deployment conditions.

## Related

- [[concepts/occlusion|Occlusion]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
