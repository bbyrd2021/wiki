---
type: paper
title: "Occlusion-Aware Diffusion Model for Pedestrian Intention Prediction"
authors: "(multiple authors)"
year: 2024
venue: "arXiv"
arxiv: "2511.00858"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, occlusion, diffusion-model]
status: complete
datasets_used: [jaad, pie]
---

# Occlusion-Aware Diffusion for Pedestrian Intent

**arXiv:2511.00858 | 2024**

Models occluded pedestrian motion using a diffusion model that reconstructs plausible trajectories under occlusion, feeding reconstructed paths to an intent predictor.

## Contribution

- Diffusion model generates plausible trajectory distributions conditioned on observed (partially occluded) keyframes
- Reconstructed trajectories fed to intent predictor
- +4–5% improvement on EO5 (ego-occluded) scenario subset of JAAD/PIE

## Approach

Treats occlusion as a missing data problem: the diffusion process samples from the posterior over possible pedestrian positions given the observed frames, then marginalizes over these samples for intent prediction.

## Related

- [[concepts/occlusion|Occlusion]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[papers/occlusion-graph-2025|Graph-Based Occlusion Recovery]]
