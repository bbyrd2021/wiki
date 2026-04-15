---
type: paper
title: "Pedestrian Intention via Vision-Language Foundation Models"
authors: "(multiple authors)"
year: 2025
venue: "arXiv"
arxiv: "2507.04141"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, vlm, clip, language, intent-prediction]
status: complete
datasets_used: [pie]
---

# Pedestrian Intention via Vision-Language Foundation Models

**arXiv:2507.04141 | 2025**

Uses CLIP/LLM-based scene description as additional context for PIE intent prediction. State-of-the-art on PIE as of mid-2025.

## Contribution

- Generates natural language descriptions of the scene: "Is a crosswalk present? What is the traffic light state? Is the pedestrian near the kerb?"
- Fuses language descriptions with visual bbox features via cross-attention
- SOTA on PIE at time of publication

## Significance

Validates the VLM approach for pedestrian intent prediction. Directly related to [[projects/road-reason|ROAD_Reason]] Approach 3, which applies the same paradigm (VLM + compositional scene understanding) to ROAD++ with neuro-symbolic constraints.

## Related

- [[datasets/pie|PIE]] | [[directions/constrained-vlm-reasoning|Constrained VLM Reasoning]]
- [[projects/road-reason|ROAD_Reason]]
