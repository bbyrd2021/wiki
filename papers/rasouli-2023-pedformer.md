---
type: paper
title: "PedFormer: Pedestrian Behavior Prediction via Cross-Modal Attention Modulation and Gated Multitask Learning"
authors: "Rasouli & Kotseruba"
year: 2023
venue: "ICRA"
arxiv: "2210.07886"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, transformer, multimodal, multitask, intent-prediction]
status: complete
datasets_used: [jaad, pie]
---

# PedFormer

**Rasouli & Kotseruba | ICRA 2023 | arXiv:2210.07886**

State-of-the-art pedestrian behavior prediction model on JAAD and PIE (at time of publication). Jointly predicts trajectory, crossing action, and intention via cross-modal Transformer.

## Contribution

- Inputs: bounding box sequences, ego OBD speed, pose keypoints, context image crops
- Cross-modal Transformer attention fuses all modalities
- **Gated multitask head**: selects which prediction heads to activate per sample
- Joint prediction: trajectory regression + crossing action + crossing intent

## Results

- ~88% balanced accuracy on PIE intent task
- SOTA on JAAD and PIE at publication (ICRA 2023)

## Significance

PedFormer is the de facto strong baseline for JAAD/PIE comparison. [[projects/efficient-pie|EfficientPIE]] is described as distilling a deployment-efficient variant. GTransPDM (2025) surpasses it.

## Related

- [[methods/pedformer|PedFormer Method]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[papers/rasouli-2024-encore|ENCORE]] (follow-up work)
- [[comparisons/model-comparison|Model Comparison]]
