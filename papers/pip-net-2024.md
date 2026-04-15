---
type: paper
title: "PIP-Net: Pedestrian Intention Prediction in the Wild"
authors: "(multiple authors)"
year: 2024
venue: "arXiv"
arxiv: "2402.12810"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, domain-shift, domain-adaptation]
status: complete
datasets_used: [pie]
---

# PIP-Net: Pedestrian Intention Prediction in the Wild

**arXiv:2402.12810 | 2024**

Evaluates whether PIE-trained intent models generalize to different cameras and geographies ("in the wild"), and proposes domain adaptation strategies.

## Contribution

- Tests PIE-trained models under distribution shift to other cameras and geographies
- Proposes domain adaptation using unlabeled target-domain footage
- Highlights that PIE (Toronto-only) has limited generalization out of the box

## Significance

Directly relevant to [[directions/cross-dataset-generalization|Cross-Dataset Generalization]] direction. Demonstrates empirically that dataset-specific overfitting is a real problem for PIE models.

## Related

- [[datasets/pie|PIE]] | [[concepts/domain-shift|Domain Shift]]
- [[directions/cross-dataset-generalization|Cross-Dataset Generalization]]
