---
type: paper
title: "SSL-Lanes: Self-Supervised Learning for Motion Forecasting"
authors: "Bhattacharyya et al."
year: 2023
venue: "CoRL"
arxiv: "2206.14116"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, self-supervised, trajectory, pretraining]
status: complete
datasets_used: [pie]
---

# SSL-Lanes

**Bhattacharyya et al. | CoRL 2023 | arXiv:2206.14116**

Self-supervised pretraining on PIE OBD + bbox sequences (masked trajectory completion, speed regression), improving downstream intent prediction.

## Contribution

- SSL pretext tasks: masked trajectory completion + speed regression on PIE sequences
- Fine-tuning on PIE intent labels: +4% accuracy over supervised-from-scratch baseline
- Demonstrates that PIE's rich OBD sensor data (speed, GPS, accelerometer) provides a powerful SSL signal

## Significance

Shows that PIE's unique OBD sensor modality (unavailable in JAAD or ROAD++) enables self-supervised pretraining approaches not possible with other datasets.

## Related

- [[datasets/pie|PIE]] | [[concepts/intention-probability|Intention Probability]]
