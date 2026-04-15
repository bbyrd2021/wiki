---
type: paper
title: "EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention"
authors: "(IJCAI authors)"
year: 2025
venue: "IJCAI"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md", "EfficientPIE/README.md"]
tags: [paper, jaad, pie, efficientnet, real-time, efficient]
status: complete
datasets_used: [jaad, pie]
---

# EfficientPIE (IJCAI 2025)

**IJCAI 2025**

Targets deployment efficiency for pedestrian crossing intent prediction. Achieves competitive accuracy at real-time speed on embedded hardware.

## Contribution

- Lightweight EfficientNet-inspired backbone for single-frame intent prediction
- Sub-millisecond inference (0.21ms), 0.69M parameters
- 0.920 accuracy / 0.917 AUC on PIE test set
- 0.890 accuracy on JAAD — competitive with larger models

## Significance

This paper is the direct upstream of [[projects/efficient-pie|SparseTemporalPIE]] (the local research project). SparseTemporalPIE v3 extends it to multi-frame cross-attention, improving AUC to 0.947 (+0.030) on PIE.

## Extended By

- [[projects/efficient-pie|SparseTemporalPIE]] — multi-frame cross-attention extension (Brandon Byrd, xDI Lab, NC A&T)

## Related

- [[methods/sparse-temporal-pie|SparseTemporalPIE Method]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
