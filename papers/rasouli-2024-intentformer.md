---
type: paper
title: "IntentFormer: Predicting Pedestrian Intentions with Multimodal Co-learning"
authors: "Rasouli et al."
year: 2024
venue: "Pattern Recognition"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, jaad, pie, transformer, multimodal, intent-prediction]
status: complete
datasets_used: [jaad, pie]
---

# IntentFormer

**Rasouli et al. | Pattern Recognition 2024**

Multimodal co-learning Transformer for pedestrian crossing intention prediction on JAAD and PIE.

## Contribution

- Fuses RGB pedestrian crops, scene segmentation maps, and trajectory coordinates
- Co-learning Transformer aligns modalities at each attention step (not just at final fusion)
- **93% accuracy on PIE, 92% on JAAD** — among the highest published results

## Architecture

Three input streams → shared Transformer attention → co-learning alignment at each layer → crossing intent binary output

## Related

- [[methods/intentformer|IntentFormer Method]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
