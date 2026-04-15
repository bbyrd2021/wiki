---
type: method
title: "IntentFormer"
aliases: ["IntentFormer"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [method, jaad, pie, transformer, multimodal, co-learning]
status: complete
datasets_evaluated: [jaad, pie]
best_accuracy_pie: 0.93
best_accuracy_jaad: 0.92
---

# IntentFormer

Multimodal co-learning Transformer for pedestrian crossing intent. Highest published accuracy on PIE and JAAD (2024).

## Architecture

Three input streams fused via co-learning Transformer:
1. RGB pedestrian crops (visual appearance)
2. Scene segmentation maps (scene context)
3. Trajectory coordinates (motion history)

Co-learning: modalities align at each attention layer, not just at final fusion.

## Results

| Dataset | Accuracy |
|---------|---------|
| PIE | **93%** |
| JAAD | **92%** |

## Related

- [[papers/rasouli-2024-intentformer|IntentFormer Paper]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
