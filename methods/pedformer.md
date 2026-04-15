---
type: method
title: "PedFormer"
aliases: ["PedFormer"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [method, jaad, pie, transformer, multimodal, multitask]
status: complete
datasets_evaluated: [jaad, pie]
---

# PedFormer

Cross-modal attention Transformer for pedestrian behavior prediction. State-of-the-art on JAAD and PIE at publication (ICRA 2023).

## Inputs

- Bounding box sequences (trajectory)
- Ego-vehicle OBD speed (PIE only)
- Pose keypoints (ViTPose)
- Context image crops (pedestrian + surrounding scene)

## Architecture

1. Per-modality encoders (1D CNN or Transformer)
2. Cross-modal attention: each modality attends to all others
3. Gated multitask head: selects active prediction heads per sample

## Outputs (joint)

- Future trajectory regression (bounding box sequence)
- Crossing action prediction (will they cross?)
- Intent classification (binary crossing intent)

## Results

~88% balanced accuracy on PIE intent task. SOTA on JAAD at publication.

## Related

- [[papers/rasouli-2023-pedformer|PedFormer Paper]] | [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[comparisons/model-comparison|Model Comparison]]
