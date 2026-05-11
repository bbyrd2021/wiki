---
type: paper
title: "VMCNet: Modulating CNN Features with Pre-Trained ViT Representations for Open-Vocabulary Object Detection"
aliases: ["vmcnet", "vmc-net", "vit-modulated cnn"]
authors: "Gao et al."
year: 2025
venue: "arXiv"
arxiv: "2501.16981"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/vmcnet_2025.pdf"
tags: [paper, detection, open-vocabulary, cnn-vit-fusion, feature-modulation, frozen-backbone]
status: complete
---

# VMCNet — Modulating CNN Features with Pre-Trained ViT Representations

Gao et al., arXiv 2025. [arXiv:2501.16981](https://arxiv.org/abs/2501.16981)

## Key Idea

Two-branch backbone for open-vocabulary detection: a **trainable multi-scale CNN** + a **frozen CLIP ViT**. The VMC (ViT-Feature-Modulated Multi-Scale Convolutional Network) module fuses features from both branches via **modulation** rather than simple addition.

The CNN is optimized with detection labels (bounding boxes) while the frozen ViT retains VLM generalization. Multi-scale CNN features are modulated with adapted ViT features — the ViT representations condition how CNN features are processed, similar to FiLM-style affine transforms.

Inspired by [[papers/xia-2024-vit-comer|ViT-CoMer]]'s dual-branch design but focused on open-vocabulary detection with a frozen VLM.

## Results

- OV-COCO: 44.3 AP50_novel (ViT-B/16) / 48.5 AP50_novel (ViT-L/14)
- OV-LVIS: 27.8 mAP (ViT-B/16) / 38.4 mAP (ViT-L/14)
- Outperforms state-of-the-art on both benchmarks

## Relevance to ROAD_Reason

**Most directly comparable to Exp2b's architecture** — trainable CNN (EfficientNet+FPN) + frozen VLM (Qwen ViT). The key difference is the fusion mechanism:

| Aspect | Exp2b (current) | VMCNet |
|--------|----------------|--------|
| Fusion | `fpn + scalar_gate * vlm` | Modulation: `gamma(vlm) * cnn + beta(vlm)` |
| Gate granularity | 1 scalar per FPN level | Per-channel (256 independent weights) |
| Spatial selectivity | None | Via multi-scale CNN features |

Adopting VMCNet-style **per-channel modulation** (FiLM conditioning) is a straightforward upgrade from the current scalar gate — predicting per-channel gamma and beta from VLM features instead of a single scalar.

## Related

- [[papers/perez-2018-film|FiLM]] — the underlying conditioning mechanism VMCNet builds on
- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — alternative approach: encoder-level fusion
- [[papers/xia-2024-vit-comer|ViT-CoMer]] — bidirectional CNN-ViT fusion inspiration
- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — our dual-backbone model
