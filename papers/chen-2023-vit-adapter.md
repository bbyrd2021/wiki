---
type: paper
title: "Vision Transformer Adapter for Dense Predictions"
aliases: ["vit-adapter", "vit adapter"]
authors: "Chen et al."
year: 2023
venue: "ICLR (Spotlight)"
arxiv: "2205.08534"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/vit-adapter_iclr2023.pdf"
tags: [paper, detection, segmentation, vit, adapter, spatial-prior, cross-attention, dense-prediction]
status: complete
---

# ViT-Adapter — Vision Transformer Adapter for Dense Predictions

Chen, Duan, Wang, He, Lu, Dai, Qiao. ICLR 2023 (Spotlight). [arXiv:2205.08534](https://arxiv.org/abs/2205.08534) | [GitHub](https://github.com/czczup/ViT-Adapter)

## Key Idea

A **pre-training-free adapter** that introduces vision-specific inductive biases (spatial priors) into a plain ViT without modifying its architecture. Three components:

1. **Spatial Prior Module**: A lightweight CNN generates multi-scale spatial feature maps from the input image (capturing local structure the ViT misses).
2. **Spatial Feature Injector**: Cross-attention injects CNN spatial features into ViT patch tokens at multiple stages — ViT tokens attend to CNN features to absorb spatial precision.
3. **Multi-Scale Feature Extractor**: Reconstructs a feature pyramid from the adapted ViT for dense prediction heads.

The adapter preserves the ViT's flexibility for multi-modal pre-training while closing the performance gap with vision-specific transformers (Swin, PVT).

## Results

- 60.9 box AP / 53.0 mask AP on COCO test-dev (ViT-Adapter-L with multi-modal pre-training)
- 49.6 box AP on COCO val with ViT-Adapter-B (ImageNet-1K only) — outperforms Swin-B by 1.0 AP
- 62.1 mIoU on ADE20K semantic segmentation
- Used by DINOv2 for dense prediction tasks

## Relevance to ROAD_Reason

**Most architecturally relevant for the VLM spatial precision problem.** ViT-Adapter solves exactly the gap Exp2b faces: a plain ViT (Qwen) lacks multi-scale spatial precision needed for detection.

The adapter concept can be applied in reverse for Exp2b:
- **Current**: CNN spatial features are the base; VLM features are added via scalar gate
- **ViT-Adapter style**: Use cross-attention where FPN features are queries and VLM patch tokens are keys/values. Each FPN spatial position selectively attends to relevant VLM tokens.

This is the most principled fusion approach but also the most expensive (cross-attention at each FPN level). Best suited for P4/P5 levels where spatial dimensions are smaller; use [[papers/woo-2018-cbam|CBAM]] for P3.

## Related

- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — simpler encoder-level fusion alternative
- [[papers/xia-2024-vit-comer|ViT-CoMer]] — bidirectional CNN-ViT interaction (builds on ViT-Adapter ideas)
- [[papers/woo-2018-cbam|CBAM]] — lightweight attention alternative for large feature maps
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the problem ViT-Adapter addresses
