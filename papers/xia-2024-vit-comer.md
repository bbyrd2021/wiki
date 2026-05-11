---
type: paper
title: "ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction for Dense Predictions"
aliases: ["vit-comer", "vit comer"]
authors: "Xia et al."
year: 2024
venue: "CVPR (Highlight)"
arxiv: "2403.07392"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/vit-comer_cvpr2024.pdf"
tags: [paper, detection, segmentation, cnn-vit-fusion, bidirectional-fusion, multi-scale, dense-prediction]
status: complete
---

# ViT-CoMer — CNN-Transformer Multi-scale Feature Interaction

Xia, Wang, Lv, Hao, Shi (Baidu). CVPR 2024 (Highlight). [arXiv:2403.07392](https://arxiv.org/abs/2403.07392) | [GitHub](https://github.com/Traffic-X/ViT-CoMer)

## Key Idea

Two-branch backbone: plain ViT + CNN with two novel modules:

1. **MRFP (Multi-Receptive Field Feature Pyramid)**: Parallel dilated convolutions on CNN features at multiple receptive fields, providing richer multi-scale spatial features than standard FPN.
2. **CTI (CNN-Transformer Bidirectional Fusion Interaction)**: Cross-attention in **both directions** — CNN features attend to ViT tokens AND ViT tokens attend to CNN features — at multiple stages during feature extraction.

The bidirectional interaction means the ViT also benefits from CNN spatial cues, not just the other way around. The adapted backbone paradigm allows loading open-source pre-trained ViT weights directly.

## Results

- **64.3% AP on COCO val2017** (ViT-CoMer-L with advanced pre-training) — best record without extra training data (Objects365)
- 62.1% mIoU on ADE20K val
- Outperforms ViT-Adapter, Swin, and other vision-specific transformers under fair comparison

## Relevance to ROAD_Reason

The full ViT-CoMer architecture is overkill for Exp2b (requires a trainable dual-branch system, not a frozen VLM). However, two ideas are worth borrowing:

1. **MRFP concept**: Apply parallel dilated convolutions to VLM features before fusion, giving them multi-scale receptive fields that better match different FPN levels. Currently VLM features are just bilinearly interpolated — they lack multi-scale structure.

2. **Bidirectional fusion**: In principle, even with a frozen Qwen ViT, you could apply cross-attention from ViT tokens → CNN features (as a form of feature selection) during the FPN fusion stage. Only the attention weights would be trainable.

## Related

- [[papers/chen-2023-vit-adapter|ViT-Adapter]] — predecessor using CNN spatial priors injected into ViT
- [[papers/gao-2025-vmcnet|VMCNet]] — inspired by ViT-CoMer; applies modulation for OVOD
- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — simpler foundation model fusion paradigm
