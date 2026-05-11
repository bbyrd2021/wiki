---
type: paper
title: "CBAM: Convolutional Block Attention Module"
aliases: ["cbam", "convolutional block attention module"]
authors: "Woo et al."
year: 2018
venue: "ECCV"
arxiv: "1807.06521"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/cbam_eccv2018.pdf"
tags: [paper, attention, channel-attention, spatial-attention, detection, feature-fusion]
status: complete
---

# CBAM — Convolutional Block Attention Module

Woo, Park, Lee, Kweon. ECCV 2018. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521) | [GitHub](https://github.com/Jongchan/attention-module)

## Key Idea

A lightweight attention module with two sequential sub-modules:

1. **Channel Attention**: "what" to attend to. Global average pool + global max pool → shared MLP → sigmoid → per-channel weights [B, C, 1, 1].
2. **Spatial Attention**: "where" to attend. Channel-wise avg pool + max pool → concat → 7×7 conv → sigmoid → spatial mask [B, 1, H, W].

Applied sequentially: input → channel attention → spatial attention → refined output. Can be inserted at every convolutional block with negligible overhead (~10K params per instance).

## Results

- +1-2% mAP on VOC 2007 and MS COCO when added to Faster R-CNN with ResNet backbone
- Consistent improvement across ImageNet-1K, COCO, VOC 2007
- Spatial attention component specifically improves detection at high IoU thresholds (localization quality)
- Grad-CAM visualization shows CBAM-enhanced networks focus more precisely on target objects

## Relevance to ROAD_Reason

**Recommended as the first fusion upgrade for Exp2b.** The current scalar gate in `vlm_fusion.py` treats all 256 channels and all spatial locations identically. CBAM provides both:

- **Channel selectivity**: suppress VLM channels carrying spatial noise, amplify those with useful semantics
- **Spatial selectivity**: preserve FPN's precise edge/boundary features while injecting VLM semantics in object interiors

Implementation: replace the 3 scalar gates in `VLMSemanticFusion` with 3 CBAM modules (~10K params each). The spatial attention directly addresses the localization precision loss observed in Exp2b (agent f-mAP 1.71% vs baseline 17.76%).

## Related

- [[papers/gao-2025-vmcnet|VMCNet]] — channel-wise modulation (FiLM) alternative
- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — encoder-level fusion alternative
- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — target model for this upgrade
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the spatial precision problem CBAM addresses
