---
type: paper
title: "Frozen-DETR: Enhancing DETR with Image Understanding from Frozen Foundation Models"
aliases: ["frozen-detr", "frozen detr"]
authors: "Fu et al."
year: 2024
venue: "NeurIPS"
arxiv: "2410.19635"
created: 2026-05-06
updated: 2026-05-07
sources:
  - "ROAD_Reason/papers/frozen-detr_neurips2024.pdf"
tags: [paper, detection, foundation-model, detr, feature-fusion, frozen-backbone]
status: complete
---

# Frozen-DETR — Enhancing DETR with Image Understanding from Frozen Foundation Models

Fu et al., NeurIPS 2024. [arXiv:2410.19635](https://arxiv.org/abs/2410.19635) | [GitHub](https://github.com/iSEE-Laboratory/Frozen-DETR)

## Key Idea

Uses frozen foundation models (DINOv2, CLIP) as **plug-and-play feature enhancers** alongside a trainable CNN backbone, rather than replacing the backbone. Two fusion mechanisms:

1. **Class token as "image query"**: The CLS token is concatenated with DETR object queries in the decoder, providing global scene context for query decoding.
2. **Patch tokens as extra FPN level**: Patch tokens are reshaped to a 2D feature map and concatenated with CNN multi-scale feature maps, then fused adaptively through the DETR encoder's self-attention layers.

The foundation model runs in parallel with the backbone and is frozen during training. No architecture constraints — detector and foundation model can use different structures and input resolutions.

## Results

- DINO (R50) baseline: 49.0 AP on COCO → **51.9 AP (+2.9)** with one foundation model → **53.8 AP (+4.8)** with two
- +6.6% AP on LVIS (long-tail)
- +8.8% novel AP on open-vocabulary COCO
- Works with CLIP, DINOv2, and MAE as foundation models

## Relevance to ROAD_Reason

**Directly inspired Exp2b's VLM fusion design**, but the actual Frozen-DETR fusion is more sophisticated than what Exp2b implements. Exp2b uses a simple scalar gate per FPN level (`fused = fpn + gate * vlm`), whereas Frozen-DETR:
- Passes patch tokens through the **encoder self-attention** (not a scalar gate)
- Uses the CLS token as an **additional decoder query** (not implemented in Exp2b)

**Exp2c implements Frozen-DETR faithfully** (2026-05-07): CLIP ViT-L/14 patch tokens as 4th encoder scale, 6-layer deformable encoder with VLM token stripping, per-layer CLS injection in the decoder. See [[findings/exp2c-frozen-detr|Exp2c finding page]].

## Related

- [[papers/woo-2018-cbam|CBAM]] — alternative attention-based fusion mechanism
- [[papers/gao-2025-vmcnet|VMCNet]] — similar dual-backbone (CNN + frozen ViT) for detection
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the spatial precision problem Frozen-DETR addresses
- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — faithful implementation on ROAD-Waymo
- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — predecessor with scalar gate fusion
