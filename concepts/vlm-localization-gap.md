---
type: concept
title: "VLM Localization Gap — Why Frozen ViT Features Fail at Detection"
aliases: ["vlm localization", "vit detection gap", "single-scale detection"]
created: 2026-04-24
updated: 2026-04-24
sources:
  - "ROAD_Reason/experiments/exp2_detr_qwen/eval_baseline_compat.py"
tags: [concept, detection, vlm, vit, multi-scale, fpn, detr, localization]
status: complete
---

# VLM Localization Gap — Why Frozen ViT Features Fail at Detection

Vision-Language Models like Qwen2.5-VL achieve strong semantic understanding but produce features that are poorly suited for precise object detection. This page documents why, and what the literature says about fixing it.

---

## The Core Problem: Single-Scale Features

Plain ViT architectures produce a **single spatial resolution** — typically 1/16 of input size. Qwen2.5-VL's ViT outputs a 16×16 grid for 448×448 input (or equivalent via dynamic resolution). This means:

1. **No fine-grained spatial detail** — every patch token covers a 28×28 pixel region. Small objects (pedestrians at distance, traffic lights) are represented by 1-2 tokens at best.
2. **No multi-scale representation** — unlike CNN backbones (ResNet, EfficientNet) that naturally produce feature maps at 1/4, 1/8, 1/16, 1/32 scales, ViT has one scale. Detection requires matching anchor/query resolution to object size.
3. **Semantic over spatial** — ViT features are optimized for global understanding (image captioning, VQA), not spatial precision. The self-attention mechanism spreads information globally, diluting local boundary information.

### Evidence from Exp2

Our Exp2 architecture feeds Qwen2.5-VL's flat 16×16 tokens directly into a DETR decoder:

```
Qwen ViT (16×16, 3584-dim) → Linear(256) → flat DETR decoder (100 queries)
```

Results at IoU=0.5:
- **Recall is healthy**: agent mR=26.0% — the model *finds* agents at coarse level
- **Precision is terrible**: agent f-mAP=0.63% — boxes don't overlap GT by 50%
- **Exp1b FCOS was better**: agent f-mAP=3.2% with the same ViT features but dense per-token prediction

The recall/precision disconnect proves the semantic features carry detection signal, but the spatial resolution can't produce precise boxes.

---

## Key Papers

### ViTDet (Li et al., Meta, ECCV 2022)

Showed that plain ViT needs a **simple feature pyramid** to be competitive at detection. Adding a lightweight FPN to ViT-L gave **+3.2 box AP on COCO**. The pyramid is built by upsampling/downsampling the single-scale ViT output to create 1/4, 1/8, 1/16, 1/32 feature maps. This is cheap (a few conv layers) but critical.

### ViT-CoMer (CVPR 2024)

Bidirectional CNN-Transformer fusion at multiple scales. Injects "spatial pyramid multi-receptive field convolutional features" into ViT to fix "limited local information interaction and single-feature representation." Outperforms plain ViT backbones on COCO detection and instance segmentation.

### Frozen-DETR (NeurIPS 2024)

Directly relevant to our frozen VLM setup. Key findings:
- Frozen foundation models (CLIP, DINOv2) **enhance** detection but should NOT replace the detection backbone
- Used as **plug-and-play feature enrichment**: CLS token → decoder context, patch tokens → encoder enrichment
- Kept a real detection backbone (ResNet-50) for the spatial pyramid
- Boosted DINO detector from 49.0% → 51.9% AP on COCO (+2.9%)
- Using two foundation models: 49.0% → 53.8% AP (+4.8%)

**Implication for us:** The VLM's value is semantic understanding, not spatial precision. It should *augment* a detection-native backbone, not replace it.

### Deformable DETR (Zhu et al., ICLR 2021)

Extended DETR with **multi-scale deformable attention** — queries attend to a small set of sampling points across multi-scale feature maps rather than doing full attention over a flat sequence. This dramatically improves both convergence speed and small-object detection. Requires multi-scale input features (from an FPN).

---

## Qwen2.5-VL's Grounding vs. Our Detection Setup

Qwen2.5-VL reports strong object grounding accuracy, but this works through a fundamentally different pathway:

| Aspect | Qwen's native grounding | Our Exp2 detection |
|--------|------------------------|-------------------|
| Box output | Language decoder generates coordinate tokens | Separate DETR decoder regresses sigmoid boxes |
| Spatial precision | In the language head's coordinate vocabulary | In ViT patch embeddings (coarse) |
| Training signal | Grounding datasets with text-box pairs | Hungarian matching on ROAD++ tubes |
| Multi-scale | Dynamic resolution + native resolution tokens | Single 1/16 scale |

The spatial precision in Qwen's grounding lives in the **language decoder**, not in the raw ViT patch features we're extracting. We bypass the language decoder entirely.

---

## Architectural Solutions

### Option A: ViTDet-style FPN on existing ViT features

Add a lightweight feature pyramid after the ViT projection:

```
ViT tokens (16×16, 256) → SimpleFeaturePyramid → {P2(64×64), P3(32×32), P4(16×16), P5(8×8)}
```

- **Pros:** Minimal change to existing pipeline, no new backbone, cheap (3-4 conv layers)
- **Cons:** Still limited by the 1/16 base resolution — upsampled features lack true fine-grained detail
- **Expected gain:** Moderate (ViTDet showed +3.2 AP, but our base is much lower)

### Option B: Lightweight CNN branch + VLM enrichment (Frozen-DETR style)

Run a lightweight CNN (EfficientNet-B3/B4) in parallel for spatial features, use VLM features for semantic enrichment:

```
Input frames → EfficientNet → FPN → multi-scale spatial features
Input frames → Qwen ViT (frozen+LoRA) → semantic context
Decoder queries attend to both: spatial for boxes, semantic for classification
```

- **Pros:** Real multi-scale features from a detection-native backbone; VLM provides classification power; matches Frozen-DETR's validated approach
- **Cons:** More parameters, two forward passes per clip, more complex architecture
- **Expected gain:** Significant — addresses root cause directly

### Option C: Multi-scale token extraction from ViT layers

Extract features from multiple ViT layer depths (early=spatial, deep=semantic) and build a pseudo-pyramid:

```
ViT layer 8 tokens → P_early (more spatial detail)
ViT layer 16 tokens → P_mid
ViT layer 24 tokens → P_deep (more semantic)
→ Fuse into multi-scale maps
```

- **Pros:** No additional backbone, exploits existing ViT computation
- **Cons:** Early ViT layers may not have enough spatial detail for precise boxes; untested for Qwen specifically
- **Expected gain:** Moderate — depends on how much spatial info early Qwen layers retain

### Option D: Deformable DETR decoder on multi-scale features

Replace the standard DETR decoder with Deformable DETR's multi-scale deformable attention. Requires multi-scale input from Option A, B, or C.

- **Pros:** State-of-the-art for multi-scale detection; faster convergence than vanilla DETR; better small-object handling
- **Cons:** More complex decoder; needs multi-scale features as prerequisite
- **Expected gain:** High when combined with Option A or B

---

## Recommended Direction

**Option B + D** — EfficientNet CNN branch for spatial features, Qwen ViT for semantic enrichment, Deformable DETR decoder for multi-scale attention. This follows the Frozen-DETR paradigm validated at NeurIPS 2024 and addresses every identified weakness:

| Weakness | Fix |
|----------|-----|
| Single-scale features | EfficientNet FPN provides 4 scales |
| No local spatial detail | CNN features have precise boundaries |
| Semantic understanding needed | Qwen ViT CLS/patch tokens enrich decoder |
| Vanilla DETR slow convergence | Deformable attention on multi-scale maps |

This is essentially the **OpenMixer** direction — the next planned experiment already envisions a dual-stream architecture.

---

## Related

- [[findings/exp2-detr-detection|Exp2 DETR Detection]] — empirical evidence of the localization gap (0.63% agent f-mAP with healthy recall)
- [[findings/exp1b-fcos-detection|Exp1b FCOS Detection]] — FCOS achieved 3.2% agent f-mAP on same ViT features (better than DETR)
- [[concepts/vlm-architectures|VLM Architectures]] — three VLM patterns and how they handle spatial information
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline uses 3D CNN with FPN, achieving 17.76% agent f-mAP
- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — implements the recommended Option B+D approach (EfficientNet + Frozen-DETR encoder fusion)
- [[projects/road-reason|ROAD_Reason Project]] — parent project
