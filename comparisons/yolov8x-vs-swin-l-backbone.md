---
type: comparison
title: "YOLOv8x vs Swin-L: Detection Backbone Comparison for ROAD-Waymo"
aliases: ["yolo vs swin", "backbone comparison"]
created: 2026-05-12
updated: 2026-05-12
sources:
  - "ROAD_Reason/experiments/exp2d_swin_detr/config.py"
  - "ROAD_Reason/experiments/exp2d_swin_detr/backbone.py"
tags: [comparison, detection, yolov8, swin, backbone, road-plusplus, frozen-detr]
status: draft
---

# YOLOv8x vs Swin-L: Detection Backbone for ROAD-Waymo

Choosing the CNN backbone for our [[papers/fu-2024-frozen-detr|Frozen-DETR]] pipeline. Both are proven detection backbones; the question is which fits our dual-backbone architecture (CNN + frozen CLIP) and the ROAD-Waymo spatiotemporal task.

---

## Architecture Overview

| | **YOLOv8x** | **Swin-L** |
|--|-------------|------------|
| Type | CSPDarknet (pure CNN) | Hierarchical Vision Transformer |
| Params | ~68M | ~195M |
| Core operation | Cross-stage partial convolutions | Shifted window self-attention |
| Multi-scale | 3 stages → PANet neck (P3/P4/P5) | 4 stages → native hierarchical (C1/C2/C3/C4) |
| Pretraining | COCO detection (Ultralytics) | ImageNet-22K classification |
| Input resolution | Flexible (typically 640) | 384×384 (window_size=12) |
| Designed for | End-to-end real-time detection | Feature extraction for any detection head |

**Swin-L** uses windowed self-attention at every stage — each local 12×12 window attends internally, then windows shift to exchange information. This gives it a growing receptive field similar to CNNs but with attention's capacity for complex spatial relationships.

**YOLOv8x** is a classic CNN optimized for throughput: depthwise convolutions, cross-stage partial connections, and a tightly integrated PANet feature pyramid. Everything is designed to run fast end-to-end.

---

## ROAD-Waymo Performance

### Known results

| Method | Agent f-mAP | Action f-mAP | Backbone | Total trainable |
|--------|------------|-------------|----------|-----------------|
| YOLOv8x (standalone, agent-only) | **31.6%** | — | YOLOv8x (68M) | ~68M |
| ECCV 2024 winner (v-mAP@0.5) | 18.41% (video) | — | YOLOv8x/m | ~68M |
| 3D-RetinaNet I3D (baseline) | 16.7% | 13.9% | ResNet-50 I3D (46M) | ~50-55M |
| Exp2c Frozen-DETR (ep15) | 1.76% | 1.58% | EfficientNet-B0 (5.3M) | ~20M |
| **Exp2d Frozen-DETR (training)** | **TBD** | **TBD** | **Swin-L (195M)** | **~207M** |

YOLOv8x's 31.6% is the highest known agent f-mAP on ROAD-Waymo — but that's as a **standalone detector** with its own neck and head, fully fine-tuned. No action/location/triplet classification.

Swin-L has no ROAD-Waymo results yet. Our Exp2d is the first test.

### What we expect from Exp2d

Swin-L + DINO achieves 58.0 AP on COCO (vs YOLOv8x at 53.9). If that advantage transfers to ROAD-Waymo within our Frozen-DETR pipeline, Exp2d should significantly beat Exp2c's 1.76% agent f-mAP. The encoder/decoder/CLIP/heads are all warm-started from Exp2c — only the backbone and FPN are fresh.

---

## Fit for Frozen-DETR Pipeline

Our architecture: **CNN backbone → FPN → deformable encoder (+ CLIP patches) → decoder (+ CLS injection) → heads**.

| Factor | YOLOv8x | Swin-L |
|--------|---------|--------|
| **Feature extraction** | Need to hack into Ultralytics internals to extract intermediate features | `timm.create_model(..., features_only=True)` — one line |
| **Paper-tested** | Not tested in Frozen-DETR | **Directly tested** — Frozen-DETR's best results use Swin-L |
| **FPN compatibility** | CSPDarknet outputs at 3 scales → works with FPN | 4 hierarchical stages → pick 3 for FPN, natural fit |
| **Encoder fusion with CLIP** | CNN features only — attention-based fusion may underperform since backbone doesn't "speak attention" | Swin is already attention-based — fuses naturally with CLIP tokens in the deformable encoder |
| **Output channels** | Non-standard (varies by stage, needs mapping) | Clean: [384, 768, 1536] at stages 1/2/3 |

**Winner: Swin-L.** The Frozen-DETR architecture was designed around Swin. Using YOLOv8x would mean fighting the architecture rather than leveraging it.

---

## Training Characteristics

| Factor | YOLOv8x | Swin-L |
|--------|---------|--------|
| **GPU memory** | ~8-10 GB (smaller model) | ~16 GB (current Exp2d) |
| **Fits on A6000 (48 GB)?** | Easily | Easily (32 GB headroom) |
| **Training speed per epoch** | Faster (CNN ops, less memory bandwidth) | Slower (attention at every stage) |
| **Freezing strategy** | Freeze early stages (custom) | Freeze stages 0-1 via timm (4.8M frozen, 190M trainable) |
| **Optimizer memory** | ~0.5 GB (AdamW for 68M params) | ~1.5 GB (AdamW for 190M trainable params) |
| **Convergence** | Fast (detection-pretrained on COCO) | May need more epochs (ImageNet-22K classification pretrained) |

---

## Feature Quality Comparison

### Spatial precision (localization)

YOLOv8x was **designed** for tight bounding boxes — CSPDarknet preserves spatial detail through skip connections and the PANet neck fuses low-level and high-level features bidirectionally. This is why it achieves 31.6% agent f-mAP standalone.

Swin-L's patch embedding (4×4 patches at stage 0) loses some spatial granularity compared to a CNN's pixel-level convolutions. However, the windowed attention gives it strong local feature extraction, and the FPN top-down pathway recovers spatial detail from the coarse-to-fine hierarchy.

### Semantic richness (classification)

Swin-L was pretrained on ImageNet-22K (14M images, 22K classes) — it has seen far more visual concepts than YOLOv8x's COCO pretraining (118K images, 80 classes). For ROAD-Waymo's **action and triplet classification** (22 actions × 16 locations × 10 agents), this broader semantic base should help distinguish fine-grained categories.

### VLM fusion compatibility

Both produce multi-scale features. But Swin-L's features are **already attention-derived** — when they enter the deformable encoder alongside CLIP patch tokens, the self-attention mechanism operates on a homogeneous feature space. YOLOv8x's CNN features are fundamentally different in nature from CLIP's ViT features, which could create a modality gap in the encoder.

---

## The Strategic Question

| If you want... | Choose... |
|----------------|-----------|
| Best proven ROAD-Waymo agent detection | YOLOv8x standalone (31.6% f-mAP) |
| Best backbone for Frozen-DETR + CLIP pipeline | **Swin-L** (paper-tested, attention-native) |
| Fastest training | YOLOv8x (smaller, CNN) |
| Best action/triplet classification potential | **Swin-L** (ImageNet-22K pretraining) |
| Easiest implementation | **Swin-L** (timm one-liner vs Ultralytics hacking) |
| Fallback if Swin-L underperforms | YOLOv8x as backbone (Exp2e) — strip neck/head, extract CSPDarknet features |

### Our decision: Swin-L (Exp2d)

1. It's what Frozen-DETR was tested with
2. Attention-based features fuse naturally with CLIP in the encoder
3. ImageNet-22K pretraining brings richer semantics for action classification
4. Clean integration via timm
5. 48 GB A6000 handles it easily

If Exp2d's f-mAP is still low after 30 epochs, YOLOv8x backbone becomes the next experiment (Exp2e). That would essentially be extending the [[papers/eccv24-track1|ECCV 2024 challenge winner]]'s detector with CLIP semantic features — a strong fallback.

---

## Experiment Status

| Experiment | Backbone | Status | Agent f-mAP |
|-----------|----------|--------|-------------|
| Exp2c | EfficientNet-B0 (5.3M) | Ep18/30, training | 1.76% (ep15) |
| **Exp2d** | **Swin-L (195M)** | **Ep1/30, training** | **TBD** |
| Exp2e (potential) | YOLOv8x (68M) | Not started | — |

Both Exp2c and Exp2d are running in parallel on separate GPUs (A6000 × 2).

---

## Related

- [[papers/eccv24-track1|ECCV 2024 Track 1 Winner]] — YOLOv8x-based, 30.82% v-mAP
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — architecture using Swin-L + CLIP
- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — EfficientNet-B0 baseline (1.76% f-mAP)
- [[methods/3d-retinanet|3D-RetinaNet]] — official ROAD baseline (16.7% f-mAP)
- [[comparisons/fusion-for-detection-lit-review|CNN-VLM Fusion Lit Review]] — why encoder fusion beats scalar gates
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the core problem being addressed
