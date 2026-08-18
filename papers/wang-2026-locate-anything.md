---
type: paper
title: "LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding"
aliases: ["LocateAnything", "Parallel Box Decoding", "PBD"]
created: 2026-05-27
updated: 2026-05-27
sources:
  - "https://arxiv.org/abs/2605.27365"
  - "https://research.nvidia.com/labs/lpr/locate-anything/"
tags: [paper, detection, vlm, grounding, nvidia, open-vocabulary]
status: complete
authors: "Wang et al."
year: 2026
venue: "arXiv (NVIDIA Tech Report)"
arxiv: "2605.27365"
datasets_used: [coco, lvis, screenspot-pro]
---

# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

**Wang et al. (2026)** — NVIDIA, Hong Kong PolyU, Princeton, Nanjing, UIUC

## Problem

VLMs that do visual grounding serialize each bounding box into multiple 1D coordinate tokens decoded autoregressively. This (1) breaks geometric coherence — each coordinate is predicted independently without knowing the others, and (2) creates an inference bottleneck — box coordinates are generated strictly sequentially, one token at a time.

## Key Contribution: Parallel Box Decoding (PBD)

Instead of serializing `(x1, y1, x2, y2)` as four separate tokens, PBD treats each bounding box (or point) as an **atomic unit** decoded in a single parallel step. This preserves intra-box geometric coherence and unlocks substantial parallelism.

### Architecture

| Component | Details |
|-----------|---------|
| Vision encoder | MoonViT (up to 2.5K resolution input) |
| Language model | Qwen2.5-3B-Instruct |
| Projector | MLP |
| Total params | 3B |
| Training seq length | 25,600 tokens |
| Max new tokens | 8,192 |
| Box format | `<box> x1, y1, x2, y2 </box>` or `<box> x, y </box>` (points) |

### Three Inference Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `fast` | MTP only, no fallback | Fastest | Good for simple scenes |
| `slow` | Pure autoregressive decoding | Slowest | Most robust |
| `hybrid` (default) | MTP first, AR fallback on uncertain boxes | Balanced | Best overall |

The hybrid mode detects anomalies during parallel decoding and falls back to autoregressive generation for those boxes — balancing throughput with reliability.

## LocateAnything-Data

A large-scale training corpus curated via a scalable data engine:

| Stat | Value |
|------|-------|
| Images | 12M unique |
| Language queries | ~140M |
| Bounding boxes | 785M |
| Labeling | Human + synthetic (Qwen3-VL, Molmo, SAM 3, Rex-Omni) + auto-verification |

**Data composition by task:**
- General object detection: 66.9%
- GUI element grounding: 16.5%
- Referring expression: 7.3%
- Text localization/OCR: 3.6%
- Layout grounding: 3.5%
- Point-based localization: 2.2%

Domains span natural scenes, robotics, **driving**, GUI interaction, and document understanding.

## Supported Tasks

| Task | Output |
|------|--------|
| Open-set object detection | Multiple boxes |
| Phrase grounding (single/multi) | Single/multiple boxes |
| Text grounding + scene text detection | Boxes |
| Document layout detection | Boxes |
| GUI element grounding | Box or point |
| Point-based localization | Point |

## Results

- **LVIS:** +3.8% F1 over prior SOTA
- **COCO / ScreenSpot-Pro:** competitive or SOTA across IoU thresholds
- **Throughput:** 2.5x higher boxes-per-second than coordinate-based autoregressive alternatives (measured on H100, batch=1)
- Evaluates on ~48K images (box tasks) and ~35K images (point tasks)

The paper emphasizes the **speed-accuracy frontier** — PBD doesn't just trade accuracy for speed; it improves both simultaneously because geometric coherence eliminates cross-coordinate errors.

## Relevance to This Research

### Connection to Exp2 Series (ROAD_Reason)

The exp2 series progressively builds a DETR-style detector on frozen VLM features for ROAD-Waymo. LocateAnything tackles a related but different bottleneck:

| Aspect | Exp2 series | LocateAnything |
|--------|-------------|----------------|
| Detection paradigm | DETR queries + Hungarian matching | Generative VLM + parallel coordinate decoding |
| VLM role | Frozen feature extractor (CLIP ViT-L/14) | End-to-end backbone + decoder (MoonViT + Qwen2.5) |
| Box generation | Learned queries attend to features, predict boxes directly | Language model generates box coordinates as structured tokens |
| Multi-label | Sigmoid heads + focal loss (184-dim) | Language-based class descriptions |
| Key bottleneck addressed | Score-localization decorrelation, query coverage | Autoregressive coordinate serialization, geometric incoherence |

### Potential Applications

1. **Alternative to DETR for Approach 4:** If generative grounding can handle ROAD-Waymo's multi-label structure (49 duplexes, 86 triplets), PBD could replace the entire DETR decoder pipeline while maintaining geometric coherence. However, the multi-label sigmoid structure would need to be mapped to language.

2. **Dataset labeling:** LocateAnything-3B is explicitly designed for automated annotation. Could be used to generate pseudo-labels for ROAD-Waymo clips where annotation coverage is sparse, or for the CoVLA/BDD-X datasets in Approach 3's Stage 1 pre-training.

3. **Open-vocabulary detection baseline:** The model supports open-set detection across diverse domains including driving. A zero-shot evaluation on ROAD-Waymo agent classes (Car, Ped, LarVeh, etc.) would provide an interesting comparison point against the 3D-RetinaNet baseline.

### Key Architectural Insight

PBD's core idea — **treating structured outputs as atomic units rather than serialized tokens** — parallels the exp2 series' lesson about flat vs. hierarchical classification heads. In exp2f, collapsing 6 separate sigmoid heads into a single flat 184-dim vector (treating the full label set as one atomic prediction) fixed the score-localization decorrelation. Both approaches recognize that decomposing coupled structures into independent predictions creates misalignment.

## Related

- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — frozen VLM features for detection (the approach exp2 replicates)
- [[papers/bao-2025-openmixer|OpenMixer]] — DETR-style open-vocabulary action detection (planned for Approach 4)
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the broader problem of using VLM features for detection
- [[concepts/vlm-architectures|VLM Architectures]] — generative VLM pattern that LocateAnything extends
- [[findings/exp2-series-narrative|Exp2 Series Narrative]] — the detection pipeline this could inform
