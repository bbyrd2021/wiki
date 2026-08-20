---
type: paper
title: "First Place: ECCV 2024 ROAD++ Challenge — Spatiotemporal Agent Detection"
aliases: ["ECCV 2024 ROAD++ winner", "Zhang 2024 ROAD challenge"]
authors: "Zhang et al."
year: 2024
venue: "ECCV Workshop"
arxiv: "2410.23077"
created: 2026-04-07
updated: 2026-05-13
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, yolov8, challenge, detection, video-mAP, tube-detection, small-object]
status: complete
datasets_used: [road-plusplus, road-waymo]
---

# ECCV 2024 ROAD++ Track 1 Winner — Spatiotemporal Agent Detection

Tengfei Zhang, Heng Zhang, Ruyang Li, Qi Deng, Yaqian Zhao, Rengang Li. **arXiv:2410.23077 | ECCV 2024 Workshop.**

**Winner of Track 1** with **30.82% average video-mAP** (baseline: 5.35%).

---

## Method

A **YOLOv8-based multi-branch engineering solution** — no novel architecture, but heavy domain-specific optimization:

1. **Dual-stream detection** — main stream + dedicated low-light enhancement stream with cross-branch feature fusion. Addresses ROAD-Waymo's nighttime sequences.

2. **Extreme-size detection heads** — separate head paths for very large objects (trucks, buses) and very small objects (distant pedestrians). ROAD++'s agent scale ranges from ~10px to full-frame.

3. **Multi-branch class-imbalance mitigation** — separate branches for majority classes (Car, Ped) vs minority classes (LarVeh, Cyc, TL). YOLOv8x for car/ped (higher capacity); YOLOv8m for minority classes (faster convergence on limited data).

4. **Pre-train + fine-tune** — 30 epochs (LR 0.005, SGD, batch 32) then 20 epochs fine-tuning (LR 0.0005). Augmentation disabled in final 5 epochs.

5. **75/25 train/val split** of the official training data.

---

## Results

### Video-mAP on ROAD++ Test Set

| Method | v-mAP@0.1 | v-mAP@0.2 | v-mAP@0.5 | **Average** |
|--------|-----------|-----------|-----------|-------------|
| Baseline (3D-RetinaNet) | 8.96 | 5.71 | 1.37 | 5.35 |
| **This method** | **39.57** | **34.48** | **18.41** | **30.82** |

**No f-mAP (frame-level mAP) is reported.** The challenge metric is video-mAP (tube-level).

### ROAD-Waymo Baseline f-mAP Reference (from arXiv:2411.01683)

| Backbone | Agent | Action | Location | Duplex | Event |
|----------|-------|--------|----------|--------|-------|
| I3D-08 | 16.7 | 13.9 | 13.2 | 10.1 | 5.3 |
| SlowFast-08 | 15.3 | 14.0 | 12.4 | 10.2 | 5.7 |
| YOLOv8 (agent-only) | **31.6** | — | — | — | — |

YOLOv8 achieves 31.6% agent f-mAP but only for agent detection — no action/location/triplet heads.

---

## Key Takeaways

1. **YOLOv8x is SOTA on ROAD-Waymo agent detection** (31.6% f-mAP, ~68M backbone). 13x our EfficientNet-B0.
2. **No VLM features used** — pure detection engineering. The gap between detection quality and action classification suggests VLM semantics would be complementary.
3. **Low-light handling matters** — dual-stream approach targets ROAD-Waymo's nighttime content.
4. **Class imbalance is a core challenge** — multi-branch (separate models per group) is effective but heavyweight vs our t-norm constraint approach.
5. **Video-mAP@0.5 = 18.41% is the best strict-IoU number reported**, comparable to the 3D-RetinaNet baseline's f-mAP — suggesting tube linking adds little over strong per-frame detection.

---

## Extension Opportunity: YOLOv8 + Frozen-DETR + VLM

Replace our EfficientNet-B0 with YOLOv8x backbone inside the Frozen-DETR architecture:

```
YOLOv8x backbone → FPN ──────────────────────────┐
                                                   ├→ Frozen-DETR Encoder (4+ scales)
CLIP ViT-L/14 → patch tokens (24×24, 1024-dim) ──┘       │
(frozen)       → CLS token (768-dim)                       │
                    │                           strip VLM tokens
                    └── per-layer injection ──→ Decoder → action/loc/triplet heads
```

- YOLOv8x provides the localization precision our EfficientNet-B0 lacks (31.6% vs ~1.76% agent f-mAP)
- CLIP provides scene-level semantics that YOLOv8 doesn't have (enabling action reasoning)
- Frozen-DETR encoder fuses them via bidirectional attention (our [[findings/exp2c-frozen-detr|Exp2c]] architecture)
- T-norm constraints on output heads enforce valid label combinations

This would extend the challenge winner with the novel contribution of [[projects/road-reason|ROAD_Reason]]: VLM-grounded causal reasoning + neuro-symbolic constraints.

---

## Related

- [[datasets/road-plusplus|ROAD++]] | [[findings/road-ped-tube-statistics|ROAD++ Tube Statistics]]
- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — our Frozen-DETR implementation
- [[methods/3d-retinanet|3D-RetinaNet]] — official ROAD baseline
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — architecture we're building on
- [[papers/eccv24-track3|ECCV 2024 Track 3 winner]]
