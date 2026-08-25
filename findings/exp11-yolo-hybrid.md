---
type: finding
title: "Exp11 — YOLO Box Source + I3D Hybrids on the Full-Candidate Protocol"
aliases: []
created: 2026-08-24
updated: 2026-08-24

sources:
  - "ROAD_Reason/experiments/exp11_yolo/convert_to_yolo.py"
  - "ROAD_Reason/experiments/exp11_yolo/results_i3d_fullcand_baseline.json"
  - "ROAD_Reason/experiments/exp11_yolo/results_hybrid_score_transfer.json"
  - "ROAD_Reason/experiments/exp11_yolo/results_roialign_lam0_bs16k_gated.json"
  - "ROAD_Reason/experiments/exp11_yolo/results_v8x_best_ep1_agent_fmap.json"
tags: [finding, road-plusplus, exp11, yolo, i3d, roialign, hybrid, f-map, full-candidate]
status: complete
---

# Exp11 — YOLO Box Source + I3D Hybrids on the Full-Candidate Protocol

Moradi directive 2026-08-20: replace the detector, keep the I3D scoring, no
top-40 anywhere. Every number below is f-mAP @ IoU 0.5 on the **identical
36,717 val frames** with the **full candidate set** (≤300 boxes/frame, conf
0.001) — a new protocol era; nothing here is comparable to the top-40 control
family or the strided all-anchor rows.

## The box source

ROAD-Waymo → Ultralytics conversion (verified box-for-box: 3,304,516 label
lines = 3,304,353 boxes + 163 dual-agent annos). YOLOv8x under the ECCV
Track-1 recipe ([[papers/eccv24-track1|SGD 0.005→0.0005, batch 32, aug-off
last 5]]): **35.24 agent f-mAP** — above the report's own 31.6 reference and
2× our replicated 3D-RetinaNet (17.76). The best checkpoint is **epoch 1**;
the full 50-epoch schedule *degrades* to 29.92 (see
[[findings/road-waymo-schedule-overtraining]]). YOLO26x is mid-training on the
identical recipe (batch 16 forced by its dual-assignment memory; its early
curve *rises* where v8x's fell — epoch 3: 0.375 and climbing).

## The complete table (same rows, same evaluator)

| row | agentness | agent | action | loc | duplex | triplet |
|---|---|---|---|---|---|---|
| 3D-RetinaNet, own top-300 dets (same-rows baseline) | 35.36 | 16.67 | 13.23 | 12.77 | **11.32** | **7.54** |
| score transfer (YOLO boxes + IoU-copied I3D scores) | 65.36 | 35.25 | 12.86 | 13.36 | 10.04 | 6.81 |
| **RoIAlign head, λ=0, bs16k, conf-gated (headline)** | **65.36** | **35.25** | **15.86** | **19.68** | 11.19 | 6.64 |

The headline hybrid (AVA recipe: frozen YOLO boxes + frozen I3D P3 features
RoIAligned per box + Linear 256→184 trained minutes on cached vectors) beats
the same-rows baseline on **four of six** label types (agentness +30.0,
agent +18.6, action +2.6, loc +6.9), ties duplex (−0.13), trails triplet
(−0.90). Best known val numbers on this dataset for those four heads.

## Three mechanism findings

**1. Score transfer's ceiling is the match rate.** Only 40.6% of YOLO's
candidates find an I3D detection at IoU ≥ 0.5; unmatched boxes score zero.
The RoIAlign head exists because features exist everywhere detections don't.

**2. A head trained on synthetic negatives cannot rank detector junk.**
Trained on GT + random-rectangle background, the head scored YOLO junk
(conf < 0.01) at 0.89× the sigmoid of real boxes — ungated f-mAP collapsed
(action 5.30). Confidence-gating (head sigmoid × YOLO conf) restored ranking.
Proper fix in progress: retrain with YOLO's own unmatched train-split boxes
as negatives.

**3. Loss-side constraints cannot reach composed outputs through a linear
head.** The corrected Gödel t-norm penalizes *primitive* co-activations
(agent/action/loc columns). Under an element-wise loss, each output column of
a linear head trains independently — so duplex/triplet columns receive **zero
t-norm gradient**: their weights were bit-identical across λ ∈ {0.1, 1, 10}.
The apparent "t-norm duplex gain" was a batch-size artifact (16k vs 64k = 4×
more steps; better-converged rare columns). λ's only real effect is damaging
the primitives it constrains: action 15.86 / 13.37 / 7.07 / 3.43 across
λ = 0 / 0.1 / 1 / 10. Corollary: [[findings/exp9-attribution-grid|exp9's]]
t-norm effects were real only because its heads shared a trainable ViT that
coupled the columns. **Constraints act through shared representations, not
through independent output heads.**

## Why triplet still belongs to the baseline

The baseline's compositional sigmoids were trained end-to-end with the I3D
backbone — the same coupling that finding 3 says constraints (and gradients
generally) need. A frozen-feature linear head starts compositional classes
from scratch with 61–3,240 positives each. This is the precise gap the
junk-negative retrain and [[directions/vlm-reasoning-layer|the contrastive
phrase head]] (text as label-space geometry for the tail) are aimed at.

## Related

- [[findings/road-waymo-schedule-overtraining]] — why best = epoch 1
- [[findings/exp9-attribution-grid]] — the coupled-backbone contrast case
- [[findings/exp8-joint-lora-fusion]] — the protocol family this section retires
- [[papers/eccv24-track1]] — recipe source; its 31.6 reference now surpassed
- [[methods/3d-retinanet]] — the baseline, now feature extractor
