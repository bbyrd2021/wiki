---
type: project
title: "EfficientPIE / SparseTemporalPIE"
aliases: ["EfficientPIE", "SparseTemporalPIE", "sparse temporal pie"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "EfficientPIE/README.md"
  - "EfficientPIE/docs/RESULTS.md"
  - "EfficientPIE/docs/SPARSE_TEMPORAL_PIE.md"
tags: [project, pedestrian-intent, pie, jaad, efficientnet, phd-core]
status: complete
---

# EfficientPIE / SparseTemporalPIE

Fork of the IJCAI-25 paper "EfficientPIE" extended with multi-frame cross-attention, pose velocity, and explicit motion/behavioral context. Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University.

**Repo:** `/data/repos/EfficientPIE/`

## Overview

**EfficientPIE (original)** predicts pedestrian crossing intention from a single image crop using an EfficientNet-inspired backbone. No temporal modeling. Achieves 0.92 accuracy at 0.69M params, sub-millisecond inference.

**SparseTemporalPIE** adds three information streams:
- **Pose features** — ViTPose-B keypoints (34-d static + 34-d velocity)
- **Multi-frame cross-attention** — up to K=4 evenly-spaced context frames attend to current frame
- **Motion + behavioral context** — bbox trajectory statistics (12-d) + ego-vehicle speed, pedestrian action/look (5-d) via late-fusion MLP

## Variants

| Variant | Architecture | Params | Inference | Accuracy | AUC |
|---------|-------------|--------|-----------|----------|-----|
| **v3** | Cross-attention + pose velocity + ctx MLP | 9.0M | 2.50ms | **0.926** | **0.947** |
| v4 | No attention, static pose + ctx MLP | 1.1M | 0.46ms | 0.919 | 0.922 |
| EfficientPIE (paper) | Single frame, visual only | 0.69M | 0.21ms | 0.920 | 0.917 |

End-to-end (including ViTPose-B at 3.875ms): v3 = 6.38ms, v4 = 4.34ms — both real-time at 30fps.

## PIE Test Results

| Metric | EfficientPIE (paper) | v4 | v3 |
|--------|---------------------|----|----|
| Accuracy | 0.920 | 0.919 | **0.926** |
| AUC | 0.917 | 0.922 | **0.947** |
| F1 | 0.952 | 0.953 | **0.957** |
| Precision | 0.960 | 0.958 | 0.957 |

**v3 improves AUC by +0.030 on PIE, +0.025 on JAAD** — better-calibrated risk scores for safety-critical planners.

## JAAD Cross-Dataset Results (v3)

| Metric | EfficientPIE (paper) | v3 (ours) |
|--------|---------------------|-----------|
| Accuracy | **0.890** | 0.878 |
| AUC | 0.860 | **0.885** |
| F1 | 0.620 | **0.633** |

## Architecture (v3)

```
f_current → backbone → emb (1280-d) ← pose_proj(pose_current, 68-d)
                           │
f_context[0..K] → backbone → K context embs ← pose_proj(pose_context)
                           │
                  cross_attn(Q=emb, K/V=context, K=4)
                           │
                    attn_norm + FF(1280→512→1280) + ff_norm
                           │  (enriched, 1280-d)
bbox_traj (12-d) ──┐
ctx_feats  (5-d) ──┴──► ctx_proj MLP → ctx (128-d)
                           │
                 classifier(1408 → 256 → 2)
```

Context frames selected as `np.linspace(0, step-1, min(K=4, step))` — temporal window expands across domain-incremental learning steps (steps 0→14).

## Training Protocol

Domain-incremental learning (IL): trained on progressively larger temporal windows (step 0 = single frame, step 14 = full 15-frame sequence). PIE splits: sets 01+02+04 (train), 05+06 (val), 03 (test).

## Key Scripts

```
scripts/sparsetemporalpie/
  train_SparseTemporalPIE_v3.py          # PIE base training (step 0)
  pie_sparse_incremental_learning_v3.py   # PIE IL steps 2→14
  train_SparseTemporalPIE_v3_jaad.py     # JAAD base training
  jaad_sparse_incremental_learning_v3.py  # JAAD IL steps 2→14
  test_SparseTemporalPIE.py              # evaluation (PIE + JAAD)
```

## Related

- [[datasets/pie|PIE Dataset]] | [[datasets/jaad|JAAD Dataset]]
- [[methods/sparse-temporal-pie|SparseTemporalPIE Method]]
- [[comparisons/model-comparison|Model Comparison Table]]
