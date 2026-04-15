---
type: method
title: "SparseTemporalPIE (v3 / v4)"
aliases: ["SparseTemporalPIE", "sparse temporal pie", "v3", "v4"]
created: 2026-04-07
updated: 2026-04-07
sources: ["EfficientPIE/README.md", "EfficientPIE/docs/RESULTS.md"]
tags: [method, pie, jaad, cross-attention, pose, efficientnet, local-research]
status: complete
datasets_evaluated: [pie, jaad]
best_accuracy_pie: 0.9261
best_auc_pie: 0.9468
---

# SparseTemporalPIE

Local research contribution (Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T). Extends the IJCAI-25 EfficientPIE paper with three additional information streams.

## v3 Architecture (Best Variant)

```
f_current → backbone → emb (1280-d) ← pose_proj(pose_current, 68-d)
f_context[0..K] → backbone → K context embs
                       │
              cross_attn(Q=emb, K/V=context, K=4)
                       │
               attn_norm + FF + ff_norm
bbox_traj (12-d) ──┐
ctx_feats (5-d) ───┴──► ctx_proj MLP → ctx (128-d)
                       │
             classifier(1408 → 256 → 2)
```

## Results

| Variant | Params | Accuracy (PIE) | AUC (PIE) | Inference |
|---------|--------|----------------|-----------|-----------|
| **v3** | 9.0M | **0.926** | **0.947** | 2.50ms |
| v4 | 1.1M | 0.919 | 0.922 | 0.46ms |
| EfficientPIE (paper) | 0.69M | 0.920 | 0.917 | 0.21ms |

**v3 improves AUC by +0.030 PIE, +0.025 JAAD** — better-calibrated risk scores.

## Related

- [[projects/efficient-pie|EfficientPIE Project]] | [[papers/efficientpie-ijcai-2025|EfficientPIE Paper (IJCAI-25)]]
- [[datasets/pie|PIE]] | [[datasets/jaad|JAAD]]
- [[comparisons/model-comparison|Model Comparison]]
