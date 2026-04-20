---
type: finding
title: "Exp1 Qwen2.5-VL vs 3D-RetinaNet Baseline — ROAD-Waymo Val"
aliases: ["exp1 results", "qwen vs retinanet", "road-waymo baseline comparison"]
created: 2026-04-16
updated: 2026-04-16
sources:
  - "ROAD_Reason/experiments/exp1_road_r/logs/eval_results.json"
  - "PedestrianIntent++/ROAD_plus_plus_Baseline/output/baseline_val_metrics.csv"
tags: [finding, road-plusplus, exp1, qwen, retinanet, baseline, t-norm]
status: complete
---

# Exp1 Qwen2.5-VL vs 3D-RetinaNet Baseline — ROAD-Waymo Val

Comparison of our Experiment 1 (Qwen2.5-VL-7B + classification heads, epoch 6 best checkpoint) against the locally-replicated 3D-RetinaNet-I3D baseline on the ROAD-Waymo val split.

**Important caveat:** metrics are not directly comparable. The 3D-RetinaNet baseline uses **f-mAP @IoU=0.5** (detection + classification jointly scored). Exp1 uses **macro-mAP over PR curves** with **GT boxes provided** (classification only). Exp1 numbers will be inflated on simple heads because there is no detection penalty.

---

## Results Table

| Head | RetinaNet-I3D (f-mAP) | RetinaNet+Gödel (f-mAP) | **Exp1 Qwen (macro-mAP, GT boxes)** |
|------|----------------------|-------------------------|--------------------------------------|
| agent-ness | 23.35% | 23.29% | — |
| agent | 17.76% | 17.01% | **35.7%** |
| action | 15.28% | 15.21% | **22.2%** |
| loc | 13.73% | 13.44% | **33.6%** |
| duplex | 13.44% | **13.62%** | 12.3% |
| triplet | 9.17% | **9.37%** | 8.8% |

Exp1 training details: 10 epochs, Łukasiewicz t-norm (λ=0.1), ViT frozen, cosine LR decay from 2e-4. Best val loss at epoch 6 (val L=0.4003).

---

## Key Findings

### 1. Agent / Action / Loc heads are stronger with GT boxes
Expected — Exp1 is not penalized for missed detections. The Qwen ViT encoder produces richer semantic features than the Kinetics-pretrained 3D ResNet, which helps on agent classification (35.7% vs 17.8%) and loc (33.6% vs 13.7%).

### 2. Duplex and triplet are weaker despite GT boxes
This is the notable result. Even with perfect box inputs, Exp1 scores lower on compound heads than the detection-based baseline:
- duplex: 12.3% vs 13.44%
- triplet: 8.8% vs 9.17%

Root cause: the action head has near-zero recall on rare classes (`IncatLft`, `TurLft/Rht`, `Rev`, `Ovtak`, all F1≈0.000). Compound label predictions require the action head to be correct, so poor action recall cascades directly into duplex/triplet failures.

### 3. T-norm had negligible effect in both models
- RetinaNet: Gödel t-norm changed action mAP by −0.07pp (effectively zero)
- Exp1: logged L_tnorm ≈ 0.00005 throughout (1000× smaller than L_cls)
- Constraint violation rate at eval: **0.02%** — the model almost never co-predicts invalid pairs
- Likely explanation: with ViT frozen and conservative sigmoid heads, predictions are low-confidence on rare classes, trivially satisfying Łukasiewicz constraints of the form `p(A) + p(B) ≤ 1`

### 4. Class imbalance is the core problem
Action class distribution on val: Stop (207K instances) vs Ovtak (88). The model predicts high-frequency classes well and ignores rare ones entirely at threshold=0.5. A threshold sweep or focal loss reweighting is needed before duplex/triplet are meaningful.

---

## Action Head — Per-Class Breakdown (Exp1, sorted by F1)

| Action | F1 | AP | n_gt |
|--------|----|----|------|
| Green | 0.905 | 0.942 | 1,962 |
| Red | 0.859 | 0.890 | 4,519 |
| Stop | 0.731 | 0.817 | 207,203 |
| MovAway | 0.436 | 0.528 | 68,877 |
| Brake | 0.383 | 0.478 | 23,324 |
| MovTow | 0.252 | 0.431 | 61,213 |
| HazLit | 0.072 | 0.072 | 2,223 |
| XingFmLft | 0.050 | 0.102 | 7,150 |
| XingFmRht | 0.044 | 0.147 | 9,563 |
| Xing | 0.019 | 0.128 | 4,396 |
| IncatLft | 0.005 | 0.034 | 3,106 |
| Mov | 0.002 | 0.123 | 7,864 |
| Wait2X | 0.001 | 0.093 | 6,123 |
| Amber | 0.000 | 0.033 | 159 |
| Rev | 0.000 | 0.004 | 364 |
| IncatRht | 0.000 | 0.013 | 2,129 |
| TurLft | 0.000 | 0.018 | 2,468 |
| TurRht | 0.000 | 0.018 | 2,450 |
| MovRht | 0.000 | 0.006 | 673 |
| MovLft | 0.000 | 0.006 | 728 |
| Ovtak | 0.000 | 0.001 | 88 |
| PushObj | 0.000 | 0.007 | 596 |

AP > 0 on near-zero F1 classes (Mov: AP=0.123, Xing: AP=0.128) indicates the model has learned some signal but threshold=0.5 is too high to recover recall. A threshold sweep is recommended.

---

## Fair Comparison (Pending)

The fair apples-to-apples comparison against published numbers requires **Exp1b** (with detection head, computing video-mAP). References:
- ECCV24 Track 1 winner (spatiotemporal detection): **30.82% video-mAP**
- ECCV24 Track 3 winner (activity recognition): **69% mAP**

---

## Next Steps

1. Run `--sweep-threshold` to find optimal action head threshold
2. Add class-weighted BCE or focal loss to address rare action imbalance
3. Unfreeze ViT (Phase 1b with LoRA) — may increase t-norm violations and test whether constraints become meaningful
4. Implement detection head → Exp1b for fair video-mAP comparison

---

## Related

- [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task Architecture]]
- [[methods/3d-retinanet|3D-RetinaNet Baseline]]
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints (T-Norm)]]
- [[datasets/road-plusplus|ROAD++ Dataset]]
- [[projects/road-reason|ROAD_Reason Project]]
