---
type: comparison
title: "Model Comparison — Input/Output Patterns"
aliases: ["model comparison", "input output patterns"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [comparison, methods, jaad, pie, road-plusplus]
status: complete
---

# Model Comparison — Input/Output Patterns

From SYNTHESIS.md Part 6. Overview of published models and their architectures.

## ROAD++ Models

| Model | Input | Output | Key Result |
|-------|-------|--------|------------|
| [[methods/3d-retinanet\|3D-RetinaNet]] | RGB video clips → 3D CNN | Agent/action/event multi-label per tube | Baseline; all challenge comparison |
| ECCV24 Track 1 winner | Multi-scale backbone + temporal aggregation | Agent class per tube | 30.82% video-mAP |
| ECCV24 Track 3 winner | Video Transformer (Kinetics pretrained) | 64-class atomic activity per agent | 69% mAP |
| ROAD-R constrained | Same as 3D-RetinaNet + t-norm loss | Agent + logic compliance | Improved requirement compliance |

## JAAD + PIE Models

| Model | Dataset | Input | Accuracy |
|-------|---------|-------|---------|
| [[methods/intentformer\|IntentFormer]] (2024) | JAAD+PIE | RGB crops + segmentation + trajectory | 93% PIE / 92% JAAD |
| [[methods/gtranspdm\|GTransPDM]] (2025) | JAAD+PIE | Pose skeleton + position decoupled + ego speed | 92% PIE / 87% JAAD / **0.05ms** |
| [[methods/pedformer\|PedFormer]] (2023) | JAAD+PIE | bbox + OBD + pose + context (cross-modal Transformer) | ~88% PIE |
| [[methods/ista-net\|ISTA-Net]] (2022) | PIE | bbox + optical flow + OBD → graph + Transformer | 89.5% PIE |
| [[methods/encore\|ENCORE]] (2024) | PIE | bbox + OBD (scale-aware) | MSE ~73.4px² at 1.5s |
| [[methods/sparse-temporal-pie\|SparseTemporalPIE v3]] | PIE+JAAD | Multi-frame cross-attn + pose + ctx | **0.926 acc / 0.947 AUC** PIE |
| Occ-Aware Diffusion (2024) | JAAD+PIE | bbox + occlusion → diffusion | +5% under occlusion |

## Common Patterns (from SYNTHESIS Part 6)

1. **Transformer over time** has replaced LSTM/GRU across all three dataset families
2. **Multimodal fusion** (RGB + pose + trajectory + ego speed) consistently outperforms single modality
3. **Multitask learning** (trajectory + intent jointly) improves both tasks over separate models
4. **Occlusion robustness** is an active frontier
5. **VLM/foundation models** are just emerging for this task (2025)
6. **Class imbalance**: JAAD/PIE use balanced accuracy; ROAD++ uses focal loss for rare classes

## Related

- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]] | [[datasets/road-plusplus|ROAD++]]
- [[comparisons/dataset-comparison|Dataset Comparison]]
