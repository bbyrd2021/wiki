---
type: finding
title: "Exp2b — Deformable DETR with EfficientNet + Iterative Refinement"
aliases: ["exp2b", "exp2b deformable detr", "deformable detr road-waymo"]
created: 2026-04-27
updated: 2026-05-04
sources:
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/model.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/deformable_decoder.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/losses.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/train.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/config.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/backbone.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/fpn.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/vlm_fusion.py"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/eval_baseline_compat.py"
tags: [finding, road-plusplus, exp2b, deformable-detr, detection, efficientnet, iterative-refinement, auxiliary-losses]
status: draft
---

# Exp2b — Deformable DETR with EfficientNet + Iterative Refinement

Experiment 2b redesigns Exp2's decoder using **Deformable DETR** (Zhu et al., ICLR 2021) with three standard components that were omitted in Exp2: per-frame decoding with temporal self-attention, iterative box refinement, and auxiliary decoder losses. It also adds an **EfficientNet-B0 spatial backbone** with FPN alongside the Qwen2.5-VL ViT, fused via learned gates.

**Status:** Training epoch 27/30 (best checkpoint ep26). Baseline-compat f-mAP evaluated (2026-05-04): agent 1.71%, 10x below 3D-RetinaNet (17.76%). Iterative refinement improved localization over Exp2 (2.7x agent f-mAP) but VLM spatial precision remains the fundamental bottleneck.

**Motivation:** Exp2's DETR decoder produced 0.63% agent f-mAP — 28x below 3D-RetinaNet (17.76%). Root cause analysis identified three missing standard Deformable DETR components, plus a spatial information bottleneck from the ViT's 16x16 patch tokens. Exp2b addresses both.

**Key finding:** VLM features provide strong recall (48-62%) but poor localization at IoU=0.5. Even with EfficientNet + FPN + iterative refinement, the detection pipeline cannot match a simple 3D-ResNet with anchors. This validates the architectural split in Approach 4: use a proper detection backbone (OpenMixer) for localization, reserve VLMs for downstream reasoning.

---

## Architecture

```
8 RGB frames (448x448)
  -> Qwen2.5-VL ViT (frozen + LoRA)       [T, 256, H_vit, W_vit]  (semantic features)
  -> EfficientNet-B0 (partial freeze)       [T, C_i, H_i, W_i]     (spatial features)
  -> FPN (3 levels: P3/P4/P5)              [T, 256, H_i, W_i]      (multi-scale)
  -> VLM Fusion Gates (per-level)           [T, 256, H_i, W_i]      (fused = gate*vlm + (1-gate)*cnn)
  -> Deformable DETR Decoder (6 layers, per-frame with B=T)
      300 learnable queries
      Each layer:
        Self-attention (queries <-> queries)
        -> Deformable cross-attention (queries <-> multi-scale features, K=4 points/head/level)
        -> Temporal self-attention (queries across T frames, learned pos encoding)
        -> FFN
      Per-layer box heads (iterative refinement)
      Auxiliary outputs from every layer
  -> Per-query outputs:
      pred_boxes: [300, T, 4] (sigmoid cx,cy,w,h)
      query_feats: [300, 256]
      aux_outputs: 5 intermediate (feats, boxes) pairs
      agentness_head: [300, 1]
      ClassificationHeads: agent(10), action(22), loc(16), duplex(49), triplet(86)
```

**692M total params, 15.6M trainable** (LoRA + EfficientNet trainable blocks + FPN + fusion gates + decoder + heads).

---

## Three Fixes from Exp2

### 1. Per-frame decoding with temporal self-attention

**Old (Exp2):** Temporal stacking `[T,C,H,W] -> [1,C,T*H,W]` created unusual geometry where the decoder had to learn that y-ranges correspond to different time steps.

**New (Exp2b):** Each frame is a batch element (B=T), standard 2D spatial shapes. Temporal reasoning via dedicated temporal self-attention inside each decoder layer — per-query across T frames, with learned position encoding on Q and K.

### 2. Iterative box refinement

**Old (Exp2):** Single box head on final decoder layer only — no coarse-to-fine localization.

**New (Exp2b):** Per-layer box MLPs (`nn.ModuleList`), each predicting deltas in inverse-sigmoid space. Reference points updated per layer: `new_ref = sigma(sigma_inv(ref) + delta_xy).detach()`. Box head final layers zero-initialized for stable early training. Width/height predicted as `sigma(delta_wh)` starting at ~0.5.

### 3. Auxiliary decoder losses

**Old (Exp2):** Only final decoder layer supervised — early layers got weak gradient signal.

**New (Exp2b):** Every decoder layer produces predictions, all supervised independently. Auxiliary losses averaged and added to main loss. Shared classification heads across layers (only box heads are per-layer).

---

## Dual-Backbone Design

| Branch | Model | Purpose | Resolution |
|--------|-------|---------|------------|
| Spatial | EfficientNet-B0 | Multi-scale spatial features for localization | 448x448, 3 FPN levels |
| Semantic | Qwen2.5-VL ViT + LoRA | Rich semantic features from 7B VLM | 448x448, single scale |

**VLM Fusion:** Per-level learned gates fuse ViT features (upsampled to match FPN spatial dims) with EfficientNet FPN features: `fused = gate * vlm + (1 - gate) * cnn`. Gate bias initialized so `sigmoid(bias) ~ 0.1` — starts CNN-dominant, learns to incorporate VLM features.

**Rationale:** ViT patches (16x16) lack the multi-scale spatial precision needed for tight box regression. EfficientNet provides native multi-scale features (C3/C4/C5 at different resolutions). The ViT contributes semantic understanding; EfficientNet contributes localization precision.

---

## Deformable Attention

Unlike Exp2's standard cross-attention (O(N_q x N_tokens) = 100 x 2048), deformable attention samples only K=4 reference points per head per scale level:

```
Complexity: O(N_q x n_levels x n_heads x K) = O(300 x 3 x 8 x 4) = O(28,800)
```

This enables 300 queries (vs Exp2's 100) and multi-scale feature maps without quadratic cost.

---

## Loss Function

Total loss = **L_main + L_aux**

Where L_main = `lambda_cls * L_cls + lambda_bbox * L_bbox + lambda_giou * L_giou + L_agentness + L_tnorm`

L_aux = average of the same loss computed independently at each of the 5 intermediate decoder layers.

**Auxiliary loss weight = 1.0** (standard practice). Shared classification heads across all layers; only box heads are per-layer.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone (spatial) | EfficientNet-B0 (blocks 0-1 frozen) |
| Backbone (semantic) | Qwen2.5-VL ViT (frozen + LoRA r=8, 8 blocks) |
| FPN levels | 3 (P3/P4/P5), out=256 |
| D_MODEL | 256 |
| NUM_QUERIES | 300 |
| NUM_DECODER_LAYERS | 6 |
| N_DEFORM_POINTS | 4 |
| NHEAD | 8 |
| DIM_FFN | 1024 |
| Clip length | 8 frames |
| MAX_EPOCHS | 30 |
| GRAD_ACCUM | 4 |
| LR_LORA | 5e-5 |
| LR_BACKBONE | 2e-5 (EfficientNet trainable blocks) |
| LR_DECODER | 1e-4 (FPN + fusion + decoder) |
| LR_HEADS | 1e-4 |
| WARMUP_STEPS | 500 |
| lambda_cls / lambda_bbox / lambda_giou | 2.0 / 5.0 / 2.0 |
| lambda_tnorm | 1.0 |
| Warm-start | Exp2 best.pt (438 keys transferred, 0 skipped) |

---

## Eval Pipeline Improvement

Added agentness > 0.01 pre-filter in `eval_baseline_compat.py`. DETR queries predicting "no object" have near-zero agentness — filtering them doesn't affect mAP (AP sweeps thresholds internally) but cuts ~90% of detections, reducing `evaluate_frames()` from hours to minutes.

---

## Key Design Decisions

1. **Shared cls heads across decoder layers** (not per-layer) — standard practice, reduces params. Only box heads are per-layer for iterative refinement.
2. **Temporal attention inside each decoder layer** (not just after) — progressive spatial + temporal refinement.
3. **Temporal position encoding on Q and K only** (not V) — standard practice.
4. **Reference points are 2D** (cx, cy only). Width/height predicted as absolute sigmoid per layer.
5. **Box head final layers zero-initialized** — stable iterative refinement startup.
6. **`queries.clone()` in decoder** prevents aliasing across layers.
7. **Agentness pre-filter at 0.01** — AP-neutral, massive eval speedup.

---

## Smoke Test Results

Forward pass with dummy 8-frame clip (1280x720 PIL images):
- `pred_boxes`: [300, 8, 4]
- `query_feats`: [300, 256]
- 5 auxiliary outputs, each with correct shapes
- Box values in [0, 1] (sigmoid output)
- Backward pass: 429 trainable param groups with gradient
- Total: 692M params, 15.6M trainable

---

## Results — Baseline-Compatible f-mAP at IoU=0.5

| Model | agent_ness | agent | action | loc | duplex | triplet |
|-------|-----------|-------|--------|-----|--------|---------|
| 3D-RetinaNet + Gödel | 23.29 | 17.01 | 15.21 | 13.44 | 13.62 | 9.37 |
| Exp1b FCOS + Gödel | 6.04 | 3.20 | 1.60 | 2.55 | 0.38 | 1.37 |
| Exp2 DETR + Gödel | 2.08 | 0.63 | 0.76 | 1.00 | 0.14 | 0.80 |
| **Exp2b Deformable + Gödel** | **9.72** | **1.71** | **1.78** | **2.46** | **0.47** | **1.25** |

**Recall (mR):** agent_ness=48.7%, agent=59.2%, action=62.2%, loc=49.1%, duplex=60.6%, triplet=59.9%

**Interpretation:** Exp2b significantly improves over Exp2 (agent 2.7x, agent_ness 4.7x) and matches Exp1b FCOS on most heads. However, it remains 10x below 3D-RetinaNet on agent f-mAP. The very high recall confirms the model *detects* agents effectively — the gap is entirely in box precision at IoU=0.5. VLM spatial features (even when augmented with EfficientNet + FPN) lack the multi-scale spatial precision of 3D convolutional backbones for tight localization.

**Conclusion:** VLMs should not be used as detection backbones. Their value is in semantic understanding and reasoning, not spatial localization. This validates the Approach 4 architecture: detection via OpenMixer (purpose-built detector on CLIP-ViP features) with VLM reasoning applied downstream on detected tubes.

---

## Constraint Violation Analysis

T-norm (Gödel) constraint violation rates — percentage of co-predictions that violate valid duplex/triplet combinations:

**Duplex violations (agent + action):**

| Confidence Threshold | 3D-RetinaNet+Gödel | Exp1b (FCOS) | Exp2b (Deformable DETR) |
|---------------------|-------------------|--------------|------------------------|
| conf > 0.3 | **13.2%** (N=40K) | 33.4% (N=81K) | 38.5% (N=2.9M) |
| conf > 0.5 | **13.6%** (N=12K) | 27.3% | 33.5% |
| conf > 0.7 | **10.3%** (N=6.3K) | 23.2% | 25.9% |
| conf > 0.9 | **3.2%** (N=2.1K) | 26.8% | 15.0% |

**Triplet violations (agent + action + location):**

| Confidence Threshold | 3D-RetinaNet+Gödel | Exp2b (Deformable DETR) |
|---------------------|-------------------|------------------------|
| conf > 0.3 | 72.8% (N=43K) | — |
| conf > 0.5 | 60.7% (N=11K) | — |
| conf > 0.7 | 42.6% (N=4.8K) | — |
| conf > 0.9 | 14.7% (N=1.2K) | — |

**Key observations:**

1. **Baseline dominates on duplex violations** — 13.6% vs 27-34% for our models at conf > 0.5. The anchor-based design naturally constrains co-predictions: each anchor represents one physical object, so agent/action predictions are structurally tied to the same spatial location. DETR's 300 queries produce many more soft co-predictions.

2. **Baseline struggles on triplet violations** — 61% at conf > 0.5. Even with the Gödel t-norm loss, independently predicting 3-way (agent, action, location) constraints is much harder than 2-way. The t-norm penalizes pairs effectively but the 3-way combinatorial space (10×22×16 = 3,520 triples, only 86 valid) overwhelms the loss signal.

3. **Exp2b improves at high confidence** — at conf > 0.9, Exp2b (15%) beats Exp1b (27%) and approaches the baseline (3.2%). The t-norm loss is most effective on the model's highest-confidence predictions.

**Implication for Approach 4:** Joint detection + constraint learning in a query-based decoder doesn't improve constraint compliance over anchor-based architectures. The baseline's structural advantage (one anchor = one object) is hard to replicate in DETR-style models. Constraints should be applied *after* a proper detector produces clean tubes — where each tube is already tied to a single agent, making constraint enforcement structurally simpler.

---

## Comparison: Exp2b vs Exp2

| Aspect | Exp2 (standard DETR) | Exp2b (Deformable DETR) |
|--------|---------------------|------------------------|
| Backbone | Qwen ViT only (single-scale) | EfficientNet-B0 + FPN (multi-scale) + Qwen ViT (fused) |
| Attention | Standard cross-attention | Deformable attention (K=4 points/head/level) |
| Temporal | Stacked [1,C,T*H,W] | Per-frame (B=T) + temporal self-attention per layer |
| Queries | 100 | 300 |
| Box refinement | Single box head (final layer) | Per-layer box heads, iterative in inverse-sigmoid space |
| Decoder supervision | Final layer only | All 6 layers (auxiliary losses) |
| Complexity | O(N_q x N_tokens) = O(204,800) | O(N_q x L x H x K) = O(28,800) |
| f-mAP (agent) | 0.63% | TBD (training in progress) |

---

## Related

- [[findings/exp2-detr-detection|Exp2 DETR Detection]] — predecessor; 0.63% agent f-mAP motivates this redesign
- [[findings/exp1b-fcos-detection|Exp1b FCOS Detection]] — FCOS predecessor; 3.2% agent f-mAP
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — why ViT-only features fail at detection
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline (17.76% agent f-mAP)
- [[projects/road-reason|ROAD_Reason Project]] — parent project
