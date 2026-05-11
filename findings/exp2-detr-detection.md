---
type: finding
title: "Exp2 — DETR-Style Tube Detection on ROAD-Waymo"
aliases: ["exp2", "exp2 detr", "detr road-waymo"]
created: 2026-04-21
updated: 2026-04-24
sources:
  - "ROAD_Reason/experiments/exp2_detr_qwen/model.py"
  - "ROAD_Reason/experiments/exp2_detr_qwen/matcher.py"
  - "ROAD_Reason/experiments/exp2_detr_qwen/losses.py"
  - "ROAD_Reason/experiments/exp2_detr_qwen/train.py"
  - "ROAD_Reason/experiments/exp2_detr_qwen/config.py"
  - "ROAD_Reason/experiments/exp2_detr_qwen/eval_baseline_compat.py"
  - "ROAD_Reason/tnorm_loss.py"
tags: [finding, road-plusplus, exp2, detr, detection, lora, focal-loss, t-norm, qwen, hungarian-matching]
status: draft
---

# Exp2 — DETR-Style Tube Detection on ROAD-Waymo

Experiment 2 replaces Exp1b's FCOS dense detection with a **DETR-style set-prediction detector** on the same Qwen2.5-VL ViT features. The core motivation: Exp1b's internal classification metrics were strong (agent=60.6%) but baseline-compatible f-mAP was only 3.2% — far below 3D-RetinaNet's 17.0%. The root cause was FCOS box quality under the IoU=0.5 criterion, not classification. DETR's learnable queries + Hungarian matching + L1+GIoU supervision addresses this directly.

**Status:** Complete (Apr 24, ep30). Best checkpoint by matched action mAP = 0.3095 (ep26). Baseline-compatible f-mAP evaluated with fixed pipeline (Apr 24) — agent 0.63%, ~28× below 3D-RetinaNet's 17.76%. See [[#eval-pipeline-fix]] for prior eval bugs. **Superseded by [[findings/exp2b-deformable-detr|Exp2b]]** which fixes three missing standard Deformable DETR components.

---

## Architecture

```
8 RGB frames (448×448)
  → Qwen2.5-VL ViT (frozen + LoRA, same as exp1b)    [T=8, H'=16, W'=16, D=3584]
  → FeatureProjection(3584 → 256)                     [T*H'*W' = 2048 tokens, 256]
  → Flatten + SpatiotemporalPositionEncoding           [2048, 256]
  → DETR Decoder (6 layers, clip-level)
      100 learnable queries × 2048-token memory
      self-attn → cross-attn → FFN per layer (pre-norm)
  → Per-query outputs:
      box_head(query_feat + temporal_embed[t]) → sigmoid  [N_queries, T, 4]  (one tube per query)
      agentness_head(query_feat) → [N_queries, 1]         (real learned binary score)
      ClassificationHeads: agent(10), action(22), loc(16), duplex(49), triplet(86)
  → Hungarian matching against GT tubes (training)
  → Filter by agentness confidence (inference)
```

**Key change from Exp1b:** Instead of 2048 independent per-token FCOS predictions, 100 learnable object queries attend to ALL 2048 spatiotemporal tokens simultaneously via cross-attention. Each query represents a **tube** — its single classification and T boxes capture one agent across the clip. This gives cross-frame temporal context analogous to the 3D convolutions in RetinaNet, without duplicating predictions.

**Warm-start:** Loads Exp1b's `best.pt` — only `vit.*` keys (including LoRA adapters) transfer. Projection, decoder, and classification heads initialize fresh.

---

## Loss Function

Total loss = **λ_cls · L_cls + λ_bbox · L_bbox + λ_giou · L_giou + L_agentness + L_tnorm**

### L_agentness — Focal on all 100 queries
```
targets[matched_pred] = 1.0   (matched to a GT tube)
targets[unmatched]    = 0.0
FL(p, y) = -(1-p_t)^γ · log(p_t),  γ=2.0
```
Applied to all N=100 queries. Directly mirrors RetinaNet's objectness head design (Lin et al., ICCV 2017). The real learned agentness head is `nn.Linear(d_model, 1)` — no proxies or synthetic scores.

### L_cls — Focal per head (matched queries only)
Same focal BCE with per-class inverse-frequency α as Exp1b. Applied only to Hungarian-matched queries against their assigned GT tubes.

### L_bbox — L1 on tube boxes (matched queries only)
Mean L1 distance in [cx,cy,w,h] across frames where GT is present (`tube["box_mask"]`). Only supervises frames where the GT agent actually appears — no loss on frames where the tube is absent.

### L_giou — Generalized IoU (matched queries only)
Mean (1 − GIoU) across frames where GT is present. GIoU provides gradients even when predicted and GT boxes don't overlap (pure L1 is ambiguous for relative position of non-overlapping boxes).

### L_tnorm — Gödel t-norm (matched queries only)
```
flat = [agentness(1), agent(10), action(22), loc(16)]  → [N_matched, 49]
```
Same Gödel constraint violation structure as Exp1b. Uses real predicted agentness — the t-norm penalty is modulated by detection confidence.

---

## Baseline Numbers (Verified)

All Qwen-based experiments use the same ViT backbone. Evaluation methodology differs — read notes carefully before comparing across rows.

### Baseline-compatible f-mAP at IoU=0.5 (apples-to-apples)

| Model | Epoch | Eval | agent_ness | agent | action | loc | duplex | triplet |
|-------|-------|------|-----------|-------|--------|-----|--------|---------|
| 3D-RetinaNet-I3D | 25 | f-mAP | 23.35 | 17.76 | 15.28 | 13.73 | 13.44 | 9.17 |
| 3D-RetinaNet-I3D + Gödel | 29 | f-mAP | 23.29 | 17.01 | 15.21 | 13.44 | **13.62** | 9.37 |
| Exp1b FCOS + Gödel | 15 | f-mAP | 6.0 | 3.2 | 1.6 | 2.5 | 0.38 | 1.37 |
| **Exp2 DETR + Gödel** | **26** | **f-mAP** | **2.08** | **0.63** | **0.76** | **1.00** | **0.14** | **0.80** |

**Exp2 is below Exp1b on all heads except action and triplet.** The DETR decoder's 100 queries with Hungarian matching did not improve localization over FCOS — agent f-mAP is 0.63% vs Exp1b's 3.2% (5× worse). Recall is healthy (agent mR=26.0%), suggesting the model finds agents but box precision at IoU=0.5 is poor.

### For reference — oracle-box macro-mAP (not directly comparable)

| Model | Epoch | Eval | agent | action | loc | duplex | triplet |
|-------|-------|------|-------|--------|-----|--------|---------|
| Exp1 frozen ViT + Łukasiewicz | 6 | macro-mAP, GT boxes | 35.7 | 22.2 | 33.6 | 12.3 | 8.8 |
| Exp1b FCOS + Gödel | 15 | macro-mAP, fg tokens | 60.6 | 32.4 | 50.0 | 23.1 | 17.5 |

Exp1/Exp1b internal numbers are inflated vs f-mAP: GT boxes remove the detection penalty entirely; fg-token mAP only scores tokens already matched to GT. The 20× gap between Exp1b internal (60.6%) and Exp1b f-mAP (3.2%) on agent is structural — FCOS box quality under IoU=0.5, not classification failure.

---

## Comparison: Exp2 vs 3D-RetinaNet + Gödel

| Aspect | 3D-RetinaNet + Gödel | Exp2 DETR + Gödel |
|--------|---------------------|-------------------|
| Backbone | 3D-ResNet (Kinetics pretrained) | Qwen2.5-VL-7B ViT + LoRA |
| Temporal context | 3D convolutions (implicit) | Cross-attention over T×H'×W' tokens (explicit, longer range) |
| Detection mechanism | IoU-based anchor assignment + NMS | Hungarian bipartite matching, no NMS |
| Box supervision | L1 on anchor offsets only | **L1 + GIoU** (GIoU gradient exists for non-overlapping boxes) |
| Objectness | Focal on anchor scores | Focal on all 100 queries (matched=1, unmatched=0) |
| Classification | Focal per head | Focal per head + **per-class inverse-frequency α** |
| T-norm | Gödel on tube preds | Gödel on matched queries, same flat vector layout |
| Output representation | Per-anchor → NMS-merged tubes | 100 native tubes per clip (no merging needed) |
| Duplicate predictions | Eliminated by NMS | Eliminated by matching formulation |

**Expected advantages of Exp2:**
1. Better localization: L1+GIoU box supervision with Hungarian matching vs IoU-based anchor assignment
2. No NMS required: Hungarian matching is a set prediction — one query = one tube, no duplicates
3. Stronger backbone features from Qwen2.5-VL (7B parameters vs 3D-ResNet)
4. Cross-frame temporal context from attention (each query sees all 8 frames simultaneously)

**Known weakness:** Duplex is expected to lag — caused by averaged 5-head loss diluting duplex gradient to 1/5th of baseline's flat-vector treatment, plus t-norm doesn't touch duplex logits directly. See Duplex Loss Analysis section.

---

## Comparison: Exp2 vs Exp1b

| Aspect | Exp1b (FCOS) | Exp2 (DETR) |
|--------|-------------|------------|
| Detector | Dense per-token FCOS, 2048 candidates | 100 learnable queries |
| Box regression | L,T,R,B distances from token center | [cx,cy,w,h] direct sigmoid, L1 + GIoU |
| Assignment | FCOS center-inside (multiple positives per GT) | Hungarian 1:1 bipartite matching |
| Temporal context | Per-frame independent | Cross-attention over full clip (8 frames) |
| Tube output | None (frame-level only) | Native tube per query (supports v-mAP) |
| Post-processing | NMS at IoU=0.5 | None |
| Classification scope | Per foreground token | Per matched query (tube-level) |
| Baseline-compat f-mAP | agent=3.2% (FCOS box quality bottleneck) | agent=0.63% (worse — DETR localization weaker than FCOS) |
| v-mAP | Not applicable | First experiment with tube output |

---

## Implementation Details

**Box format:** DETR decoder outputs sigmoid [cx,cy,w,h] in [0,1]. GT boxes from dataset are [x1,y1,x2,y2] in [0,1]. Matcher and loss both convert to the appropriate format internally (`box_xyxy_to_cxcywh` / `box_cxcywh_to_xyxy`).

**T-norm flat vector:** `[agentness(1), agent(10), action(22), loc(16)]` — matches `TNormConstraintLoss` hardcoded offsets (AGENT_OFFSET=1, ACTION_OFFSET=11, LOC_OFFSET=33). This was a bug in the initial Codex implementation (was missing agentness at offset 0, shifting all class indices) — fixed in code review.

**Tube box indexing in loss:** `pred_cxcywh[i][tube["box_mask"]]` selects `[n_present_frames, 4]` for matched query `i`, compared against `gt_cxcywh` of the same shape. Per-frame GIoU loops explicitly over frames where GT is present.

**Matcher class cost:** Iterates over `tube["labels"]` keys (agent/action/loc/duplex/triplet) — agentness is not a semantic class and does not enter the class cost. The matching is driven entirely by semantic class + box cost.

**Three optimizer param groups:**
- LoRA: 5e-5 (same ViT adapters as Exp1b)
- Decoder + projection + position encoding: 1e-4 (fresh-init, needs faster LR)
- Classification heads: 1e-4

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone | Qwen2.5-VL-7B ViT (frozen + LoRA r=8, 8 blocks) |
| D_MODEL | 256 (decoder hidden dim) |
| NUM_QUERIES | 100 |
| NUM_DECODER_LAYERS | 6 |
| NHEAD | 8 |
| DIM_FFN | 1024 |
| DROPOUT | 0.1 |
| Clip length | 8 frames |
| Clip stride | 16 frames |
| MAX_EPOCHS | 30 |
| GRAD_ACCUM | 4 (effective batch 4) |
| WARMUP_STEPS | 500 |
| GRAD_CLIP | 1.0 |
| λ_cls / λ_bbox / λ_giou | 2.0 / 5.0 / 2.0 |
| λ_tnorm | 1.0 |
| T-norm type | Gödel |
| CONFIDENCE_THRESHOLD | 0.3 (inference) |
| Best ckpt criterion | val matched action mAP |

---

## What to Watch During Training

1. **Matched query count** (`n_matched` in log): should stabilize at ~10–20 GT tubes per clip as the decoder learns to assign queries to agents
2. **L_bbox and L_giou falling together**: both should decrease; if L_giou stays high while L_bbox is low, the boxes may be regressing to wrong positions
3. **L_agentness**: should see rapid improvement in early epochs as the model learns to suppress unmatched queries
4. **Baseline-compat f-mAP at epoch 5**: any improvement over Exp1b's 3.2% validates the DETR localization hypothesis

---

## Duplex Loss Analysis

Duplex (49 classes, agent×action pairs) consistently lags behind every other label type — including triplet (86 classes), which is a strictly harder task. Three compounding causes:

### 1. Averaged classification loss dilutes duplex gradient

The baseline (3D-RetinaNet) predicts all label types as a **single flat vector** (1+10+22+16+49+86 = 184 dims) and applies one focal loss to the entire vector. Every dimension gets the same gradient scale.

Our Exp2 uses **5 separate classification heads** and *averages* the 5 focal losses before applying λ_cls:

```python
return torch.stack(losses).mean()   # each head gets 1/5th weight
```

This means each head — including duplex — gets 1/5th the effective gradient magnitude compared to the baseline's flat-vector treatment. This is not a deliberate design choice; it's the consequence of averaging rather than summing.

### 2. T-norm loss does not touch duplex

The Gödel t-norm constraint loss operates on a flat vector of `[agentness(1), agent(10), action(22), loc(16)]` = 49 dims. Duplex and triplet logits are **not included**. The t-norm indirectly shapes duplex by constraining its component dimensions (agent and action), but duplex predictions receive no direct constraint gradient.

The baseline's `TNormConstraintLoss` has the same limitation — it also operates only on agent/action/loc. So both models share this gap; it doesn't explain the relative weakness in Exp2.

### 3. Class imbalance — duplex vs. triplet paradox

Duplex class 13 accounts for ~43.4% of all duplex annotations; 8/49 duplex classes have zero validation examples. On the surface this should hurt triplet more (86 classes, even rarer per class). But the opposite holds:

**The fragmentation effect:** The dominant duplex class (e.g., "Car+Moving") fragments into multiple triplet classes when location is added ("Car+Moving+NearLeft", "Car+Moving+NearRight", etc.). This makes triplet's per-class distribution *more balanced* per class than duplex's. Since mAP averages AP equally across all classes, better per-class balance → higher macro-mAP even with more total classes.

### Summary

| Factor | Baseline | Exp2 |
|--------|----------|------|
| Duplex loss weight | 49/184 of total flat loss | 1/5th of averaged 5-head loss |
| T-norm on duplex | No | No (same) |
| Sparse query problem | N/A (dense anchors) | 100 queries, fewer duplex positives |

**Potential fix:** Replace `torch.stack(losses).mean()` with a weighted sum, giving duplex a higher λ (e.g., 3× agent's weight). Alternatively, include duplex in the t-norm flat vector as a constrained output.

---

## Eval Pipeline Fix

The initial `eval_baseline_compat.py` had three critical issues that made f-mAP numbers non-comparable with the 3D-RetinaNet baseline. All prior Exp2 f-mAP numbers (ep8, ep15) are invalid.

### Bug 1: Single-frame inference (distribution shift)

The DETR decoder was trained on 8-frame clips (T×H'×W' = 2048 spatiotemporal tokens). The eval ran 1-frame "clips" (256 tokens). The learned query attention patterns are completely different at 1/8th context size.

**Fix:** Run proper 8-frame clips during eval. Extract per-frame boxes from the tube predictions (`pred_boxes[:, t, :]` for each frame t). Non-overlapping clips with padding for short video tails.

### Bug 2: Score double-gating

The eval computed `score = agentness × class_prob` for per-class detection scoring. The baseline (val.py) uses **raw per-class sigmoid** — `confidence[b, s, :, class_idx]` — with no agentness multiplication. The multiplication pushed all scores down, hurting AP ranking.

**Fix:** Use raw `probs[head]` for per-class scores. Agentness is only used for the `agent_ness` label type itself.

### Bug 3: Aggressive pre-filtering

`CONFIDENCE_THRESHOLD=0.3` on agentness removed ~60% of queries before per-class evaluation. `CLASS_THRESHOLD=0.05` on the already-multiplied scores further removed detections. The baseline's AP evaluator sweeps thresholds internally — pre-filtering removes detections the sweep needs.

**Fix:** Pass all 100 queries with `class_threshold=0.0`. Let `evaluate_frames()` handle the confidence sweep.

### Impact

The previous "124.3% agent_ness mAP" was not a normalization bug — it was 1.24% mAP (the JSON values from `evaluate_frames` are already in percentage form, but were incorrectly multiplied by 100 when written to CSV). All prior Exp2 f-mAP values in the wiki and CSV were 100× inflated.

---

## Related

- [[findings/exp1b-fcos-detection|Exp1b FCOS Detection]] — predecessor; 3.2% baseline f-mAP motivates this redesign
- [[findings/exp1-vs-retinanet-baseline|Exp1 vs RetinaNet Baseline]] — oracle-box starting point
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints (T-Norm)]] — t-norm loss theory
- [[papers/marconato-2022-road-r|Marconato 2022 — ROAD-R]] — Gödel t-norm (Table 7)
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline being compared against
- [[datasets/road-plusplus|ROAD++ Dataset]] — 86-class compositional labels with duplex/triplet constraints
- [[projects/road-reason|ROAD_Reason Project]] — parent project
