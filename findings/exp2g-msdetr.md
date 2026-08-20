---
type: finding
title: "Exp2g — MS-DETR: Two-Stage + O2M + Full Encoder Loss"
aliases: ["exp2g", "exp2g msdetr", "ms-detr experiment"]
created: 2026-05-26
updated: 2026-05-26
sources:
  - "ROAD_Reason/experiments/exp2g_msdetr/config.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/model.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/losses.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/deformable_decoder.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/train.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/matcher_o2m.py"
  - "ROAD_Reason/experiments/exp2g_msdetr/eval_baseline_compat.py"
tags: [finding, road-plusplus, exp2g, ms-detr, frozen-detr, resnet50, clip, detection, two-stage, o2m, thesis-core]
status: draft
---

# Exp2g — MS-DETR: Two-Stage + O2M + Full Encoder Loss

Implements the full MS-DETR recipe (Frozen-DETR paper) on top of [[findings/exp2f-flat-head|exp2f]], addressing the core remaining gap: **insufficient positive training signal**. With one-to-one Hungarian matching, only ~20 of 300 queries receive positive gradients per clip. MS-DETR fixes this with three additions: two-stage query initialization (encoder-generated proposals replace learned queries), one-to-many (O2M) auxiliary loss (6x more positive signals), and encoder proposal supervision.

**Status:** Training epoch 1/30 on GPU 0 (started 2026-05-26, third launch — see Bugs Fixed below).

---

## Motivation: Why Exp2f Plateaued

Exp2f fixed the missing-negative-supervision bug and jumped from 1.4% to 4.4% agent f-mAP. But by epoch 18, matched action mAP plateaued at 0.199 with loss barely moving (2.07 → 2.05 over 5 epochs). The fundamental limit: **query coverage**.

| | 3D-RetinaNet (baseline) | Exp2f (DETR) |
|---|---|---|
| Proposals per frame | ~100K dense anchors | 300 learned queries |
| Positive proposals per frame | ~500-1000 (IoU >= 0.5) | ~20 (Hungarian 1-to-1) |
| Negative supervision | Focal loss on all anchors | Focal loss on all queries (fixed in exp2f) |

Even with correct negative supervision, 300 queries with 1-to-1 matching produce ~20 positive signals per clip — 50x fewer than the baseline. MS-DETR closes this gap.

---

## What Changed from Exp2f

### 1. Two-Stage Query Initialization

**Problem:** Learned queries start with random positions and must discover where objects are through training. Slow convergence, poor coverage of rare spatial positions.

**Fix:** The encoder proposes candidate regions. Top-900 scoring proposals become decoder queries with spatially-informed positions. Mixed selection: learnable content embeddings + proposal-derived positional embeddings.

```
Encoder memory → Linear + LayerNorm → cls_heads[-1] (184-dim) + box_head (4-dim)
  → top-900 by dim-0 score → sinusoidal pos embed → pos_trans → query_pos
  → content: learned tgt_embed (mixed selection)
```

### 2. One-to-Many (O2M) Auxiliary Loss

**Problem:** Each GT tube matches exactly 1 query (Hungarian matching). ~20 GTs × 1 match = ~20 positive signals per clip.

**Fix:** O2M matching assigns k=6 queries per GT (IoU-threshold + top-k selection). The O2M branch taps features *after* cross-attention but *before* self-attention (aux FFN per decoder layer), so extra positive signals improve the encoder without disrupting the one-to-one decoder output used at inference.

- `Stage2AssignerTubes`: k=6, IoU threshold=0.4, cost = 0.7*IoU + 0.3*cls_score
- O2M loss: same focal + L1 + GIoU as O2O, weighted by LAMBDA_O2M_CLS=2, LAMBDA_O2M_BBOX=5, LAMBDA_O2M_GIOU=2

### 3. Encoder Proposal Loss

**Problem (original):** Encoder proposals had a separate 1-dim binary objectness head (`enc_class_head = nn.Linear(d_model, 1)`) with IoU-threshold matching. This head was not learning — loss stuck at ~0.98 for 3 full epochs.

**Fix (2026-05-26):** Replaced with the reference's exact approach:
- Encoder cls head is the **last clone of `cls_heads`** (full 184-dim, same as decoder)
- **Binary targets** (`bin_targets`): copies GT tubes but zeros all labels → single-class "is there an object?" detection
- **Hungarian matching** (same matcher as decoder, not IoU-threshold)
- **Same loss functions** as decoder (focal + L1 + GIoU)

### 4. Softmax Agent Head

Agent type is strictly single-label in ROAD++ (every agent has exactly 1 type). Exp2g uses softmax CE on agent dims [1:11] for matched queries, sigmoid focal on all other dims. This respects the mutual exclusivity that sigmoid ignores.

### 5. Additional Reference Replication

Systematic audit against Frozen-DETR reference repo (2026-05-24) fixed multiple gaps:

| Gap | Before | After (reference-matched) |
|-----|--------|--------------------------|
| Reference points | 2D (cx, cy) | **4D (cx, cy, w, h)** — enables box-proportional sampling offsets |
| Box refinement | 2D center only | **4D iterative** — all coords refined per decoder layer |
| valid_ratios | Not implemented | **Full pipeline** — scales reference points per feature level |
| Padding masks | Not implemented | **Full pipeline** — masks padded positions in MSDeformAttn |
| LAMBDA_CLS | 2.0 | **1.0** (reference: `--cls_loss_coef 1`) |
| look_forward_twice | N/A | Already correct — non-detached pred_box in aux outputs |

---

## Architecture

```
PIL frames (8 per clip, 1920x1280 original → multi-scale train/val)
    |
    +--> R50 (FrozenBN) + FPN → P3/P4/P5
    |
    +--> CLIP ViT-L/14@336 (frozen) → patch tokens (4th encoder scale) + CLS token
    |
    +--> Deformable Encoder (6L, 4 scales: P3+P4+P5+CLIP)
    |         +--> strip CLIP tokens
    |         +--> gen_encoder_output_proposals → cls_heads[-1] + box_head
    |         +--> top-900 proposals → pos_trans → query_pos
    |         +--> encoder loss: Hungarian + focal + L1 + GIoU (bin_targets)
    |
    +--> MS-DETR Decoder (6L, 3 scales, CLS injection)
    |    O2O path: cross-attn → self-attn → temporal → FFN
    |    O2M path: cross-attn → aux_FFN (per layer, before self-attn mixes queries)
    |    --> 900 queries × T frames → boxes + features
    |    --> per-layer aux outputs (O2O + O2M)
    |
    +--> cls_heads[0:6]: 6× nn.Linear(256, 184) for decoder layers
    |    cls_heads[6]: encoder proposal head (same architecture)
    |    cls_heads_o2m[0:5]: 6× nn.Linear(256, 184) for O2M branch
    |
    +--> Softmax on agent dims [1:11] (matched only)
    |    Sigmoid focal on remaining 174 dims (all queries)
```

**358.7M total params | 54.2M trainable | 304.5M frozen (CLIP)**

---

## Loss Components (3-Stream)

| Stream | Components | Matching | Queries |
|--------|-----------|----------|---------|
| **O2O** (main) | focal(184-dim) + L1 + GIoU + t-norm | Hungarian 1-to-1 | 900 |
| **O2M** (auxiliary) | focal(184-dim) + L1 + GIoU | O2M top-k=6 per GT | 900 |
| **Encoder** | focal(184-dim, bin_targets) + L1 + GIoU | Hungarian 1-to-1 | N_spatial (~14K) |

Loss coefficients (all matched to reference training script `train_ms_detr_pp_900.sh`):

| Coefficient | O2O | O2M | Encoder |
|------------|-----|-----|---------|
| cls | 1.0 | 2.0 | 2.0 |
| bbox | 5.0 | 5.0 | 5.0 |
| giou | 2.0 | 2.0 | 2.0 |

---

## Training Details

| Setting | Value | Source |
|---------|-------|--------|
| Backbone LR | 2e-5 | Paper |
| Encoder/Decoder LR | 2e-4 | Paper (`--lr 2e-4`) |
| Deformable LR | 2e-5 (0.1x mult) | Paper |
| Heads LR | 2e-4 | Paper |
| Dropout | 0.0 | Paper |
| FFN dim | 2048 | Paper |
| Grad clip | 0.1 | Paper |
| Grad accumulation | 4 | Ours (effective batch=4) |
| DINO warm-start | 461 keys (R50 + encoder + decoder + enc_output/norm) | |
| Max epochs | 30 | |

---

## Early Training Signal

### Run 3 (current — with full encoder head fix)

| Clip | Total | cls | bbox | giou | tnorm | o2m | enc |
|------|-------|-----|------|------|-------|-----|-----|
| 25 | 18.25 | 0.148 | 0.072 | 1.471 | 0.024 | 9.67 | 1.43 |
| 500 | 14.62 | 0.039 | 0.058 | 1.179 | 0.003 | 7.74 | 1.34 |
| 1250 | 13.40 | 0.044 | 0.063 | 1.127 | 0.005 | 6.84 | 1.33 |

Encoder loss now computing over 184 dims with Hungarian matching — starting at 1.43 and decreasing (vs 0.98 flat with old binary head).

### Run 2 (epochs 1-3, before encoder head fix)

| Epoch | Train Loss | Val Loss | Val Matched Action mAP |
|-------|-----------|----------|----------------------|
| 1 | 9.65 | 8.35 | 0.1044 |
| 2 | 7.34 | 6.78 | 0.1193 |
| 3 | 6.49 | 6.22 | 0.1387 |

Val localization improving rapidly:

| Epoch | Val GIoU | Val Bbox L1 |
|-------|----------|------------|
| 1 | 0.801 | 0.033 |
| 2 | 0.698 | 0.019 |
| 3 | 0.615 | 0.016 |

For comparison, exp2f's matched action mAP didn't reach 0.14 until around epoch 5-6. Exp2g hit it in 3 epochs.

---

## Bugs Fixed During Implementation

### 1. GIoU OOM (Run 1, clip 6700/7027)

The original `_encoder_loss` computed `generalized_box_iou(pos_pred_xyxy, pos_gt_xyxy)` which builds an N_pos x N_pos matrix. As training progressed and more proposals crossed the IoU >= 0.5 threshold, N_pos grew into thousands. With N_pos ~ 10K, the matrix allocation (~400MB) exceeded remaining GPU memory (45.8GB already in use).

**Fix:** Replaced with pairwise GIoU computation (O(N) instead of O(N^2)). Later superseded by the full encoder head rewrite which reuses `_box_losses` on small matched-only matrices.

### 2. Binary Encoder Head Not Learning (Runs 1-2, all epochs)

The 1-dim `enc_class_head` used IoU-threshold matching with focal loss on binary objectness. Loss stuck at ~0.98 (epoch 1) → 0.89 (epoch 4) — barely moving. The massive class imbalance (~900 proposals, ~20 positives) with a 1-dim head produced too weak a gradient signal.

**Root cause:** Deviated from reference. Frozen-DETR/MS-DETR uses the same full classification head (last clone of `class_embed`) for encoder proposals, with `bin_targets` (all GT labels set to class 0) and standard Hungarian matching. Not a separate binary head.

**Fix (2026-05-26):** Removed `enc_class_head`. Added extra clone to `cls_heads` (7 total: 6 decoder + 1 encoder). Rewrote `_encoder_loss` to use Hungarian matching + bin_targets + same loss functions as decoder. Verified with CPU smoke test: all three encoder loss components finite and reasonable.

---

## Results

| Epoch | Train Loss | Val Loss | Val Matched Action mAP | f-mAP Agent | Notes |
|-------|-----------|----------|----------------------|------------|-------|
| 1 | 9.65 | 8.35 | 0.1044 | TBD | Run 2 (old encoder head) |
| 2 | 7.34 | 6.78 | 0.1193 | TBD | |
| 3 | 6.49 | 6.22 | 0.1387 | TBD | Best before encoder fix |
| TBD | | | | | Run 3 (full encoder head) |

**Success criteria:** Close the gap with 3D-RetinaNet baseline (17.76% agent f-mAP). Exp2f reached 4.4% at epoch 14. If O2M + two-stage deliver the expected 2-4x improvement, exp2g should reach 10-15%.

---

## Related

- [[findings/exp2f-flat-head|Exp2f Flat Head]] — predecessor; fixed negative supervision; 4.4% agent f-mAP
- [[findings/exp2-series-narrative|Exp2 Series Narrative]] — full series story
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline target (17.76% agent f-mAP)
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — reference architecture paper
- [[projects/road-reason|ROAD_Reason Project]] — parent project
