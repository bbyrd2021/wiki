---
type: finding
title: "Exp2 Series — From DETR Failure to Baseline-Compatible Detection"
aliases: ["exp2 narrative", "exp2 series", "detection narrative"]
created: 2026-05-20
updated: 2026-05-28
sources:
  - "ROAD_Reason/experiments/exp2_detr_qwen/"
  - "ROAD_Reason/experiments/exp2b_efficientnet_detr/"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/"
  - "ROAD_Reason/experiments/exp2d_swin_detr/"
  - "ROAD_Reason/experiments/exp2e_r50_detr/"
  - "ROAD_Reason/experiments/exp2f_flat_head/"
  - "ROAD_Reason/experiments/exp2g_msdetr/"
  - "ROAD_Reason/results/val_metrics.csv"
tags: [finding, road-plusplus, detection, detr, narrative, thesis-core]
status: draft
---

# Exp2 Series — From DETR Failure to Baseline-Compatible Detection

This document tells the story of the exp2 experiment series: six iterations of DETR-based detection on ROAD-Waymo, each motivated by a specific diagnosis of the previous failure. The series began with 0.63% agent f-mAP (28x below the 3D-RetinaNet baseline) and systematically identified two root causes — one architectural, one in the loss function — that explain the full gap.

The narrative matters for the thesis because this detection pipeline is the foundation for Approach 4 (OpenMixer + DSDAG causal reasoning). Every lesson learned here informs the final architecture design.

---

## Architecture Evolution at a Glance

### Backbone & Foundation Model per Experiment

| Exp | CNN Backbone | Foundation Model | Resolution | Queries | Frozen-DETR? | DINO Pretrained? |
|-----|-------------|-----------------|------------|---------|--------------|-----------------|
| **exp2** | None | Qwen2.5-VL-7B ViT (LoRA) | 448×448 | 100 | No | No |
| **exp2b** | EfficientNet-B0 + FPN | Qwen2.5-VL-7B ViT (LoRA) | 448×448 | 300 | No | No |
| **exp2c** | EfficientNet-B0 + FPN | **CLIP ViT-L/14@336** (frozen) | 448×448 | 300 | **Yes — first** | No |
| **exp2d** | **Swin-L@384** | CLIP ViT-L/14@336 (frozen) | 384×384 | 300 | Yes | Yes (192 keys) |
| **exp2e** | **ResNet-50** | CLIP ViT-L/14@336 (frozen) | **800×1333 (var)** | 300 | Yes | Yes (457 keys) |
| **exp2f** | ResNet-50 | CLIP ViT-L/14@336 (frozen) | 800×1333 (var) | 300 | Yes | Yes (457 keys) |
| **exp2g** | ResNet-50 | CLIP ViT-L/14@336 (frozen) | 800×1333 (var) | **900** | Yes + MS-DETR | Yes (457 keys) |

### Classification Head Evolution

| Exp | Head Design | Negative Supervision | Agent Activation |
|-----|------------|---------------------|-----------------|
| **exp2–exp2e** | 6 separate heads: agentness `Linear(256,1)` + agent `Linear(256,10)` + action `Linear(256,22)` + loc `Linear(256,16)` + duplex `Linear(256,49)` + triplet `Linear(256,86)` | **Broken** — focal loss on matched queries only; unmatched queries get zero gradient on 5 classification heads; agentness head covers only 1/184 dims | All sigmoid |
| **exp2f** | Single flat `Linear(256,184)` — layout: `[agentness(1) \| agent(10) \| action(22) \| loc(16) \| duplex(49) \| triplet(86)]` | **Fixed** — focal loss on ALL queries × ALL 184 dims; unmatched get explicit all-zeros target | All sigmoid |
| **exp2g** | Same flat `Linear(256,184)` + separate O2M heads per decoder layer | Fixed (same as exp2f) | **Softmax on agent dims** (single-label); sigmoid on rest |

### Input/Output Specification (all experiments)

**Input:** 8 RGB frames from a ROAD-Waymo video clip (resolution varies per experiment — see table above).

**Output per query (×N queries):**
- **Tube boxes:** 8×4 — one `[cx, cy, w, h]` bounding box per frame (sigmoid, normalized 0-1)
- **Agentness:** 1 scalar — "is this query an object or background?"
- **Agent:** 10 logits — Car, Ped, Cyclist, LarVeh, MedVeh, etc. (single-label)
- **Action:** 22 logits — Moving, Waiting, Xing, etc. (multi-label)
- **Location:** 16 logits — NearLeft, FarRight, etc. (multi-label)
- **Duplex:** 49 logits — valid agent×action pairs (multi-label)
- **Triplet:** 86 logits — valid agent×action×location triples (multi-label)

At inference: filter by agentness confidence, take top detections. In exp2f+, agentness is slot 0 of the flat 184-dim vector rather than a separate head.

### When Each Architecture Was Adopted

- **exp2:** Vanilla DETR decoder (standard cross-attention, no deformable attention, no encoder)
- **exp2b:** Deformable DETR decoder (Zhu et al., ICLR 2021) — added deformable attention, iterative refinement, auxiliary losses. No encoder — CNN-VLM fusion via scalar gates. No single published paper; assembled from standard Deformable DETR components.
- **exp2c:** **Frozen-DETR adopted here** (Fu et al., NeurIPS 2024) — added 6-layer deformable encoder with 4-scale bidirectional self-attention (P3+P4+P5+CLIP patches). CLIP tokens stripped after encoding; CLS injected per-layer into decoder. This architecture persists through exp2d–exp2g.
- **exp2g:** Added MS-DETR recipe on top of Frozen-DETR — two-stage encoder proposals, O2M auxiliary loss, encoder proposal supervision. These are the paper's core contributions that were deferred from exp2c.

---

## The Starting Point: Why DETR?

Exp1b (FCOS dense detection on Qwen2.5-VL ViT features) achieved strong internal classification — 60.6% agent macro-mAP on foreground tokens — but only 3.2% baseline-compatible f-mAP. The problem was box quality: single-scale ViT features at 448x448 couldn't produce tight bounding boxes at IoU>=0.5. FCOS assigns every spatial token independently, with no mechanism to refine box predictions across the feature map.

DETR offered a fundamentally different approach: learnable queries attend globally to all features, Hungarian matching enforces one-to-one assignment (no NMS), and L1 + GIoU loss directly supervises box coordinates. The promise was that global attention + direct box supervision would fix the localization bottleneck.

---

## Exp2: First DETR Attempt (Apr 2026)

**Architecture:** 100 learned queries, Qwen2.5-VL ViT (frozen + LoRA) as sole backbone, 6-layer vanilla Transformer decoder, 6 sigmoid heads (5 classification + agentness), Hungarian matching, focal loss on matched queries only.

```
8 RGB frames (448×448)
  → Qwen2.5-VL ViT (frozen + LoRA on 8 blocks)   [T=8, 16×16 patches, D=3584]
  → Linear projection (3584 → 256)
  → Spatiotemporal position encoding
  → Vanilla DETR Decoder (6 layers, standard cross-attention)
      100 learnable queries attend to all T×H'×W' = 2048 tokens
      self-attn → cross-attn → FFN per layer
  → Per-query: box_head → T×4 tube, 6 separate sigmoid heads
  → Hungarian matching + focal (matched only) + L1 + GIoU + Gödel t-norm
```

No CNN backbone. No multi-scale features. No deformable attention. No iterative refinement. Single-scale 3584-dim ViT features projected to 256-dim for a vanilla decoder.

**Result:** 0.63% agent f-mAP — worse than FCOS.

**Diagnosis at the time:** The decoder lacked three standard Deformable DETR components (per-frame decoding, iterative box refinement, auxiliary losses), and the ViT provided only single-scale features. We attributed the failure to architectural incompleteness.

**What we didn't know yet:** The loss design — focal loss on matched queries only — was the dominant problem. But with so many other issues, it was invisible.

---

## Exp2b: Deformable DETR + EfficientNet (Apr–May 2026)

**What changed:** Added all three missing Deformable DETR components (per-frame decoding with temporal self-attention, iterative box refinement with per-layer box heads, auxiliary decoder losses). Added EfficientNet-B0 + FPN as a spatial backbone alongside the Qwen ViT, fused via learned scalar gates. Increased to 300 queries. No single reference paper — assembled from standard Deformable DETR (Zhu et al., ICLR 2021) components.

```
8 RGB frames (448×448)
  → Qwen2.5-VL ViT (frozen + LoRA)       → 3584-dim semantic features
  → EfficientNet-B0 (partially frozen)    → C3/C4/C5 multi-scale
  → FPN (256-dim) ← scalar gate fusion from Qwen features
      gate = sigmoid(learnable scalar, init=0.1)
      fused = gate × vlm_proj + (1-gate) × cnn
  → Deformable DETR Decoder (6 layers, 4 sampling points, 3 FPN scales)
      300 learnable queries
      self-attn → deformable cross-attn → temporal self-attn → FFN
      iterative box refinement per layer
      auxiliary outputs from every layer
  → Per-query: tube boxes + 6 separate sigmoid heads (same as exp2)
```

**Result:** 1.71% agent f-mAP — 2.7x better than exp2, still 10x below baseline.

**Diagnosis at the time:** Recall was high (48-62% across all heads) — the model found agents — but precision at IoU>=0.5 was terrible. We concluded that "VLM features are fundamentally unsuitable as detection backbones" and that the scalar gate fusion was too simple.

**What we didn't know yet:** Same loss bug. The improved f-mAP came from better architecture (deformable attention, iterative refinement), not from fixing the scoring problem.

---

## Exp2c: Frozen-DETR Architecture (May 2026) — Architecture Transition

**This is where we adopted the Frozen-DETR architecture** (Fu et al., NeurIPS 2024). Two major changes:

1. **Replaced scalar gate fusion with a 6-layer deformable encoder.** CNN and CLIP tokens attend to each other through bidirectional multi-scale self-attention across 4 scales (P3+P4+P5+CLIP patches). After encoding, CLIP tokens are stripped — the decoder only sees CNN-origin tokens, now enriched with CLIP semantics.

2. **Replaced Qwen2.5-VL with CLIP ViT-L/14@336px** (Dr. Moradi approved, May 7). Saves ~5 GB GPU and provides clean separation of CLS token (global semantic summary) + patch tokens (spatial features). No LoRA — CLIP is fully frozen.

```
8 RGB frames (448×448 for EfficientNet, 336×336 for CLIP)
  → EfficientNet-B0 → FPN → P3/P4/P5 (3 spatial scales, 256-dim)
  → CLIP ViT-L/14 (fully frozen) → patch tokens (4th encoder scale) + CLS token
  → Deformable Encoder (6 layers, 4 scales: P3+P4+P5+CLIP patches)
      each token attends to K=4 learned offset points per head per scale
      bidirectional: CNN tokens absorb CLIP semantics, CLIP tokens absorb CNN spatial info
  → Strip CLIP tokens from encoder output
  → Deformable Decoder (6 layers, 3 scales: P3+P4+P5 only)
      300 learnable queries
      CLIP CLS injected per-layer as "image query" (global semantic context)
      deformable cross-attn + temporal self-attn + iterative box refinement
  → Per-query: tube boxes + 6 separate sigmoid heads (same structure as exp2/2b)
```

This architecture — deformable encoder (4 scales) → strip CLIP → deformable decoder (3 scales) with CLS injection — persists through all subsequent experiments (exp2d–exp2g). No DINO COCO pretraining yet; encoder/decoder trained from scratch (warm-started from exp2b).

**Result:** 1.76% agent f-mAP. Internal classification was strong — val action mAP peaked at 43.72% (ep23). GIoU steadily improving (0.793 → 0.538 over 23 epochs).

**Diagnosis at the time:** The encoder fusion was clearly better than the scalar gate (richer internal metrics), but f-mAP barely moved. We noticed that the paper used R50 at 800x1333 while we were running EfficientNet-B0 at 448x448. We hypothesized **input resolution** was the bottleneck: at 448px, a 16px pedestrian becomes 3.7px — sub-pixel at FPN stride 8.

**What we didn't know yet:** Resolution was real but secondary. Same loss bug still the dominant factor.

---

## Exp2d: Swin-L Backbone (May 2026)

**What changed:** Swapped EfficientNet-B0 (~5.3M params) for Swin-L@384 (~195M params) — 37x larger backbone. Same Frozen-DETR architecture otherwise. First experiment to use **DINO COCO pretrained encoder/decoder** (192 keys transferred; Swin-L architecture doesn't match R50, so only encoder/decoder weights, not backbone).

```
8 RGB frames (384×384 for Swin-L, 336×336 for CLIP)
  → Swin-L@384 (patch_embed + layers 0-1 frozen, layers 2+ trainable)
      → C3/C4/C5 (384/768/1536 channels) → FPN → P3/P4/P5 (256-dim)
  → CLIP ViT-L/14 (frozen) → patch tokens + CLS
  → Deformable Encoder (6 layers, 4 scales) — DINO COCO pretrained
  → Strip CLIP → Deformable Decoder (3 scales, 300 queries) — DINO COCO pretrained
  → Per-query: tube boxes + 6 separate sigmoid heads
```

**v1:** Zero augmentation. Val action mAP 44.55% at epoch 2 (beating exp2c's 43.72% at epoch 23) — fast learning. Overfitted by epoch 3. **v2:** Added DETR augmentations + strong color aug + drop path 0.2 + weight decay 0.05.

**f-mAP:** 1.67% agent — no improvement over exp2c despite a 37x larger backbone.

**Diagnosis at the time:** The backbone wasn't the bottleneck. Swin-L at 384x384 had the same resolution problem as EfficientNet at 448x448. This ruled out backbone capacity and pointed squarely at resolution.

**Lesson learned:** Replicate the reference paper's exact config before swapping components. We had been changing backbones without matching the paper's resolution — the wrong variable.

---

## Exp2e: R50 at Paper Resolution (May 2026)

**What changed:** Faithful replication of the Frozen-DETR paper's exact spatial config. Two changes from exp2d:

1. **ResNet-50** replaces Swin-L — matches the DINO COCO checkpoint architecture exactly. **457 keys transferred** (backbone + encoder + decoder — the full pipeline pretrained on COCO) vs 192 for Swin-L.

2. **Variable aspect ratio at 800×1333** — the paper's actual resolution. Train multi-scale [480..800] short side, val 800×max1333. Previous experiments were all 384-448px fixed square.

```
8 RGB frames (variable aspect ratio, up to 800×1333)
  → ResNet-50 (FrozenBatchNorm, DINO COCO pretrained) → C3/C4/C5 → FPN → P3/P4/P5
  → CLIP ViT-L/14@336 (frozen) → patch tokens + CLS
  → Deformable Encoder (6 layers, 4 scales) — DINO COCO pretrained (457 keys)
  → Strip CLIP → Deformable Decoder (3 scales, 300 queries) — DINO COCO pretrained
  → Per-query: tube boxes + 6 separate sigmoid heads
```

**Result:** 5.54% agentness f-mAP — a ~4x improvement over exp2c/2d. Resolution was real.

**But:** Agent f-mAP was still only 1.40%. And recall was high (29-38% across heads). Something deeper was wrong.

**The breakthrough diagnostic:** We ran a direct IoU analysis bypassing the evaluator:

- 38.7% of GT boxes matched at least one query at IoU>=0.5 across all 300 queries — **the model could localize**
- Top-20 Car detections: all had score > 0.997, all had IoU < 0.25 with any GT box
- The scoring system was broken, not the localization

We traced it to the loss:

```
_classification_loss:  applied to matched queries only (~20/300)
_agentness_loss:       applied to all queries, but only the 1-dim agentness head
```

The 280 unmatched queries received **zero gradient on all 5 classification heads** (agent, action, loc, duplex, triplet). Their class scores were never pushed toward 0. Over training, they drifted to near-saturation (>0.99), dominating the ranked detection list with high-scoring, poorly-localized junk.

**Why standard DETR doesn't have this problem:** Standard DETR uses softmax with a "no-object" class. The softmax target on unmatched queries naturally suppresses all real class logits. Our sigmoid heads (required for multi-label) needed explicit negative targets but didn't have them. The agentness head was our version of "no-object," but it only covered 1 of 184 dims.

**Why the baseline doesn't have this problem:** 3D-RetinaNet uses a single flat 184-dim sigmoid vector with focal loss on ALL anchors. Negative anchors get target = all-zeros across all 184 dims. Every class score on every background prediction is explicitly suppressed.

---

## Exp2f: The Fix — Flat 184-dim Head (May 2026)

**What changed:** One change — replaced the 6 separate classification heads with a **single `nn.Linear(256, 184)`**, the 3D-RetinaNet baseline's exact design. Everything else identical to exp2e (same R50, same CLIP, same encoder/decoder, same resolution).

```
Layout: [agentness(1) | agent(10) | action(22) | loc(16) | duplex(49) | triplet(86)] = 184

All 300 queries × all 184 dims:
  Matched queries   → GT labels as target
  Unmatched queries → all-zeros as target (explicit negative supervision)
  Focal loss on everything
```

```
8 RGB frames (variable aspect ratio, up to 800×1333)
  → ResNet-50 → FPN → P3/P4/P5
  → CLIP ViT-L/14@336 (frozen) → patch tokens + CLS
  → Deformable Encoder (6 layers, 4 scales) → Strip CLIP
  → Deformable Decoder (3 scales, 300 queries)
  → Single nn.Linear(256, 184) → sigmoid → focal loss on ALL queries
```

No separate agentness head — agentness is slot 0 in the flat vector.

**Result:** Classification loss dropped **26x in 2 epochs** — unmatched queries immediately learned to predict ~0 across all 184 dims. Baseline-compatible f-mAP at epoch 14: **agent 4.40%** — a 3x improvement over exp2e. Best matched action mAP: 0.199 (epoch 18). Final best matched action mAP: 0.209 (epoch 30).

**Epoch 30 f-mAP (full training run):**

| Task | ep14 | ep30 | Δ |
|------|------|------|---|
| Agent-ness | 5.82% | **13.38%** | +7.56 |
| Agent | 4.40% | **5.51%** | +1.11 |
| Action | 3.28% | **4.20%** | +0.92 |
| Location | 2.04% | **3.96%** | +1.92 |
| Duplex | 0.62% | **1.11%** | +0.49 |
| Triplet | 1.16% | **1.72%** | +0.56 |

Continued training from ep14→30 yielded significant improvement across all heads, with agent-ness more than doubling. Constraint violations at IoU 0.5: duplex 0.62% (very low), triplet 80.09% (still high — triplet combinatorics remain hard).

**But:** Agent f-mAP at 5.51% is still 3.2x below the baseline's 17.8%. The remaining gap: **query coverage**. 300 queries with one-to-one matching produce ~20 positive signals per clip vs the baseline's ~500-1000 from dense anchors.

---

## Exp2g: MS-DETR — Two-Stage + O2M + Encoder Loss (May 2026, in progress)

**What changed:** Implements the full MS-DETR recipe from the Frozen-DETR paper to close the query coverage gap. Four additions on top of exp2f:

1. **Two-stage query init:** Instead of 300 learned queries, the encoder generates region proposals. Top-900 scoring proposals become decoder queries with spatial priors — queries start where objects actually are. Mixed selection: learnable content embeddings + proposal-derived position embeddings.

2. **One-to-many (O2M) loss:** Each GT matches k=6 queries (not just 1). The O2M branch uses auxiliary FFN heads per decoder layer, tapping features after cross-attention but before self-attention mixes queries. 6x more positive training signals per clip (~120 vs ~20).

3. **Encoder proposal loss:** Full 184-dim classification head (last clone of decoder's `cls_heads`) with binary targets + Hungarian matching + standard focal/L1/GIoU. Matched proposals get agentness=1, all other dims zeroed (bin_targets pattern). Matches the Frozen-DETR reference exactly.

4. **Softmax agent head:** Agent type is strictly single-label (a Car can't be a Ped); softmax CE on agent dims [1:11] for matched queries. Rest stays sigmoid (multi-label).

```
8 RGB frames (variable aspect ratio, up to 800×1333)
  → ResNet-50 → FPN → P3/P4/P5
  → CLIP ViT-L/14@336 (frozen) → patch tokens + CLS
  → Deformable Encoder (6 layers, 4 scales)
      │
      ├── Encoder proposals: cls_heads[-1](memory) → 184-dim per spatial token
      │   Top-900 by agentness score → decoder queries
      │   pos_trans(sinusoidal_embed(proposals)) → query position embeddings
      │   Encoder loss: Hungarian matching + focal + L1 + GIoU
      │
  → Strip CLIP → Deformable Decoder (6 layers, 3 scales, 900 queries)
      Each layer: cross-attn → [aux FFN for O2M branch] → self-attn → temporal-attn → FFN
      O2O path: features after full pipeline → cls_heads → 184-dim (1:1 matching)
      O2M path: features after cross-attn only → cls_heads_o2m → 184-dim (1:k matching, k=6)
      4D reference points [cx, cy, w, h] with iterative refinement
  → Outputs: O2O predictions + O2M predictions + encoder proposals
```

**358.7M total params | 54.2M trainable | 304.5M frozen (CLIP)**

**Implementation notes:** Required systematic audit against the Frozen-DETR reference repo. Fixed 4D reference points (was 2D), added valid_ratios + padding mask pipeline, corrected LAMBDA_CLS from 2.0 to 1.0 (matching `--cls_loss_coef 1` in reference training script). Three bugs found and fixed across three training runs:

- **Run 1:** GIoU OOM — N² matrix allocation in encoder loss
- **Run 2:** Binary encoder head (1-dim objectness) not learning — stuck at ~0.98 loss. Replaced with full 184-dim cls_heads[-1] + Hungarian matching + bin_targets, matching the reference exactly.
- **Run 3 (current):** Fixed LAMBDA_CLS from 2.0 to 1.0. Encoder loss actively decreasing (1.16 → 1.10). Epoch 3 val matched action mAP = 0.1018. Training ongoing (2026-05-27).

---

## The Three Root Causes

Looking back, the entire exp2 series was chasing three problems, identified in order:

### 1. Input Resolution (identified exp2d, fixed exp2e)

At 384-448px, small objects become sub-pixel and can't achieve IoU>=0.5. The paper runs at 800x1333. This explains the ~4x improvement from exp2c/2d (~1.4%) to exp2e (5.5% agentness). Fixed in exp2e by matching the paper's exact resolution config.

### 2. Missing Negative Supervision (identified exp2e, fixed exp2f)

Sigmoid classification heads on unmatched queries received zero gradient. Scores drifted to saturation, polluting the ranked detection list. Present in every experiment from exp2 through exp2e — the loss design was copied forward unchanged while we changed architecture around it. Fixed in exp2f by replacing 6 separate heads with a single flat 184-dim sigmoid vector + focal loss on ALL queries (matched get GT labels, unmatched get all-zeros).

The reason it took 5 experiments to find: we were changing too many variables (backbone, fusion method, resolution) while the loss was constant. Each architectural improvement produced marginal f-mAP gains (0.63% → 1.71% → 1.76%) that looked like progress rather than symptoms. It wasn't until exp2e — where we finally matched the paper's config exactly and still got 1.4% agent f-mAP — that we ran the direct IoU diagnostic that revealed scores and localization were completely decorrelated.

### 3. Query Coverage (identified exp2f, being addressed in exp2g)

300 queries with one-to-one Hungarian matching produce ~20 positive training signals per clip. The 3D-RetinaNet baseline uses dense anchors producing ~500-1000 positives. Even with correct scoring (exp2f), 5.51% agent f-mAP at epoch 30 is still 3.2x below the baseline's 17.8% — there simply aren't enough positive examples to train the decoder effectively. Exp2g addresses this with two-stage proposals (900 queries with spatial priors) + O2M loss (k=6 matches per GT = ~120 positives per clip).

### 4. CLIP Encoder Fusion Mismatch (identified exp2g, 2026-05-28)

A structural problem present in all Frozen-DETR experiments (exp2c–2g): CLIP ViT-L/14@336 patch tokens are injected as a 4th scale level in the deformable encoder alongside FPN P3/P4/P5. Two issues:

1. **Spatial mismatch:** R50 sees frames at 480–800px with preserved aspect ratio. CLIP sees them squashed to 336×336 square. The deformable attention treats CLIP patches as spatial positions, but those positions don't correspond to the FPN features' spatial grid — coordinates are distorted.

2. **Frozen encoder can't adapt:** The DINO COCO pretrained encoder was trained on 3 FPN levels from R50. CLIP tokens as a 4th level are out-of-distribution. Since the encoder is frozen, it cannot learn to use them — they are effectively noise in the multi-scale attention.

Frozen-DETR partially mitigates this by stripping CLIP tokens before the decoder and before two-stage proposal generation (exp2g). But the encoder still mixes them through deformable attention, and the mismatch persists there.

**Implication:** Rather than rebuilding detection with CLIP fused at the encoder level, a stronger approach is **late fusion** — keep a proven detector (3D-RetinaNet) frozen, extract CLIP features per detected tube via RoI pooling, and fuse semantically through cross-attention (Q-Former). Each frozen model operates in its native domain; fusion happens in a trainable module that can learn the cross-modal relationship.

---

## Summary Table

| Experiment | Key Change | Agent f-mAP | What We Learned |
|------------|-----------|-------------|-----------------|
| Exp1b (FCOS) | Dense detection baseline | 3.20% | Box quality bottleneck from single-scale ViT |
| **Exp2** | DETR queries + Hungarian matching | 0.63% | Missing standard deformable components |
| **Exp2b** | Deformable DETR + EfficientNet + FPN | 1.71% | Scalar gate fusion too simple; high recall, poor precision |
| **Exp2c** | Frozen-DETR encoder + CLIP ViT-L/14 | 1.76% | Encoder fusion works; resolution suspected |
| **Exp2d** | Swin-L backbone (37x larger) | 1.67% | Backbone isn't the bottleneck; resolution confirmed |
| **Exp2e** | R50 @ 800x1333 (paper's exact config) | 1.40% (agent) / 5.54% (agentness) | Resolution helps but scoring is broken |
| **Exp2f** | Flat 184-dim head + focal on all queries | **5.51%** (ep30) | Fix for missing negative supervision |
| **Exp2g** | MS-DETR: two-stage + O2M + encoder loss | 0.06% (ep4) | 900 queries, 6x positive signals; Run 4 training with LR fix in progress |

*All f-mAP values at IoU=0.5 on ROAD-Waymo val set. Baseline 3D-RetinaNet-I3D: 17.76% agent.*

---

## What Comes Next

**Exp2g (in progress):** Full MS-DETR implementation — the paper's core contributions (two-stage + O2M + encoder loss) plus softmax agent head. Run 4 training with fresh 5-group optimizer on GPU 0 (epoch 5/30 in progress, 2026-05-28). Epoch 4 eval (Run 3 checkpoint): 0.06% agent f-mAP — the two-stage/O2M components hadn't converged with the original optimizer. Run 4 introduces LR_FRESH=1e-3 for random-init components (pos_trans, enc_box_head, cls_heads_o2m). See [[findings/exp2g-msdetr|Exp2g finding page]].

**New direction — 3D-RetinaNet + VLM enrichment:** The exp2 series revealed a fundamental architectural limitation: CLIP features fused at the encoder level suffer from spatial mismatch and the frozen encoder can't adapt to them (Root Cause #4). Rather than continuing to rebuild detection from scratch, the emerging approach is:

1. **Keep 3D-RetinaNet frozen** — it already achieves 17.76% agent f-mAP with proven localization
2. **CLIP ViT (frozen)** provides semantic features, RoI-pooled at detected tube locations
3. **Q-Former cross-attention** fuses tube features with CLIP features in a trainable module
4. **Classification heads** on the enriched embeddings handle agent, action, loc, duplex, triplet
5. **Future expansion:** the Q-Former output embeddings are language-aligned (via CLIP) and can serve as visual tokens for a language decoder — enabling driving explanation generation (connects to BDD-X work in exp3)

This is a late-fusion architecture: each frozen model operates independently, fusion happens where trainable parameters can learn the cross-modal relationship, and the design is forward-compatible with language generation. Current ROAD++ SOTA is YOLOv8; RF-DETR (Roboflow, 2026, Apache 2.0) beats YOLOv8 on COCO with a DINOv2+deformable-DETR architecture and could serve as an even stronger detection backbone.

**Approach 4 — OpenMixer integration:** Still viable as an alternative. OpenMixer (Bao et al., WACV 2025) replaces the Frozen-DETR pipeline with its own detection on frozen CLIP-ViP features. The flat-head negative supervision fix and MS-DETR recipe are modular additions that could be grafted onto either approach.

---

## Related

- [[findings/exp2-detr-detection|Exp2]] | [[findings/exp2b-deformable-detr|Exp2b]] | [[findings/exp2c-frozen-detr|Exp2c]] | [[findings/exp2d-swin-detr-v2|Exp2d]] | [[findings/exp2e-r50-frozen-detr|Exp2e]] | [[findings/exp2f-flat-head|Exp2f]] | [[findings/exp2g-msdetr|Exp2g]] — individual experiment pages
- [[findings/exp1b-fcos-detection|Exp1b FCOS]] — predecessor with the different localization problem
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline target (17.76% agent f-mAP)
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — reference architecture
- [[directions/constrained-vlm-reasoning|Approach 4]] — where this detection pipeline is headed
- [[projects/road-reason|ROAD_Reason Project]] — parent project
