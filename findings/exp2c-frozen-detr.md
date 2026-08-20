---
type: finding
title: "Exp2c — Frozen-DETR: EfficientNet-FPN + DETR Encoder + CLIP ViT-L/14"
aliases: ["exp2c", "exp2c frozen detr", "frozen detr road-waymo"]
created: 2026-05-07
updated: 2026-05-20
sources:
  - "ROAD_Reason/experiments/exp2c_frozen_detr/model.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/deformable_decoder.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/clip_encoder.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/config.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/train.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/backbone.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/fpn.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/losses.py"
  - "ROAD_Reason/experiments/exp2c_frozen_detr/matcher.py"
tags: [finding, road-plusplus, exp2c, frozen-detr, deformable-detr, clip, detection, efficientnet, encoder]
status: draft
---

# Exp2c — Frozen-DETR: EfficientNet-FPN + DETR Encoder + CLIP ViT-L/14

Experiment 2c faithfully implements the **Frozen-DETR** architecture (Fu et al., NeurIPS 2024) to fix Exp2b's two major gaps: (1) no encoder (FPN went straight to decoder), and (2) scalar gate fusion destroyed localization. Replaces Qwen2.5-VL ViT with **CLIP ViT-L/14@336px** (approved by Dr. Moradi, 2026-05-07).

**Status:** Training in progress — epoch 24/30 (2026-05-12). Localization steadily improving. **f-mAP evaluated at epoch 15** — best checkpoint eval running. Val action mAP peaked at **43.72%** (ep23).

**Motivation:** Exp2b's scalar-gate fusion (`fused = fpn + gate * vlm`) was the simplest possible fusion and produced 1.71% agent f-mAP (10x below RetinaNet). The [[comparisons/fusion-for-detection-lit-review|fusion lit review]] identified Frozen-DETR's encoder self-attention as a fundamentally richer fusion mechanism: CNN and VLM tokens attend to each other through 6 encoder layers of bidirectional attention, rather than a single scalar gate. Exp2c implements this faithfully.

**Key design change:** Instead of the Qwen2.5-VL ViT (7B params, ~7 GB frozen), Exp2c uses CLIP ViT-L/14@336px (428M params, ~1.7 GB frozen). This saves ~5 GB GPU memory and provides two distinct feature types: CLS token (768-dim scene summary) and patch tokens (1024-dim, 24x24 spatial grid). Dr. Moradi approved the switch because Frozen-DETR demonstrated that CLIP features are sufficient for detection enhancement.

---

## Architecture

```
8 RGB frames
  |--> EfficientNet-B0 (448x448)     --> FPN [P3, P4, P5]  ---|
  |                                                             |---> Encoder (4 scales, 6 layers)
  |--> CLIP ViT-L/14 (336x336)       --> patch_proj [24x24] --|           |
       (frozen, 428M params)                                         strip VLM tokens
       |                                                              |
       |--> CLS token [768] -----> per-layer injection -----> Decoder (3 scales, 6 layers)
                                                                      |
                                                               300 queries -> heads
```

**~445M total params, ~15.7M trainable** (CLIP fully frozen, not in any optimizer group).

| Component | Params | Trainable |
|-----------|--------|-----------|
| CLIP ViT-L/14 | ~428M | 0 (frozen) |
| EfficientNet-B0 | ~5.3M | ~4.2M |
| FPN | ~0.6M | 0.6M |
| PatchTokenProjection | ~0.3M | 0.3M |
| Encoder (6 layers, 4 levels) | ~4.5M | 4.5M |
| Decoder (6 layers, 3 levels) | ~4.8M | 4.8M |
| image_query_proj/norm (6 per layer) | ~1.2M | 1.2M |
| Cls heads + agentness | ~0.1M | 0.1M |

---

## What Changed from Exp2b

### 1. Deformable encoder added (new)

Exp2b had no encoder — FPN features went directly to the decoder. Exp2c adds a **6-layer deformable self-attention encoder** operating on 4 feature scales:

| Scale | Source | Spatial dims | Tokens |
|-------|--------|-------------|--------|
| P3 | EfficientNet+FPN | 56x56 | 3136 |
| P4 | EfficientNet+FPN | 28x28 | 784 |
| P5 | EfficientNet+FPN | 14x14 | 196 |
| CLIP | Patch tokens projected | 24x24 | 576 |
| **Total** | | | **4692** |

The encoder processes all 4692 tokens per frame through 6 layers of deformable self-attention. CNN and CLIP tokens attend to each other — this is Frozen-DETR's core fusion mechanism. After encoding, CLIP tokens (576) are **stripped** from memory. The decoder sees only CNN tokens (4116) across 3 scales.

**Post-norm encoder layers** (matching reference): `src = norm(src + dropout(attn(src+pos, ref_pts, src)))`. Query = src + pos, value = raw src. Positional encoding: 2D sinusoidal + learned per-level embedding.

### 2. VLM replaced: Qwen2.5-VL -> CLIP ViT-L/14

| Aspect | Exp2b (Qwen) | Exp2c (CLIP) |
|--------|-------------|-------------|
| Model | Qwen2.5-VL-7B ViT + LoRA | CLIP ViT-L/14@336px |
| Frozen params | ~7B | ~428M |
| GPU memory | ~7 GB | ~1.7 GB |
| Feature types | Single-scale patches | CLS (768) + patches (1024, 24x24) |
| Adaptation | LoRA r=8 on 8 blocks | None (fully frozen) |
| Fusion | `fpn + scalar_gate * vlm` | Encoder self-attention + decoder CLS injection |

### 3. VLM token stripping after encoder (new)

Reference: `deformable_transformer.py:326-331`. After the encoder, CLIP patch tokens are removed from memory before it enters the decoder. The rationale: VLM semantic information has already been fused into CNN tokens through 6 rounds of bidirectional attention. Keeping VLM tokens in the decoder would add unnecessary computation and potentially confuse deformable cross-attention (CLIP's 24x24 grid doesn't align spatially with CNN feature maps).

### 4. Per-layer CLS injection in decoder (new)

Reference: `deformable_transformer.py:604-782`. Each decoder layer:
1. Projects CLS token: `image_query_proj[layer_id](cls) + image_query_norm[layer_id]` -> [T, 1, 256]
2. Concatenates as extra query (301 total)
3. Runs the full decoder layer (self-attn, cross-attn, temporal-attn, FFN)
4. Strips CLS from output (back to 300 queries)

Each layer has its own `Linear(768, 256)` + `LayerNorm(256)` — the CLS representation is freshly projected per layer, not shared. This injects scene-level context ("intersection with vehicles and pedestrians") at every decoding stage.

### 5. Scalar gate fusion removed

The `vlm_fusion.py` module and all LoRA code are eliminated entirely. No per-level gates, no LoRA adapters.

### 6. Optimizer simplified to 3 groups

| Group | LR | Components |
|-------|-----|-----------|
| Backbone | 2e-5 | EfficientNet trainable blocks |
| Encoder+Decoder | 1e-4 | FPN + patch_proj + encoder + decoder |
| Heads | 1e-4 | Classification heads + agentness |

CLIP is not in any group (fully frozen). Exp2b had 4 groups (included LoRA at 5e-5).

---

## CLIP ViT-L/14 Feature Extraction

The CLIP encoder is implemented from scratch in `clip_encoder.py` (no dependency on the `clip` or `open_clip` packages). Loads the official OpenAI checkpoint via URL with SHA256 verification.

**Two output types:**
- **CLS token** [B, 768]: After `ln_post @ proj`. Scene-level embedding — used as decoder image query.
- **Patch tokens** [B, 1024, 24, 24]: Raw ViT width before final projection. Spatial features — projected via `Conv2d(1024, 256, 1) + GroupNorm(32, 256)` then fed as 4th encoder scale.

**Key dimension note:** CLS dim (768) != patch dim (1024). The CLS token passes through CLIP's final projection layer (`width -> output_dim`); patches are extracted before that projection to preserve spatial detail.

---

## Warm-Start from Exp2b

Transfers compatible weights from exp2b checkpoint. Skip strategy:

| Component | Action |
|-----------|--------|
| EfficientNet backbone | Transfer |
| FPN | Transfer |
| Decoder layers (6), query_embed, reference_points_head, box_heads | Transfer |
| Decoder level_embed, temporal embeddings | Transfer |
| Classification heads + agentness | Transfer |
| CLIP ViT-L/14 | Pre-trained (OpenAI checkpoint) |
| PatchTokenProjection | Random init |
| Encoder (6 layers) | Random init |
| image_query_proj/norm (6 per layer) | Random init |
| Qwen ViT + LoRA (removed) | Ignored |
| VLM gate/projection (removed) | Ignored |

All `vlm_fusion.*` keys in the exp2b checkpoint are explicitly skipped. Keys that exist in both models with matching shapes are transferred; mismatches are reported.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone (spatial) | EfficientNet-B0 (blocks 0-1 frozen) |
| Backbone (semantic) | CLIP ViT-L/14@336px (fully frozen) |
| FPN levels | 3 (P3/P4/P5), out=256 |
| Encoder | 6 layers, 4 levels (P3+P4+P5+CLIP), post-norm |
| Decoder | 6 layers, 3 levels (P3+P4+P5), per-layer CLS injection |
| D_MODEL | 256 |
| NUM_QUERIES | 300 |
| N_DEFORM_POINTS | 4 |
| NHEAD | 8 |
| DIM_FFN | 1024 |
| Clip length | 8 frames |
| Input resolution | 448x448 (EfficientNet), 336x336 (CLIP) |
| MAX_EPOCHS | 30 |
| GRAD_ACCUM | 4 |
| LR_BACKBONE | 2e-5 |
| LR_ENCODER_DECODER | 1e-4 |
| LR_HEADS | 1e-4 |
| WARMUP_STEPS | 500 |
| lambda_cls / lambda_bbox / lambda_giou / lambda_tnorm | 2.0 / 5.0 / 2.0 / 1.0 |

---

## Encoder Token Budget

Per frame, 4692 tokens enter the encoder:

```
P3:   56 x 56 = 3136  (1/8 scale, finest spatial detail)
P4:   28 x 28 =  784  (1/16 scale)
P5:   14 x 14 =  196  (1/32 scale, most semantic)
CLIP: 24 x 24 =  576  (independent grid, 336/14)
Total:          4692 tokens per frame
```

After stripping CLIP tokens, 4116 tokens per frame enter the decoder across 3 scales.

Estimated activation memory: 6 encoder layers x [8, 4692, 256] ~ 900 MB. Total GPU savings vs Exp2b: ~5-6 GB (Qwen ViT was ~7 GB frozen).

---

## Smoke Test Results

Forward + backward pass on GPU with reduced model (2 encoder layers, 2 decoder layers, 2 frames):

- Encoder output: `[2, 4692, 256]` (correct token count across 4 scales)
- After CLIP stripping: `[2, 4116, 256]` (576 CLIP tokens removed)
- Decoder output: `[300, 256]` query features + `[300, 2, 4]` boxes
- Gradients: Both encoder and decoder parameters receive gradients
- CLIP parameters: Zero gradient (confirmed frozen)
- Peak GPU: 445 MB (reduced test configuration)

---

## Relationship to Fusion Lit Review

The [[comparisons/fusion-for-detection-lit-review|CNN-VLM Fusion Lit Review]] identified two implementation paths after Exp2b's scalar gate bottleneck:

- **Path A — Explicit gating (VMCNet + CBAM):** CNN generates gates controlling VLM flow. Interpretable, fast.
- **Path B — Encoder fusion (Frozen-DETR):** CNN and VLM tokens fuse through encoder self-attention. Deeper, learned.

**Exp2c implements Path B.** The encoder's 6 layers of deformable self-attention replace the scalar gate entirely. Each encoder layer is a round of "CNN asks VLM what's here" and "VLM asks CNN where exactly." By layer 6, the features are jointly refined — no longer purely "CNN" or "VLM."

---

## f-mAP Results — Epoch 15 Checkpoint

Evaluated via `eval_baseline_compat.py --ckpt checkpoints/epoch_15.pt --mode frame` on ROAD-Waymo val split, IoU=0.5.

| Head | f-mAP (%) | Recall (%) |
|------|-----------|------------|
| Agent-ness | 10.61 | 48.2 |
| Agent | **1.76** | 59.0 |
| Action | 1.58 | 61.2 |
| Location | 2.30 | 47.4 |
| Duplex | 0.64 | 59.9 |
| Triplet | 1.44 | 60.3 |

**Key insight:** The model has **high recall** (47-61%) — it finds agents reliably — but **low precision at IoU=0.5**. The boxes aren't tight enough yet. This is consistent with the still-improving GIoU loss trajectory.

### Comparison to baselines and prior experiments

| Method | Agent f-mAP | Backbone | Backbone Size |
|--------|-------------|----------|---------------|
| 3D-RetinaNet (I3D-08) | 16.7% | ResNet-50 I3D | ~46M |
| 3D-RetinaNet (SlowFast-08) | 15.3% | SlowFast | ~34M |
| YOLOv8x (agent-only) | 31.6% | YOLOv8x | ~68M |
| ECCV 2024 winner (v-mAP@0.5) | 18.41% | YOLOv8x/m | ~68M |
| Exp1b FCOS (ep15) | 3.2% | Qwen ViT (LoRA) | ~675M |
| **Exp2c Frozen-DETR (ep15)** | **1.76%** | EfficientNet-B0 | **~5.3M** |
| Exp2b Deformable DETR (ep27) | 1.71% | EfficientNet-B0 | ~5.3M |

Exp2c matches Exp2b's f-mAP at epoch 15 (vs Exp2b's epoch 27) while saving ~5 GB GPU. Training continues to epoch 30 with GIoU still improving — final f-mAP should be higher.

---

## Training Results (through epoch 18)

Training started 2026-05-08. Warm-started from Exp2b: 595 keys transferred (backbone, FPN, decoder, heads), Qwen/LoRA keys skipped. ~324M total params, ~20M trainable, 304M frozen (CLIP). Each epoch takes ~4h 35min (~16,540s).

| Epoch | Train Loss | Val Loss | Val Action mAP | Train GIoU | Notes |
|-------|-----------|----------|----------------|------------|-------|
| 1 | 3.810 | 3.813 | 40.12% | 0.793 | |
| 2 | 3.414 | 3.771 | 40.73% | 0.717 | |
| 3 | 3.345 | 3.909 | 40.31% | 0.704 | |
| 4 | 3.301 | 3.798 | 38.84% | 0.695 | |
| 5 | 3.239 | 3.794 | 40.51% | 0.683 | |
| 6 | 3.189 | 3.741 | 41.20% | 0.672 | |
| 7 | 3.138 | 3.682 | 41.68% | 0.662 | |
| 8 | 3.105 | 3.767 | 42.11% | 0.655 | |
| 9 | 3.052 | 3.664 | 42.34% | 0.644 | |
| 10 | 3.013 | 3.654 | 41.83% | 0.637 | |
| 11 | 2.980 | 3.615 | 42.31% | 0.629 | |
| 12 | 2.933 | 3.641 | **42.53%** | 0.620 | Best matched mAP |
| 13 | 2.894 | 3.646 | 42.11% | 0.610 | |
| 14 | 2.861 | 3.624 | 41.38% | 0.604 | |
| 15 | 2.830 | 3.620 | 42.17% | 0.596 | f-mAP eval done |
| 16 | 2.790 | 3.640 | 42.25% | 0.586 | |
| 17 | 2.757 | 3.623 | 42.88% | 0.580 | |
| 18 | 2.716 | 3.626 | 43.56% | 0.573 | |
| 19 | 2.681 | 3.648 | 43.12% | 0.564 | |
| 20 | 2.648 | 3.630 | 43.45% | 0.557 | |
| 21 | 2.619 | 3.599 | 43.42% | 0.550 | |
| 22 | 2.593 | 3.581 | 43.68% | 0.545 | |
| 23 | 2.563 | 3.619 | **43.72%** | **0.538** | New best mAP; ep24 in progress |

**Key observations:**

1. **Localization still improving at epoch 23.** GIoU loss: 0.793 → 0.538. No plateau across 23 epochs. Steady ~0.01 improvement per epoch.

2. **Val matched action mAP peaked at 43.72% (ep23).** Classification quality continues to improve alongside localization. The train-val gap is widening (2.56 vs 3.62) but val mAP keeps climbing — the model is still learning useful features.

3. **Train loss drops consistently** (3.81 → 2.56). Cosine LR decay in the final 7 epochs should tighten boxes further.

4. **No crashes or OOMs** across 23 epochs (~105 hours). Architecture is stable.

5. **Best checkpoint eval running** — `eval_baseline_compat.py` on `best.pt` (ep23) to get updated f-mAP numbers beyond the ep15 evaluation.

---

## Backbone Size Comparison: Exp2c vs 3D-RetinaNet Baseline

A key context for interpreting results: our spatial backbone is dramatically smaller than the baseline's.

| | 3D-RetinaNet (baseline) | Exp2c (ours) |
|--|--|--|
| Spatial backbone | ResNet-50 I3D (3D convs, Kinetics-pretrained) | EfficientNet-B0 (2D, ImageNet-pretrained) |
| Backbone params | ~46M | ~5.3M |
| Backbone size ratio | **9x larger** | 1x |
| Temporal modeling | 3D convolutions throughout (native) | Temporal embeddings + temporal self-attention in decoder |
| Semantic branch | None | CLIP ViT-L/14 (304M frozen) |
| Detection head | RetinaNet (anchor-based, dense) | Deformable DETR (query-based, 300 queries) |
| Total trainable | ~50-55M | ~20M |

The baseline uses a single large 3D CNN (ResNet-50 I3D) to handle both spatial precision and semantic understanding. Our architecture splits this: CLIP provides semantics through the encoder, the smaller CNN handles localization, and the encoder fuses them through 6 layers of bidirectional attention.

**Implication:** If Exp2c achieves comparable f-mAP with a backbone 9x smaller (trainable params 2.5x fewer), that validates the dual-backbone Frozen-DETR approach — frozen VLM semantics compensate for a lighter spatial backbone. If there's still a gap, the backbone can be scaled up (EfficientNet-B3 at ~12M, B4 at ~19M) without architectural changes.

---

## Expected Outcomes

Based on Frozen-DETR's results on COCO (DINO R50: 49.0 -> 51.9 AP, +2.9 AP):

1. **Localization should improve significantly over Exp2b.** The encoder gives CNN and VLM tokens 6 rounds of interaction vs one scalar gate. Deformable attention focuses on relevant spatial locations.
2. **Long-tail categories should benefit most.** Frozen-DETR showed +6.6% AP on LVIS long-tail. ROAD++ triplets are inherently long-tail (86 valid out of 3520 possible).
3. **VLM token stripping should preserve CNN precision in the decoder.** The decoder operates on CNN-only tokens, so box regression isn't confused by CLIP's independent spatial grid.
4. **CLS injection should improve classification.** Scene-level context helps distinguish ambiguous detections (e.g., "car turning" vs "car moving" depends on road context).

**Target:** Beat Exp2b's 1.71% agent f-mAP. Realistic goal: 3-5% agent f-mAP (closing the gap toward RetinaNet's 17.76%). If the architecture validates, backbone scaling (EfficientNet-B3/B4) is the next knob to turn.

---

## Post-Mortem: Resolution Was the Bottleneck (2026-05-17)

Diagnostic analysis comparing exp2c and exp2d revealed that **input resolution — not backbone strength — was the root cause of low f-mAP**. At 448×448 (exp2c) and 384×384 (exp2d):

- A 16px pedestrian in the original 1920×1280 frame becomes **3.2–3.7px** in model input
- FPN stride-8 features: 3.2px = 0.4 cells — below the minimum needed for IoU≥0.5
- Large objects (cars/trucks) achieved IoU 0.95+ but small objects were 0.2-0.4

The Frozen-DETR paper runs R50 at **800×1333** (7× more pixels). At that resolution, the same pedestrian is ~6.7px — detectable. The backbone was never the problem; resolution was. See [[findings/exp2e-r50-frozen-detr|Exp2e]] for the resolution fix experiment.

---

## Post-Mortem: Missing Negative Supervision (identified May 2026)

The 1.76% agent f-mAP was primarily caused by **missing negative supervision on unmatched queries** (same bug as exp2/2b). The resolution bottleneck (448×448 vs paper's 800×1333) was real but secondary — exp2e proved resolution alone only gets to 5.5% agentness f-mAP. The dominant failure was scoring: unmatched queries had unchecked class scores. Fixed in [[findings/exp2f-flat-head|Exp2f]].

---

## Related

- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — predecessor; scalar gate fusion bottleneck motivates this redesign
- [[findings/exp2f-flat-head|Exp2f Flat Head]] — fixes the missing negative supervision bug present in this experiment
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — reference paper implemented here
- [[comparisons/fusion-for-detection-lit-review|CNN-VLM Fusion Lit Review]] — Path B (encoder fusion) is what Exp2c implements
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — core problem being addressed
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline target (17.76% agent f-mAP)
- [[projects/road-reason|ROAD_Reason Project]] — parent project
