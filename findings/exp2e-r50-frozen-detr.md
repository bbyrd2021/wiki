---
type: finding
title: "Exp2e — Faithful R50 Frozen-DETR @ 800×1333 + T-norm"
aliases: ["exp2e", "exp2e r50", "r50 frozen detr"]
created: 2026-05-17
updated: 2026-05-20
sources:
  - "ROAD_Reason/experiments/exp2e_r50_detr/config.py"
  - "ROAD_Reason/experiments/exp2e_r50_detr/model.py"
  - "ROAD_Reason/experiments/exp2e_r50_detr/backbone.py"
  - "ROAD_Reason/experiments/exp2e_r50_detr/train.py"
  - "Frozen-DETR/MS-DETR/models/backbone.py"
  - "Frozen-DETR/MS-DETR/datasets/coco.py"
tags: [finding, road-plusplus, exp2e, frozen-detr, resnet50, clip, detection, resolution-fix]
status: draft
---

# Exp2e — Faithful R50 Frozen-DETR @ 800×1333 + T-norm

Faithful replication of the Frozen-DETR paper's spatial config (Fu et al., NeurIPS 2024) with **ResNet-50 at 800×1333** — the paper's exact backbone and resolution. The only difference from the paper: 5 ROAD multi-label heads + Gödel t-norm constraint loss (the novel contribution).

**Status:** Training continuing on GPU 0. f-mAP evaluated at epoch 11 — see Results below.

**Motivation:** Exp2c (EfficientNet-B0 @ 448×448) and exp2d (Swin-L @ 384×384) both achieved strong classification recall (~60%) but failed on f-mAP (1-2% agent f-mAP vs baseline's 17%). Diagnostic analysis revealed the root cause: **input resolution**. At 384×384, the original 1920×1280 frames are downscaled 5×, making a 16px pedestrian only 3.2px wide (0.4 FPN cells at stride 8). Boxes can't achieve IoU≥0.5 at that scale. The paper runs at 800×1333 — 7× more pixels — where the same pedestrian is 8px wide (1 full FPN cell).

---

## Resolution: The Root Cause of Low f-mAP

Debug analysis on exp2d confirmed:
- **Large objects** (cars, trucks): IoU 0.95+ — localization is fine
- **Small objects** (pedestrians, traffic lights): IoU 0.2-0.4 — below the 0.5 threshold
- At 384×384, a 16px GT pedestrian becomes 3.2px in model input
- FPN stride-8 features have one cell per 8px → 3.2px object = 0.4 cells
- Deformable attention can't sub-pixel localize — the spatial information is destroyed

At 800×1333:
- Same 16px pedestrian becomes ~6.7px (800/1920 × 16)
- FPN stride-8 → ~0.8 cells — marginal but detectable
- Larger pedestrians (50px+) → 20px+ in model, easily localizable

The backbone (EfficientNet, Swin-L, R50) was never the bottleneck. **Resolution was.**

---

## Why R50 (Not Swin-L or EfficientNet)

The Frozen-DETR paper uses ResNet-50 — not because R50 is optimal, but because:

1. **Fully convolutional** — accepts any input size. Swin-L is tied to pretrained window size (384 patches, requires multiples of window_size=12)
2. **DINO COCO checkpoint matches exactly** — R50 backbone + encoder + decoder architecture is identical to `dino_4scale_r50_1x_coco_checkpoint0011.pth` (51.9 AP on COCO). Cleanest possible weight transfer.
3. **Paper-proven config** — R50 at 800×1333 is what the paper validates. Deviating from it without evidence caused exp2c/exp2d's resolution problems.

---

## Architecture

```
PIL frames (8 per clip, 1920×1280 original)
    │
    ├─→ Augmentation (multi-scale [480..800], max 1333, flip, crop)
    │   → Resize to random scale, e.g. 800×1200
    │
    ├─→ ResNet-50 (FrozenBN, layer2/3/4 trainable)
    │   → C3: [T, 512, H/8, W/8]     e.g. [8, 512, 100, 150]
    │   → C4: [T, 1024, H/16, W/16]   e.g. [8, 1024, 50, 75]
    │   → C5: [T, 2048, H/32, W/32]   e.g. [8, 2048, 25, 38]
    │
    ├─→ FPN (512/1024/2048 → 256)
    │   → P3, P4, P5 all [T, 256, ...]
    │
    ├─→ CLIP ViT-L/14@336px (frozen)
    │   → patch tokens: [T, 256, 24, 24] (after projection)
    │   → CLS token: [T, 768]
    │
    ├─→ Deformable Encoder (6 layers, 4 scales: P3+P4+P5+CLIP)
    │   → strip CLIP tokens after encoding
    │
    ├─→ Deformable Decoder (6 layers, 3 scales, per-layer CLS injection)
    │   → 300 queries × T frames → boxes + features
    │
    └─→ Heads: agentness [300,1] + agent [300,10] + action [300,22]
              + loc [300,16] + duplex [300,49] + triplet [300,86]
```

**351M total params | 46.7M trainable | 304.5M frozen (CLIP)**

---

## DINO COCO Weight Transfer

Source: `dino_4scale_r50_1x_coco_checkpoint0011.pth` — 457 keys transferred.

| Component | Keys | Notes |
|-----------|------|-------|
| R50 backbone (layer2/3/4) | 265 | Exact architecture match — COCO-finetuned |
| Encoder (6 layers) | 72 | self_attn + FFN(2048) + norms |
| Decoder (6 layers) | 120 | self_attn + FFN + norms + box heads |
| **Total** | **457** | |

**Skipped:**
- FPN: DINO uses `input_proj` (different architecture from our lateral+smooth convs)
- Decoder cross_attn: 4-level in DINO, 3-level in ours (CLIP stripped before decoder)
- Decoder temporal_attn, CLS injection: our additions for video, not in DINO
- Class heads: 91 COCO → 5 ROAD heads
- CLIP patch_proj: not in DINO

This is the cleanest transfer of all experiments — R50 backbone matches exactly (unlike exp2d where R50→Swin-L meant no backbone transfer).

---

## Paper Config vs Exp2e

| Setting | Frozen-DETR paper | Exp2e | Source |
|---------|------------------|-------|--------|
| Backbone | R50, FrozenBN, layer2/3/4 | Same | `backbone.py:72,96-107` |
| FPN channels | [512, 1024, 2048] → 256 | Same | `backbone.py:78` |
| Input (train) | Multi-scale [480..800], max 1333 | Same | `coco.py:139-149` |
| Input (val) | 800, max 1333 | Same | `coco.py:156` |
| Encoder | 6 layers, 4 levels, FFN=2048 | Same | paper train script |
| Decoder | 6 layers, box refine, 300 queries | Same | paper train script |
| CLIP | ViT-L/14@336px | Same | paper |
| Heads | 91 COCO classes | **5 ROAD heads + t-norm** | Novel |
| Queries | 900 | **300** | Practical: 1 GPU |
| two_stage / mixed_selection | Yes | **No** | Not in codebase |
| Dropout | 0.0 | 0.0 | paper train script |
| LR backbone | 2e-5 | 2e-5 | `main.py:34` |
| Weight decay | 1e-4 | 1e-4 | paper default |
| Grad clip | 0.1 | 0.1 | paper default |

**Omitted features** (two_stage, mixed_selection, MS-DETR O2M): not in our codebase. They improve COCO AP by ~2-3% but are orthogonal to the resolution fix. Implementing them is a separate experiment.

---

## Training Config

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Backbone | ResNet-50 (FrozenBN, layer2/3/4 trainable) | Paper |
| Input (train) | Multi-scale [480..800], max 1333 | Paper |
| Input (val) | Short side 800, max 1333 | Paper |
| D_MODEL | 256 | Paper |
| DIM_FFN | 2048 | Paper |
| DROPOUT | 0.0 | Paper |
| NUM_QUERIES | 300 | Practical |
| Epochs | 30 | 1 GPU (paper: 12 on 4 GPUs) |
| Batch size | 1 × grad_accum=4 | 1 GPU |
| LR backbone | 2e-5 | Paper |
| LR encoder/decoder/heads | 1e-4 | Our schedule |
| Weight decay | 1e-4 | Paper |
| Grad clip | 0.1 | Paper |
| GPU | RTX A6000 (48 GB), GPU 0 | |

---

## Encoder Token Budget (at val resolution 800×1200)

```
P3:  100 × 150 = 15,000  (stride 8)
P4:   50 ×  75 =  3,750  (stride 16)
P5:   25 ×  38 =    950  (stride 32)
CLIP: 24 ×  24 =    576  (independent grid)
Total:           20,276 tokens per frame
```

This is **4.3× more tokens** than exp2c/exp2d (4,692 at 448×448). Deformable attention is O(N) not O(N²), so compute scales linearly. Peak GPU memory: **30.3 GB** during training (well within 48 GB A6000).

---

## Early Training Signal

| Clip | Total Loss | Cls | Box | GIoU | T-norm | Aux |
|------|-----------|-----|-----|------|--------|-----|
| 25 | 16.99 | 0.070 | 0.105 | 1.115 | 0.463 | 5.75 |
| 50 | 15.74 | 0.070 | 0.111 | 1.191 | 0.472 | 5.46 |

Loss is reasonable and declining. Higher starting loss than exp2c/exp2d is expected — more tokens, randomly initialized FPN and cross-attention need to settle.

---

## Known Gaps

1. **Decoder cross-attention randomly initialized** — same situation as the paper (DINO 4-level → 3-level after CLIP stripping). Not unique to us.
2. **FPN randomly initialized** — DINO uses `input_proj` (different arch). Small component, trains fast.
3. **300 queries vs paper's 900** — higher resolution reveals more small objects; 300 may be insufficient for dense frames.
4. **No two_stage / mixed_selection** — paper uses these for better query initialization (~2-3% AP gain on COCO). Our queries start from learned embeddings.

---

## Results

### f-mAP @ IoU=0.5 (epoch 11 best.pt)

| Head | mAP | mR |
|------|-----|-----|
| agent_ness | 5.536 | 15.06 |
| agent | 1.396 | 29.10 |
| action | 1.766 | 38.12 |
| loc | 1.373 | 22.73 |
| duplex | 0.348 | 20.05 |
| triplet | 1.011 | 35.66 |

Resolution improved f-mAP ~4× over exp2c/2d (~1.4% agent → 5.5% agent_ness), confirming resolution was a real bottleneck. But **f-mAP is still far below the baseline's 17.76%**. Recall is very high (29-38% on most heads) — the model finds agents but scores them wrong.

### Score-Localization Decorrelation (Post-Mortem)

Diagnostic analysis revealed a deeper problem beyond resolution:

- **38.7% of GT boxes** match at least one query at IoU>=0.5 — localization works
- **Top-20 Car detections:** score > 0.997, IoU < 0.25 — scoring fails
- **Root cause:** Unmatched queries (280/300) receive zero gradient on classification heads. Only `_agentness_loss` pushes agentness toward 0. Class scores on unmatched queries drift unchecked to near-saturation.
- **The baseline doesn't have this problem:** its flat 184-dim sigmoid + focal loss on ALL anchors explicitly supervises negatives toward 0 on every dimension.

**Fix:** [[findings/exp2f-flat-head|Exp2f]] replaces the 6 separate heads with a single flat `nn.Linear(256, 184)` + focal loss on all 300 queries x 184 dims.

---

## Related

- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — EfficientNet-B0 @ 448; 1.76% agent f-mAP (resolution bottleneck)
- [[findings/exp2d-swin-detr-v2|Exp2d v2 Swin-L]] — Swin-L @ 384; same resolution bottleneck
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — reference paper
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — core problem
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline target (17% agent f-mAP)
- [[projects/road-reason|ROAD_Reason Project]] — parent project
