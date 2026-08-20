---
type: finding
title: "Exp2d v2 — Swin-L Frozen-DETR with COCO Pretrained Encoder + Augmentation"
aliases: ["exp2d v2", "exp2d augmented"]
created: 2026-05-13
updated: 2026-05-20
sources:
  - "ROAD_Reason/experiments/exp2d_swin_detr/train.py"
  - "ROAD_Reason/experiments/exp2d_swin_detr/augmentations.py"
  - "ROAD_Reason/experiments/exp2d_swin_detr/config.py"
  - "ROAD_Reason/experiments/exp2d_swin_detr/backbone.py"
  - "ROAD_Reason/pretrained/dino_4scale_r50_1x_coco_checkpoint0011.pth"
tags: [finding, road-plusplus, exp2d, frozen-detr, swin-l, clip, detection, augmentation, coco-pretrained]
status: draft
---

# Exp2d v2 — Swin-L Frozen-DETR with COCO Pretrained Encoder + Augmentation

Exp2d v1 (Swin-L backbone) showed fast initial learning (val action mAP 0.4455 at epoch 2) but began overfitting by epoch 3. Three anti-overfitting interventions applied for v2:

1. **DINO COCO pretrained encoder/decoder** — 192 keys transferred from 51.9 AP COCO checkpoint
2. **DETR-standard augmentations** — flip, multi-scale resize, random crop
3. **Drop path + stronger regularization** — stochastic depth 0.2, weight decay 0.05

---

## What Changed from v1

| Setting | v1 | v2 |
|---------|----|----|
| Encoder/decoder init | Exp2c warm-start (EfficientNet-based) | DINO COCO pretrained (51.9 AP) |
| Data augmentation | None | Flip + multi-scale resize + random crop + strong color aug |
| DIM_FFN | 1024 | 2048 (match DINO architecture) |
| Drop path (Swin-L) | 0.0 | 0.2 |
| Weight decay | 0.01 | 0.05 |
| Backbone init | ImageNet-22K (timm) | Exp2d v1 best (2 epochs ROAD-Waymo) |

---

## DINO COCO Weight Transfer

Source: `dino_4scale_r50_1x_coco_checkpoint0011.pth` (Fu et al., Frozen-DETR HuggingFace)

| Component | Keys transferred | Notes |
|-----------|-----------------|-------|
| Encoder self_attn (6 layers) | 48 | MSDeformAttn: 4 levels, 4 points, 8 heads — exact match |
| Encoder FFN (6 layers) | 16 | 256→2048→256 (required DIM_FFN change) |
| Encoder norms | 8 | LayerNorm(256) |
| Decoder self_attn (6 layers) | 24 | nn.MultiheadAttention(256, 8) — exact match |
| Decoder FFN (6 layers) | 24 | linear1/2 → ffn.0/ffn.3 mapping |
| Decoder norms | 36 | Mapped by function (see below) |
| Box heads (6 per-layer MLPs) | 36 | 256→256→4 — exact match |
| **Total** | **192** | |

**Decoder norm mapping:** DINO and our model number norms differently because they have different attention blocks. The mapping is by *function*, not by name:

| DINO key | Function | Our key |
|----------|----------|---------|
| `norm2` | Post self-attention | `norm1` |
| `norm1` | Post cross-attention | `norm2` |
| `norm3` | Post FFN | `norm4` |
| (none) | Post temporal-attention | `norm3` (ours only) |

**Not transferred** (architectural differences):
- Decoder cross_attn: DINO uses 4-level deformable, ours uses 3-level (CLIP tokens stripped before decoder)
- Decoder temporal_attn: our addition for video, not in DINO
- Decoder CLS injection: our addition per [[papers/fu-2024-frozen-detr|Frozen-DETR]] design
- Backbone: R50 → Swin-L (our backbone comes from exp2d v1 training)
- Class heads: 91 COCO classes → 5 ROAD heads

## Exp2d v1 Weight Transfer (Step 2)

After DINO transfer, 766 additional keys loaded from exp2d v1 best checkpoint (epoch 2):
- Swin-L backbone (2 epochs ROAD-Waymo training, overrides timm ImageNet init)
- FPN lateral/smooth convs
- CLIP patch_proj
- Decoder cross_attn (3-level), temporal_attn, temporal_pos
- Decoder CLS injection (image_query_proj, image_query_norm)
- Classification heads + agentness head

---

## Augmentation Pipeline

Adapted from `Frozen-DETR/DINO-coco/datasets/transforms.py`. Clip-consistent: same spatial transform applied to all 8 frames, boxes adjusted per frame.

```
RandomHorizontalFlip(p=0.5)
RandomSelect(p=0.5):
  Path A: RandomResize([480..800], max_size=1333)
  Path B: RandomResize([400,500,600])
          → RandomSizeCrop(384, 600)
          → RandomResize([480..800], max_size=1333)
RandomSelect(uniform):
  AdjustBrightness(factor 1.0–2.0)  OR
  AdjustContrast(factor 1.0–2.0)    OR
  LightingNoise(RGB channel swap)
```

**Strong color augmentation** (adapted from `Frozen-DETR/DINO-coco/datasets/sltransform.py`): one of three transforms selected uniformly per clip and applied to all frames. Brightness/contrast factors sampled from `[(1+rand)/2 * 2.0]` = `[1.0, 2.0]`. Lighting noise randomly permutes RGB channels.

Model's forward pass resizes to 384×384 after augmentation, so augmentation provides scale/crop variation before the fixed backbone input size.

---

## Model Architecture

Same as exp2d v1 except `DIM_FFN=2048` (was 1024):

```
Swin-L (195M, stages 0-1 frozen) → FPN (384/768/1536 → 256)
    ↓
Deformable Encoder (6 layers, 4 scales: P3+P4+P5+CLIP, FFN=2048)
    ↓ strip CLIP tokens
Deformable Decoder (6 layers, 3 scales, per-layer CLS injection, FFN=2048)
    ↓
Box heads (6 per-layer MLPs) + 5 classification heads + agentness
```

| Component | Params | Trainable |
|-----------|--------|-----------|
| Swin-L (stages 0-1 frozen) | 195M | 190M |
| FPN | ~1.2M | 1.2M |
| CLIP ViT-L/14 (frozen) | ~304M | 0 |
| Encoder (6 layers, FFN=2048) | ~10M | 10M |
| Decoder (6 layers, FFN=2048) | ~20M | 20M |
| Heads | ~0.1M | 0.1M |
| **Total** | **~523M** | **~214M** |

---

## Training Status

- **Started:** 2026-05-13
- **GPU:** RTX A6000 (48 GB), GPU 1
- **Epochs:** 30
- **Log:** `exp2d_swin_detr/logs/train_v2.log`
- **Optimizer:** Fresh AdamW (not resumed from v1 — data distribution changed)
- **LR:** backbone 1e-5, encoder/decoder 1e-4, heads 1e-4 (cosine decay)

### Early Training Signal

Epoch 1 in progress (clip 100/7027). Initial losses with augmented data:

| Clip | Total Loss | GIoU | T-norm | Notes |
|------|-----------|------|--------|-------|
| 25 | 8.237 | 1.163 | 0.172 | First clips — higher loss than v1 start (expected with augmentation) |
| 50 | 8.171 | 1.165 | 0.163 | |
| 75 | 7.974 | 1.147 | 0.136 | |
| 100 | 7.913 | 1.153 | 0.108 | Loss declining; t-norm dropping fast |

v1 started at 3.789 total loss (no augmentation). v2's ~8.0 starting loss is expected: augmented crops/flips/color changes make the task harder initially. The key diagnostic will be val loss after epoch 1 — it should be lower than v1's epoch 1 val loss (3.626) if the COCO priors + augmentation are working.

### Results

| Epoch | Train Loss | Val Loss | Val Action mAP | Notes |
|-------|-----------|----------|----------------|-------|
| TBD | | | | Epoch 1 in progress |

---

## v1 Results (for comparison)

| Epoch | Train Loss | Val Loss | Val Action mAP |
|-------|-----------|----------|----------------|
| 1 | 3.789 | 3.626 | 0.4414 |
| 2 | 3.437 | 3.515 | **0.4455** (best) |
| 3 | 3.325 | 3.523 | 0.4380 |
| 4 | 3.258 | 3.558 | 0.4412 |

Overfitting began at epoch 3: val loss increasing while train loss decreasing.

---

## Post-Mortem: Resolution Was the Bottleneck (2026-05-17)

At 384×384, Swin-L's stronger features didn't help because **input resolution was the actual bottleneck**. The paper uses R50 at 800×1333 — a fully convolutional backbone with no window-size constraint. Swin-L is tied to 384px pretrained window size, making it incompatible with the paper's resolution without significant re-engineering.

Diagnostic on exp2d v2 (with top-K=10 eval): agent_ness mAP dropped to 5.97% (from 11.49% without top-K). This confirmed the issue isn't duplicate flooding — it's that boxes are geometrically imprecise for small objects at low resolution.

See [[findings/exp2e-r50-frozen-detr|Exp2e]] for the resolution fix: R50 at 800×1333 matching the paper exactly.

---

## Related

- [[findings/exp2c-frozen-detr|Exp2c Frozen-DETR]] — EfficientNet-B0 predecessor (peak action mAP 0.4372)
- [[findings/exp2e-r50-frozen-detr|Exp2e R50 Frozen-DETR]] — resolution fix (R50 @ 800×1333)
- [[findings/exp2f-flat-head|Exp2f Flat Head]] — fixes the missing negative supervision bug present in this experiment
- [[comparisons/yolov8x-vs-swin-l-backbone|YOLOv8x vs Swin-L]] — backbone selection rationale
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — architecture reference (NeurIPS 2024)
- [[papers/eccv24-track1|ECCV 2024 Track 1 Winner]] — YOLOv8x baseline (30.82% v-mAP)
- [[methods/3d-retinanet|3D-RetinaNet]] — official ROAD baseline (16.7% agent f-mAP)
