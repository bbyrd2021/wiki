---
type: finding
title: "Exp2f — Flat 184-dim Sigmoid Head (Baseline Classification Design)"
aliases: ["exp2f", "exp2f flat head", "flat head experiment"]
created: 2026-05-20
updated: 2026-08-17
sources:
  - "ROAD_Reason/experiments/exp2f_flat_head/config.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/model.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/losses.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/matcher.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/train.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/eval_baseline_compat.py"
tags: [finding, road-plusplus, exp2f, frozen-detr, resnet50, clip, detection, flat-head, score-decorrelation-fix]
status: draft
---

# Exp2f — Flat 184-dim Sigmoid Head (Baseline Classification Design)

Fixes the **missing negative supervision on unmatched queries** found across all DETR experiments (exp2 through exp2e) by replacing 6 separate classification heads with a **single flat `nn.Linear(256, 184)` head** — the same design used by the 3D-RetinaNet baseline. Sigmoid + focal loss on ALL 300 queries x 184 dims. Everything else stays identical to exp2e.

**Status:** Training complete (30 epochs). Best matched action mAP: 0.209 (epoch 30). Baseline-compatible f-mAP: **5.51% agent** (epoch 30 eval).

---

## The Bug: Missing Negative Supervision on Unmatched Queries

**This bug is present in every DETR experiment we ran (exp2, exp2b, exp2c, exp2d, exp2e).** It was the dominant cause of low f-mAP across the entire exp2 series.

### How DETR matching works

The decoder outputs **300 queries** per clip. Hungarian matching assigns ~20 queries to GT agents. The other ~280 are unmatched (background).

### What our loss did (exp2 through exp2e)

We had **6 separate classification heads** and two loss functions:

| Loss | Applied to | What it does |
|------|-----------|--------------|
| `_classification_loss` | **Matched queries only** (~20/300) | Focal loss on 5 heads against GT labels |
| `_agentness_loss` | All 300 queries | Pushes agentness toward 0 for unmatched, 1 for matched |

The **280 unmatched queries received zero gradient on their 5 classification heads** (agent, action, loc, duplex, triplet). Nothing told them "you're not a Car" or "you're not Moving." Their class scores drifted to near-saturation (>0.99).

### What the baseline does (3D-RetinaNet)

Every anchor gets a **single 184-dim sigmoid vector**. Focal loss on ALL anchors:
- **Positive anchors:** target = GT labels across all 184 dims
- **Negative anchors:** target = all-zeros across all 184 dims

Negative anchors are explicitly trained to output 0 on every class.

### What happens at test time

The evaluator ranks detections by score. But:

- **Unmatched queries** have class scores > 0.99 (never suppressed)
- **Matched queries** have class scores ~ 0.7-0.9 (properly calibrated by training)
- Top detections are dominated by **junk queries with high scores but terrible boxes**

Post-hoc analysis of exp2e (epoch 11) confirmed:

- **38.7% of GT boxes** match at least one query at IoU>=0.5 — the model *can* localize
- **Top-20 Car detections:** all have score > 0.997, but all have IoU < 0.25 with GT
- The scoring system fails, not the localization

### Why standard DETR doesn't have this problem

Standard DETR (Carion et al., 2020) uses a **softmax** head with a "no-object" class. Unmatched queries are assigned the no-object target, which naturally suppresses all real class logits. We used **sigmoid** heads (required for multi-label classification — an agent can be both "Moving" and "TurningRight") but didn't add the equivalent negative supervision that softmax provides for free.

### Affected experiments

| Experiment | f-mAP agent | Bug present? | Actual bottleneck |
|------------|------------|-------------|-------------------|
| Exp2 (Qwen ViT + DETR) | 0.63% | **Yes** | Unmatched query scores |
| Exp2b (EfficientNet + Deformable DETR) | 1.71% | **Yes** | Unmatched query scores |
| Exp2c (EfficientNet + Frozen-DETR + CLIP) | 1.76% | **Yes** | Unmatched query scores |
| Exp2d (Swin-L + Frozen-DETR + CLIP) | 1.67% | **Yes** | Unmatched query scores |
| Exp2e (R50 @ 800x1333 + CLIP) | 5.54% (agentness) | **Yes** | Unmatched query scores |
| **Exp2f (R50 + flat 184-dim head)** | **5.51%** (ep30) | **Fixed** | Query coverage + CLIP fusion mismatch now the bottleneck |

The gradual improvements (0.63% -> 1.71% -> 5.54%) came from architectural upgrades (deformable attention, better backbones, higher resolution), but the dominant bottleneck was always the same bug. Resolution did help (exp2e's 5.5% vs ~2% in earlier experiments), but only because better boxes meant the few correctly-scored queries ranked slightly better.

### How this differs from Exp1b's localization problem

Exp1b (FCOS dense detection) had a completely different problem:

| | Exp1b (FCOS) | Exp2-2e (DETR) |
|---|---|---|
| Classification | Strong (60.6% macro-mAP on fg tokens) | Broken (unmatched queries unsupervised) |
| Localization | Poor (single-scale ViT, no FPN) | Decent (38.7% of GT matches at IoU>=0.5) |
| f-mAP bottleneck | **Box quality** — couldn't draw tight boxes | **Score ranking** — good boxes outranked by junk |
| Negative supervision | Correct (every token gets a target) | **Missing** (unmatched queries get zero gradient) |

---

## The Fix: Flat 184-dim Head

| Component | exp2e (broken) | exp2f (fix) |
|-----------|---------------|-------------|
| Classification heads | 6 separate `nn.Linear` (agentness:1, agent:10, action:22, loc:16, duplex:49, triplet:86) | **1 flat `nn.Linear(256, 184)`** |
| Class loss scope | Matched queries only (5 heads) + agentness on all | **ALL 300 queries x 184 dims** |
| Negative supervision | Agentness -> 0 only; class heads get zero gradient | **All 184 dims -> 0 on unmatched** |
| Agentness | Separate head | **Slot 0 in flat vector** |
| Loss terms | `L_cls + L_bbox + L_giou + L_agentness + L_tnorm` | `L_cls + L_bbox + L_giou + L_tnorm` (agentness folded into L_cls) |

Everything else unchanged: R50 + FPN + CLIP ViT-L/14 + deformable encoder/decoder, box losses, Hungarian matching, augmentations, DINO warm-start, hyperparameters.

---

## Flat Vector Layout

```
Index:  [0]  [1..10]     [11..32]      [33..48]   [49..97]      [98..183]
Head:   agn  agent(10)   action(22)    loc(16)    duplex(49)    triplet(86)
Total:  1 + 10 + 22 + 16 + 49 + 86 = 184
```

- **Matched queries:** `[1, agent_multihot, action_multihot, loc_multihot, duplex_multihot, triplet_multihot]`
- **Unmatched queries:** `[0, 0, ..., 0]` (184 zeros — explicit negative supervision)

---

## Why This Should Work

The 3D-RetinaNet baseline achieves 17.76% agent f-mAP with exactly this design:
- Single flat 184-dim sigmoid vector
- Focal loss on ALL anchors (positive and negative)
- Negative anchors get target = all-zeros across all 184 dims
- Alpha weighting: per-class inverse-frequency

The baseline's approach structurally prevents score-localization decorrelation: a query that isn't matched to any GT gets supervised toward 0 on *every class dimension*, not just agentness. High classification scores can only survive on queries that actually have matching GT — which are the queries with good boxes.

---

## Architecture

Identical to exp2e except the classification heads and training hyperparameters:

```
PIL frames (8 per clip, 1920x1280 original)
    |
    +--> R50 (FrozenBN) + FPN + CLIP ViT-L/14 (same as exp2e)
    |
    +--> Deformable Encoder (6L, 4 scales) --> strip CLIP tokens
    |
    +--> Deformable Decoder (6L, 3 scales, CLS injection)
    |    --> 300 queries x T frames --> boxes + features
    |    --> per-layer aux outputs (layers 0-4)
    |
    +--> cls_heads: 6x nn.Linear(256, 184)  <-- per-layer (paper: _get_clones)
    |    --> cls_heads[5] for final output, cls_heads[0-4] for aux losses
    |    --> bias init: -4.595 (prior_prob=0.01, focal loss standard)
    |    --> sigmoid + focal loss on ALL 300 queries
```

**351M total params | 47M trainable | 304.5M frozen (CLIP)**

### v2 Training Fixes (paper gap audit, 2026-05-20)

| Setting | v1 | v2 (paper-matched) |
|---------|----|--------------------|
| Class heads | 1 shared `nn.Linear(256, 184)` | 6 independent per-layer heads (`_get_clones`) |
| Bias init | default zeros | `-4.595` (RetinaNet focal loss standard) |
| LR enc/dec | 1e-4 | 2e-4 (paper: `--lr 2e-4`) |
| LR deformable | 1e-4 (same as enc/dec) | 2e-5 (0.1x mult on `reference_points` + `sampling_offsets`) |
| LR backbone | 2e-5 | 2e-5 (unchanged) |

---

## Loss Design

```python
# For each clip:
targets = zeros(300, 184)                    # all queries start as negatives
targets[matched_pred] = pack_gt_labels()     # matched queries get GT labels

loss = sigmoid_focal_loss(pred_logits, targets)  # ALL 300 queries, ALL 184 dims
```

- **Focal loss** (gamma=2.0): down-weights easy negatives — the ~280 unmatched queries quickly learn to predict ~0 everywhere, contributing near-zero loss
- **Per-class alpha**: inverse-frequency weighting (same as baseline) — upweights rare classes like Cyclist
- **Agentness at slot 0**: matched = 1.0, unmatched = 0.0 — same signal as the old separate agentness head, but now part of the unified loss
- **T-norm**: slices [agentness, agent, action, loc] from flat vector for constraint violation penalty on matched queries

---

## Evaluation Compatibility

`eval_baseline_compat.py` includes `unpack_flat_probs()` which splits the 184-dim sigmoid output back into per-head dicts for the baseline evaluator:

```python
flat_probs = pred_logits.sigmoid()  # [N, 184]
probs = {
    "agentness": flat_probs[:, 0:1],
    "agent":     flat_probs[:, 1:11],
    "action":    flat_probs[:, 11:33],
    "loc":       flat_probs[:, 33:49],
    "duplex":    flat_probs[:, 49:98],
    "triplet":   flat_probs[:, 98:184],
}
```

---

## Early Training Signal

| Clip | Total Loss | Cls | Box | GIoU | T-norm | Aux |
|------|-----------|-----|-----|------|--------|-----|
| 25 | 7.00 | 0.116 | 0.154 | 0.994 | 0.470 | 3.54 |
| 50 | 7.01 | 0.113 | 0.149 | 1.002 | 0.469 | 3.57 |
| 100 | 6.85 | 0.105 | 0.137 | 1.001 | 0.439 | 3.51 |

Cls loss (0.105) is in a reasonable range — it now covers all 300 queries x 184 dims, not just matched queries x 5 heads. The key diagnostic will be **score calibration at epoch 2-3**: `pred_logits.sigmoid().mean(dim=0)` should be near 0 for most dims — unmatched queries suppressed.

---

## Results

| Epoch | Train Loss | Val Loss | Val Matched Action mAP | Notes |
|-------|-----------|----------|----------------------|-------|
| 6 | 2.68 | 2.75 | 0.1756 | First best |
| 8 | 2.51 | 2.56 | 0.1815 | |
| 10 | 2.37 | 2.30 | 0.1825 | |
| 11 | 2.33 | 2.37 | 0.1933 | |
| 14 | 2.20 | 2.21 | 0.1945 | f-mAP eval: agent 4.40%, action 3.28% |
| 16 | 2.14 | 2.12 | 0.1952 | |
| **18** | **2.08** | **2.14** | **0.1990** | |
| **30** | — | — | **0.2089** | **Final best** |

**Baseline-compatible f-mAP:**

| Metric | Exp2f ep14 | Exp2f ep30 | Baseline (3D-RetinaNet) | Ratio (ep30) |
|--------|-----------|-----------|------------------------|-------|
| Agentness | 5.82% | **13.38%** | 23.35% | 0.57x |
| Agent | 4.40% | **5.51%** | 17.76% | 0.31x |
| Action | 3.28% | **4.20%** | 15.28% | 0.27x |
| Location | 2.04% | **3.96%** | 13.73% | 0.29x |
| Duplex | 0.62% | **1.11%** | 13.44% | 0.08x |
| Triplet | 1.16% | **1.72%** | 9.17% | 0.19x |

Constraint violations (ep30, IoU 0.5): duplex 0.62%, triplet 80.09%. **⚠ 2026-08-17: the triplet figure is mostly artifact** — the metric validated against the mis-indexed JSON childs arrays (only 6/86 valid triplets recognized); see [[findings/road-waymo-childs-mis-indexed|the childs bug]]. Duplex stayed low only because dominant pairs sit in the 20-pair overlap.

Training from ep14→30 yielded significant improvement across all heads, with agentness more than doubling. The flat head fix delivered a **4x improvement** over exp2e (1.4% → 5.5% agent f-mAP), confirming the score-decorrelation diagnosis. But still 3.2x below baseline — the remaining gap is query coverage (300 queries with 1-to-1 matching vs dense anchors) and CLIP encoder fusion mismatch. Addressed in [[findings/exp2g-msdetr|Exp2g]].

---

## Remaining Gaps vs Frozen-DETR / MS-DETR Paper

Full audit of the paper's actual training config (`MS-DETR/scripts/train_ms_detr_pp_900.sh`) vs our exp2f implementation (2026-05-20).

### Critical Gaps (likely affecting f-mAP)

| Gap | Paper Config | Our exp2f | Expected Impact |
|-----|-------------|-----------|-----------------|
| **Two-stage query init** | `--two_stage --mixed_selection` — encoder proposes candidate regions, top-k become decoder queries with 4D reference points | Learned query embeddings (random init) | ~2-3% AP on COCO; queries start spatially-informed |
| **One-to-many (O2M) loss** | `--use_ms_detr` with k=6 — MS-DETR's core contribution, 6x more positive training signals per image | Not implemented | Paper's main novelty; significant on small datasets (7K clips) |
| **Encoder loss** | Two-stage encoder proposals supervised with separate class/box/GIoU loss | No encoder supervision | Helps encoder learn better proposals |
| **LR multiplier for deformable params** | `reference_points` and `sampling_offsets` get 0.1x base LR (2e-5) | Full LR (1e-4) — 5x too high | Sensitive params; too-high LR causes unstable sampling |
| **900 queries** | `--num_queries 900` | 300 (single A6000 GPU memory limit) | More queries = better recall in dense scenes |

### Important Gaps (correctness / stability)

| Gap | Paper Config | Our exp2f | Expected Impact |
|-----|-------------|-----------|-----------------|
| **4th backbone level (stride-64)** | Extra `Conv2d(2048, 256, 3, stride=2)` on R50 C5 | 3 FPN levels only | Helps detect large objects; standard in Deformable DETR |
| **FrozenBatchNorm** | BN in R50 frozen to pretrained statistics | Trainable BN | batch_size=1 makes BN statistics noise — actively harmful |
| **Class head bias init** | `prior_prob=0.01; bias = -log((1-p)/p) = -4.595` | Default zero init | Standard focal loss init; without it, initial preds ~0.5 cause high early loss |
| **Per-layer class heads** | Class head cloned per decoder layer for aux losses | Shared single head | Each decoder layer gets its own supervision signal |
| **Base LR** | 2e-4 | 1e-4 | 2x difference |

### Lower Priority

| Gap | Paper Config | Our exp2f |
|-----|-------------|-----------|
| `--look_forward_twice` | Iterative refinement looks forward 2 decoder layers | Standard 1-layer look-ahead |
| `--use_aux_ffn` | Separate FFN for O2M branch | N/A (no O2M yet) |

### Triage

**Quick fixes** (apply to current exp2f or hotfix): LR multiplier for deformable params, FrozenBatchNorm, class head bias init. These are correctness issues, not features.

**Dedicated experiment** (exp2g or later): Two-stage + O2M + encoder loss + 900 queries. These are the paper's core architectural contributions and should be added together since they interact (two-stage generates the proposals that O2M loss supervises).

---

## Follow-Up: Exp2g (MS-DETR)

Exp2f confirmed the flat head fixes scoring, but the remaining 4x gap to baseline is **query coverage** — 300 queries with 1-to-1 matching can't compete with dense anchors. [[findings/exp2g-msdetr|Exp2g]] implements the full MS-DETR recipe: two-stage proposals (900 queries from encoder), one-to-many loss (k=6 matches per GT), encoder proposal supervision, and softmax on agent dims. Training in progress.

---

## Related

- [[findings/exp2e-r50-frozen-detr|Exp2e R50 Frozen-DETR]] — same architecture, separate heads; f-mAP: agent 1.4% (score-localization decorrelation)
- [[methods/3d-retinanet|3D-RetinaNet]] — baseline using the same flat 184-dim design (17% agent f-mAP)
- [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] — reference architecture paper
- [[projects/road-reason|ROAD_Reason Project]] — parent project
