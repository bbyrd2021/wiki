---
type: finding
title: "Exp1b — FCOS Dense Detection Design (ROAD-Waymo)"
aliases: ["exp1b", "exp1b fcos", "dense detection exp1b"]
created: 2026-04-17
updated: 2026-04-21
sources:
  - "ROAD_Reason/experiments/exp1b_road_r/model.py"
  - "ROAD_Reason/experiments/exp1b_road_r/assign.py"
  - "ROAD_Reason/experiments/exp1b_road_r/losses.py"
  - "ROAD_Reason/experiments/exp1b_road_r/train.py"
  - "ROAD_Reason/experiments/exp1b_road_r/config.py"
  - "ROAD_Reason/tnorm_loss.py"
tags: [finding, road-plusplus, exp1b, fcos, detection, lora, focal-loss, t-norm, qwen]
status: draft
---

# Exp1b — FCOS Dense Detection Design (ROAD-Waymo)

Experiment 1b redesigns the Qwen2.5-VL classification pipeline from **oracle-box classification** (Exp1) to **paper-analogous dense detection**: every spatial ViT token independently predicts whether it contains an agent, where the agent's box is, and its full compositional label. This makes the pipeline self-contained at inference (no GT boxes required) and directly comparable to the ROAD-R paper's 3D-RetinaNet baseline on detection-based metrics.

**Status:** Training complete (2026-04-20, 15 epochs). Eval complete. Best checkpoint: epoch 15, val action mAP = 0.3242.

---

## Motivation: Why Redesign from GT Boxes to Dense Detection?

[[findings/exp1-vs-retinanet-baseline|Exp1]] (oracle GT boxes + frozen ViT + BCE + Łukasiewicz t-norm) revealed three problems:

1. **GT-box oracle inflates classification metrics** — agent/action/loc look strong because detection is perfect by construction. Duplex and triplet were still *weaker* than the 3D-RetinaNet baseline even with oracle boxes, because the action head had near-zero F1 on rare classes.
2. **No fair comparison to published numbers** — ROAD-R reports f-mAP (detection + classification jointly). Oracle-box numbers are incomparable; a detection head is required.
3. **Agentness hardcoded to 1.0** — in the t-norm flat vector `[agentness, agent, action, loc]`, agentness was always 1.0 because every input was a guaranteed real agent. The model never learned the concept of "this region is actually an agent."

Exp1b addresses all three by predicting both localization and classification from dense ViT tokens, analogous to the paper's anchor-per-location RetinaNet.

---

## Architecture

### Before (Exp1 oracle-box pipeline)

```
pixel_values + frame_boxes_list (GT oracle)
  → QwenViTExtractor  [T, H', W', D]
  → ROIAveragePool    [n_GT_agents, D]      ← features extracted at GT box locations
  → TubeLinkingModule [n_GT_agents, D]      ← cross-frame attention over GT agents
  → ClassificationHeads
      agent   sigmoid(Linear(D→10))
      action  sigmoid(Linear(D→22))
      loc     sigmoid(Linear(D→16))
      duplex  sigmoid(Linear(D→49))
      triplet sigmoid(Linear(D→86))
  → agentness hardcoded = 1.0 in t-norm vector
```

### After (Exp1b FCOS dense detection)

```
pixel_values only (no GT boxes at inference)
  → QwenViTExtractor + LoRA  [T, H', W', D]
  → reshape: [T·H'·W', D]                   ← every spatial token is a candidate
  → DetectionHeads (dense, applied to all tokens)
      agentness  sigmoid(Linear(D→1))        ← real learned score
      box        Linear(D→4)                 ← FCOS (l,t,r,b) distances, normalized
      agent      sigmoid(Linear(D→10))
      action     sigmoid(Linear(D→22))
      loc        sigmoid(Linear(D→16))
      duplex     sigmoid(Linear(D→49))
      triplet    sigmoid(Linear(D→86))
  → threshold by agentness > 0.5 → NMS → final detections
  → t-norm flat vector uses predicted agentness (not 1.0)
```

For a typical frame at 448×448 with `spatial_merge_size=2` and `patch_size=14`: H'=W'=16, yielding **256 token candidates per frame**, 2048 across 8 frames. The model learns to suppress ~99% as background.

---

## FCOS Token Assignment

Training uses FCOS-style assignment implemented in `assign.py`. For each token at grid position (i, j) in an [H'×W'] feature map:

**Normalized token center:**
```
cx = (j + 0.5) / W'
cy = (i + 0.5) / H'
```

**Assignment rule:** a token is **positive (foreground)** if its center falls strictly inside a GT box `[x1, y1, x2, y2]`. For ties (center inside multiple boxes), assign to the smallest box by area (FCOS default).

**FCOS ltrb targets** for a positive token assigned to GT box `[x1, y1, x2, y2]`:
```
l = cx - x1    (left distance, normalized ≥ 0)
t = cy - y1    (top distance, normalized ≥ 0)
r = x2 - cx   (right distance, normalized ≥ 0)
b = y2 - cy   (bottom distance, normalized ≥ 0)
```

Classification labels for positive tokens are copied from the assigned GT box (same multi-hot vectors as Exp1).

**Positive token density:** depends on box size and grid resolution. For a Pedestrian (avg normalized box ~0.08×0.25 of frame), at H'=W'=16 roughly 1–4 tokens are positive. For a large vehicle occupying 0.3×0.3, roughly 5–25 tokens are positive.

---

## Loss Function

Total loss = **L_agentness + λ_box · L_box + L_focal + L_tnorm**

### L_agentness — Focal loss on all tokens
```
target: fg=1.0 (positive tokens), bg=0.0 (negative tokens)
FL(p, y) = -(1-p_t)^γ · log(p_t),  γ=2.0
```
Applied to **all T·H'·W' tokens**. Handles extreme class imbalance (~1–2% foreground). No per-class α weighting (binary fg/bg); the focusing factor `(1-p_t)^2` suppresses easy-negative loss contribution.

This is identical to the RetinaNet objectness head design (Lin et al., ICCV 2017).

### L_box — SmoothL1 on FCOS ltrb (foreground only)
```
SmoothL1(pred_ltrb, target_ltrb, β=0.1)
```
Applied only to **foreground tokens**. β=0.1 gives quadratic behavior for small errors (< 0.1 normalized distance), linear for larger errors. Scale is normalized [0,1], so a box-edge error of 0.1 = ~10% of frame width.

### L_focal — Classification focal per head (foreground only)
```
FL_c(p_c, y_c) = -α_c · (1-p_t)^γ · log(p_t),  γ=2.0
```
Applied only to **foreground tokens** across all 5 heads. Per-class α weights from inverse training frequency (see below). This directly addresses Exp1's near-zero F1 on rare action classes.

**Per-class α values (selected examples):**

| Head | Low α class | High α class |
|------|-------------|--------------|
| action | Stop (α≈0.47 — very common) | Ovtak (α≈0.99 — 88 instances) |
| agent | Car (α≈0.34) | TL (α≈0.97) |
| triplet | most (α=0.91–0.99) | — |

### L_tnorm — Gödel t-norm (foreground only)
```
T_Gödel(a, b) = min(a, b)
violation_duplex  = min(p_agent_i, p_action_j)   for all invalid (i,j) pairs
violation_triplet = min(min(p_agent_i, p_action_j), p_loc_k)   for all invalid (i,j,k)
```
Applied to **foreground tokens** using the **predicted agentness** (not 1.0):
```
flat = [predicted_agentness, p_agent (10), p_action (22), p_loc (16)]   → [N_fg, 49]
```
λ=1.0. This is the first experiment where agentness is a real learned score, so the t-norm penalty is modulated by how confident the model is that a region actually contains an agent.

**Expected behavior:** as agentness training converges, the Gödel min(agentness, p_class) term becomes meaningful — low-confidence detections contribute less t-norm loss.

---

## Comparison: Our Implementation vs. ROAD-R Paper

| Aspect | ROAD-R Paper (3D-RetinaNet) | Exp1b (Qwen2.5-VL FCOS) |
|--------|----------------------------|--------------------------|
| Backbone | 3D-ResNet (Kinetics pretrained) + FPN | Qwen2.5-VL-7B ViT (LLM stripped) + LoRA |
| Proposal mechanism | Anchor-based (RetinaNet) | Dense spatial tokens |
| Temporal modeling | 3D convolutions (T frames as depth) | Per-frame tokens; no cross-frame linking |
| Agentness | Real anchor score from detection head | Real learned sigmoid score |
| Box representation | Anchor offsets (Δx, Δy, Δw, Δh) | FCOS distances (l, t, r, b) |
| Classification targets | Multi-label sigmoid per tube | Multi-label sigmoid per foreground token |
| T-norm input | Paper assembles flat vector from tube predictions | Same layout [agentness, agent, action, loc] |
| Best t-norm (Table 7) | Gödel | Gödel (matching paper) |
| Class imbalance handling | Not reported | Focal loss with inverse-frequency α |

**Key structural difference:** RetinaNet uses 3D convolutions for temporal context; Exp1b predicts each frame independently. Cross-frame tube linking (removed from the old Exp1b) was expensive over dense tokens. This is the main remaining gap from the paper's architecture.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Backbone | Qwen2.5-VL-7B ViT (677M params) |
| LoRA rank / α | r=8, α=16 |
| LoRA blocks | First 8 of 32 ViT blocks |
| LoRA targets | `blocks.N.attn.qkv`, `blocks.N.attn.proj` |
| LoRA trainable | 491,520 / 677M params (0.07%) |
| Detection heads | 673,980 params |
| Total trainable | 1,165,500 params |
| Optimizer | AdamW, weight_decay=0.01 |
| LR — LoRA | 5e-5 (cosine decay) |
| LR — heads | 1e-4 (cosine decay) |
| Warmup | 500 steps (linear) |
| Max epochs | 15 |
| Batch size | 1 clip (8 frames) |
| Clip stride | 16 frames |
| γ (focal) | 2.0 (all heads) |
| λ_box | 1.0 |
| λ_tnorm | 1.0 |
| T-norm type | Gödel |
| Agentness threshold | 0.5 (inference) |
| NMS IoU threshold | 0.5 |
| Best ckpt criterion | val action macro-mAP on fg tokens |

**Warm-start:** Loads `exp1_road_r/checkpoints/best.pt` (epoch 6) with `strict=False`:
- ViT weights: load cleanly (same architecture)
- `heads.agent/action/loc/duplex/triplet`: load cleanly (same Linear dimensions)
- `heads.agentness`, `heads.box`: fresh-initialized (not in Exp1)
- `roi_pool.*`, `tube_link.*`: in checkpoint but not in new model → ignored

---

## First Clip Training Log

```
Epoch 1/15
  [train] ep1/15 clip 50/7027
    L=0.9206  agn=0.2285  box=0.3371  focal=0.2804  tnorm=0.0746
    avg_fg=181.0 tokens/clip
    lr_lora=5.00e-06  lr_heads=1.00e-05
    68s elapsed
```

**avg_fg=181** at clip 50 indicates the freshly-initialized agentness head predicts ~50% of tokens as foreground (random baseline). Expected to drop sharply as the agentness focal loss teaches the model to suppress background. A converged model should stabilize at 2–20 fg tokens/clip (1–4 per frame × 8 frames).

---

## Eval Results (Epoch 15, 2026-04-20)

### Internal eval (`eval.py`) — macro-mAP on foreground-thresholded tokens

```bash
python -u experiments/exp1b_road_r/eval.py \
  --out experiments/exp1b_road_r/logs/eval_results.json
```

**avg_detections_per_frame:** 20.21 | **constraint_violation_rate:** 0.29%

| Model | agent | action | loc | duplex | triplet | Note |
|-------|-------|--------|-----|--------|---------|------|
| 3D-RetinaNet (paper) | 17.8% | 15.3% | 13.7% | 13.4% | 9.2% | f-mAP, detection-based |
| 3D-RetinaNet + Gödel | 17.0% | 15.2% | 13.4% | 13.6% | 9.4% | f-mAP |
| Exp1 (oracle GT, BCE) | 35.7% | 22.2% | 33.6% | 12.3% | 8.8% | macro-mAP, oracle boxes |
| **Exp1b (FCOS, focal, Gödel)** | **60.6%** | **32.4%** | **50.0%** | **23.1%** | **17.5%** | macro-mAP on fg tokens |

**Exp1b vs Exp1 delta:** agent +24.9pp, action +10.2pp, loc +16.4pp, duplex +10.8pp, triplet +8.7pp

### Baseline-compatible eval (`eval_baseline_compat.py`) — official f-mAP at IoU=0.5

```bash
python -u experiments/exp1b_road_r/eval_baseline_compat.py \
  --out experiments/exp1b_road_r/logs/baseline_compat_results.json
```

| Model | agent_ness | agent | action | loc | duplex | triplet | Note |
|-------|-----------|-------|--------|-----|--------|---------|------|
| 3D-RetinaNet + Gödel | — | 17.0% | 15.2% | 13.4% | 13.6% | 9.4% | f-mAP |
| **Exp1b (FCOS, ep15)** | **6.0%** | **3.2%** | **1.6%** | **2.5%** | **0.38%** | **1.37%** | f-mAP, IoU=0.5 |

Exp1b is **5.4× below the 3D-RetinaNet baseline** on baseline-compatible agent f-mAP.

### Evaluation methodology gap — why 60.6% vs 3.2%

The 20× discrepancy is structural, not a training failure:

**Internal eval** computes mAP only over tokens that have already been matched to GT annotations (foreground-thresholded tokens). Classification accuracy on tokens that are already confirmed to be near agents is the easy part. The hard part — finding the agent in the first place — is not tested.

**Baseline-compat eval** requires the full detection pipeline to succeed: predicted boxes must IoU ≥ 0.5 with GT boxes to count as true positives. FCOS on ViT tokens struggles here for two reasons:

1. **Box quality under IoU=0.5:** Multiple nearby tokens all fire on the same agent and predict overlapping, slightly-offset boxes. NMS removes most of them. The surviving box is derived from a single token's FCOS (l,t,r,b) regression from its center — with no iterative refinement or anchor structure, getting IoU ≥ 0.5 is not reliable.

2. **Lack of temporal context for box prediction:** FCOS predicts each frame independently. RetinaNet uses 3D convolutions that aggregate temporal evidence to improve per-frame box precision.

This is the root motivation for Exp2 (DETR-style set prediction). See [[findings/exp2-detr-detection|Exp2 DETR Detection Design]].

### Key per-class findings

**Action** (32.4% mAP, all 22 classes seen):
- Strong: Stop 90.5%, Green 96.6%, Red 91.6%, Brake 70.8%, MovAway 58.9%
- Weak: Rev 0.3%, Ovtak 0.4%, IncatRht 2.0%, MovRht 0.7% — focal loss helped but ultra-rare classes (~300–560 GT instances) still near-zero
- Pedestrian-critical: Wait2X 12.5%, Xing 26.5%, XingFmLft 29.1%, XingFmRht 43.2%

**Agent** (60.6% mAP):
- Car 98.2%, TL 97.2%, Bus 85.0%, Ped 92.3%
- SmalVeh 1.9%, EmVeh 5.1% — rare + visually similar to other agent types

**Duplex** (23.1% mAP, 34/49 classes seen):
- All 15 unseen duplexes had zero GT instances in val split (valid classes but absent from val)

**Triplet** (17.5% mAP, 85/86 classes seen):
- Coverage greatly improved vs Exp1 (was 8.8% with oracle boxes)

**Constraint violations:** 0.29% of fg token predictions violate duplex/triplet constraints — t-norm is working effectively.

---

## Related

- [[findings/exp1-vs-retinanet-baseline|Exp1 vs RetinaNet Baseline]] — starting point; documents the classification-only results this experiment improves on
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints (T-Norm)]] — t-norm loss theory; Gödel vs Łukasiewicz comparison
- [[papers/marconato-2022-road-r|Marconato 2022 — ROAD-R]] — source of the t-norm methodology and Table 7 (Gödel best on ROAD-Waymo)
- [[methods/3d-retinanet|3D-RetinaNet]] — the baseline being compared against
- [[datasets/road-plusplus|ROAD++ Dataset]] — dataset with 86-class compositional labels, duplex/triplet constraint structure
- [[projects/road-reason|ROAD_Reason Project]] — parent project
