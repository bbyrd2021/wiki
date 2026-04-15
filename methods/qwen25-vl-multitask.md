---
type: method
title: "Qwen2.5-VL Multi-Task Architecture (Approach 3)"
aliases: ["Qwen2.5-VL multi-task", "Approach 3 architecture"]
created: 2026-04-10
updated: 2026-04-10
sources: []
tags: [method, approach3, qwen, vlm, detection, captioning, trajectory, t-norm, road-reason]
status: draft
---

# Qwen2.5-VL Multi-Task Architecture (Approach 3)

Full architecture specification for the multi-task VLM fine-tuning approach. Shared Qwen2.5-VL-7B backbone with three task-specific output module sets.

---

## Backbone: Qwen2.5-VL-7B

```
Video / frames
      ↓
[Vision Encoder — ViT]
  - Dynamic resolution (4–16,384 tokens per image)
  - Window attention (most layers) + 4 full-attention layers
  - SwiGLU activation, RMSNorm
  - 2D-RoPE spatial + mRoPE temporal (absolute time IDs)
  - Dynamic FPS sampling for video
      ↓  V ∈ ℝ^{T×N×D}
[MLP Projector]
      ↓
[Qwen2.5-7B LLM]  ← used for Tasks 1 and 3 text output only
```

**Key property:** Visual tokens V ∈ ℝ^{T×N×D} are available before the projector as the insertion point for Task 2's detection/classification heads. Tasks 1 and 3 use the full backbone end-to-end.

---

## Task 1 — BDD-X: Video → Action + Justification Text

**Module:** Standard LM head (no addition).  
**Trainable:** LoRA (LLM, r=16) + projector MLP.  
**ViT:** Frozen.

**Prompt format:**
```
User:    [video] Describe what the vehicle is doing, then explain why.
Model:   The car slows down. Because the light ahead has turned red.
```

**Loss:** Cross-entropy on generated tokens (action sentence + justification sentence).

**Data format (BDD-X CSV → instruction pairs):**
- `Answer.Naction` → action sentence target
- `Answer.Njustification` → justification sentence target
- One instruction pair per action segment; concatenate if multiple actions in clip

---

## Task 2 — ROAD-R: Video → Tubes + Constraint-Compliant Triplets

**LLM is bypassed.** All outputs come from heads grafted directly onto ViT visual tokens V.

### 2A — Detection Head (per-frame box regression)

```
V ∈ ℝ^{T×N×D}
  ↓  (per spatial token, per frame)
Linear(D → 4)  →  (cx, cy, w, h) box offsets
  ↓
NMS per frame
  ↓  B_t = {box_i^t} per frame t
```

**Loss:** GIoU + L1 on matched GT boxes (Hungarian matching for training assignment).

### 2B — Tube-Linking Module (cross-frame attention)

```
B_t = {ROI-pooled features from V at detected boxes, frame t}
         ↓
Single cross-frame attention layer:
  Q = B_t features  K,V = B_{t-1..t+k} features
  → attended tube features F_tube ∈ ℝ^{K×D}
         ↓
Tube IDs assigned by greedy matching on attention weights
```

Lightweight — single attention layer, trainable. Directly optimizes video-mAP by providing temporally-coherent features before classification.

### 2C — Classification Heads (multi-label sigmoid)

```
F_tube ∈ ℝ^{K×D}  (K detected agents, D features)
  ↓
Agent head:    Linear(D → 10)  → sigmoid  → p(agent)
Action head:   Linear(D → 19)  → sigmoid  → p(action)
Location head: Linear(D → 10)  → sigmoid  → p(location)
Duplex head:   Linear(D → 49)  → sigmoid  → p(duplex)
Triplet head:  Linear(D → 86)  → sigmoid  → p(triplet)
```

**Multi-label rationale:** Each agent may simultaneously exhibit multiple valid label combinations (e.g., a pedestrian can be both `Ped` and `MovAway` + `VehLane`). Sigmoid per class is the correct formulation.

### 2D — T-norm Constraint Loss (Łukasiewicz)

Applied on duplex and triplet outputs to enforce the 49 valid duplexes and 86 valid triplets from ROAD-Waymo's `duplex_childs` / `triplet_childs`:

```python
# For each invalid (agent, action) combination:
violation = max(0, p(agent_i) + p(action_j) - 1)
L_tnorm = sum over all invalid pairs of violation
```

**Total Task 2 loss:**
```
L_road = L_det + L_cls + λ · L_tnorm
```

**Trainable for Task 2:** LoRA (ViT shallow, r=8) + detection head + tube-linking module + classification heads.

---

## Task 3 — CoVLA: Video → Captions + 10×3 Trajectory

### 3A — Caption Generation (LM head)

Same LM head as Task 1, prompted separately for two caption types:

```
User:    [video/frame] Describe the ego vehicle's behavior.
Model:   The ego vehicle is moving at a moderate speed and turning left. [plain_caption]

User:    [video/frame] What should the driver be careful of?
Model:   to pay attention to the traffic light and other vehicles on the road. [risk]
```

**Trainable:** LoRA (LLM, r=16) + projector.

### 3B — Trajectory Regression Head

```
V ∈ ℝ^{T×N×D}
  ↓  mean-pool over spatial tokens N
pooled ∈ ℝ^{T×D}  →  mean-pool over T frames  →  ℝ^D
  ↓
MLP: Linear(D → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 30)
  ↓  reshape
10 × 3  (10 waypoints × (x, y, z) in vehicle frame)
```

**Loss:** Smooth-L1 on 10×3 vs 10 sampled GT waypoints from CoVLA `states/*.jsonl` trajectory (subsample from 60 points at uniform stride).

**Comparison baseline:** CoVLA-Agent (ADE: 0.955 / FDE: 2.239 with predicted caption). This architecture should exceed that because Qwen2.5-VL has stronger video understanding than CLIP ViT-L used in CoVLA-Agent.

**Total Task 3 loss:**
```
L_covla = L_lm(captions) + λ · L_traj
```

**Trainable:** LoRA (LLM, r=16) + projector + trajectory head.

---

## Training Order (Confirmed 2026-04-10)

Dr. Moradi confirmed sequential training order — ROAD-R first, then BDD-X, then CoVLA, then joint:

| Step | Dataset | Status |
|------|---------|--------|
| Experiment 1 | ROAD-R (GT-box classification) | Running — ViT frozen, GT boxes for ROI-pool, triplet/duplex + t-norm |
| Experiment 1b | ROAD-R (+ detection head) | After Exp 1 converges |
| Experiment 2 | BDD-X | After ROAD-R converges |
| Experiment 3 | CoVLA | After BDD-X |
| Joint | All three | After all separate models converge |

---

## Experiment 1 vs. 1b — Staged Detection Design

### Why Experiment 1 uses GT boxes instead of a learned detection head

The full Task 2 pipeline (detect → ROI-pool → tube-link → classify) couples two hard problems. If both are trained simultaneously from scratch and something fails, it is impossible to determine whether the issue is in detection or classification.

Experiment 1 deliberately **decouples detection from classification** by using ground-truth boxes for ROI-pooling. This makes the first experiment a **classification oracle**: given perfect localization, how well can the ViT features + heads learn the label taxonomy and t-norm constraints?

**What Experiment 1 validates:**
- ViT feature quality for agent/action/loc discrimination
- ROI-pool correctness (merged token spatial layout, box-to-token mapping)
- Tube-linking module (cross-frame attention over agent features)
- All five classification heads (agent, action, loc, duplex, triplet)
- T-norm loss behaviour and constraint violation rate over training

**What Experiment 1 cannot do:**
- Run at inference time without GT boxes (not a deployable detector)
- Measure detection-specific metrics (frame-mAP, video-mAP)

**Why this is the right order:**
Classification-first provides a performance ceiling: if the oracle model (perfect boxes) fails to learn the label taxonomy, a learned detector will also fail. If it converges, the classification heads are warm-started for Experiment 1b, which adds detection on top.

### Experiment 1b — Adding the Detection Head

Once Experiment 1 converges:

1. Load Experiment 1 checkpoint (classification heads + tube-linking trained)
2. Add detection head: `Linear(3584 → 4)` on ViT spatial tokens per frame
3. Add Hungarian matching loss: GIoU + L1 against GT boxes
4. Train detection head while classification heads continue fine-tuning
5. Full loss: `L_det + L_cls + λ · L_tnorm`

Warm-starting classification heads from Experiment 1 means the detection head trains against a stable classification signal rather than random noise — substantially easier to converge.

### Alternative: joint detection + classification from scratch

Training detection and classification jointly from the start (Option B) is valid and closer to how 3D-RetinaNet is trained. It is faster to a deployable model but carries higher risk:
- Hungarian matching is sensitive to initialization
- If detection diverges early, classification heads receive garbage ROI features and also diverge
- Hard to distinguish detection failure from classification failure in the loss curve

Option B becomes reasonable once the staged approach has confirmed the classification pipeline works.

---

## Joint Training (Phase 4)

After all three separate models converge, merge adapters and train jointly:

```
L_joint = L_bddx + α · L_covla + β · L_road

Task sampling per batch:
  BDD-X:   20%
  CoVLA:   40%
  ROAD-R:  40%
```

α and β tuned so gradient magnitudes are approximately equal across tasks at the start of joint training. ROAD-R weighted highest because it has the most structurally complex outputs.

**Shared during joint training:** ViT encoder (with LoRA adapters active for whichever task is active), projector. Task-specific: each task's LoRA set + output heads.

---

## Full Pipeline Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │           Qwen2.5-VL-7B Backbone                │
                    │                                                 │
  Video/frame  →   │  ViT (dynamic res, window+full attn, mRoPE)     │
                    │    ↓  V ∈ ℝ^{T×N×D}                            │
                    │  MLP Projector  →  Qwen2.5-7B LLM              │
                    └──────┬──────────────────────┬────────────────── ┘
                           │                      │
               ┌───────────┤                      ├───────────────────┐
               │    Task 2 │                      │ Tasks 1 & 3       │
               ▼           │                      ▼                   │
   ┌─────────────────────┐ │         ┌─────────────────────────────┐  │
   │  Detection Head     │ │         │  LM Head (text generation)  │  │
   │  Linear(D→4)        │ │         │                             │  │
   │  NMS                │ │         │  Task 1 (BDD-X):            │  │
   │       ↓             │ │         │   "Describe + explain why"  │  │
   │  Tube-Linking       │ │         │   → action + justification  │  │
   │  Cross-frame attn   │ │         │                             │  │
   │       ↓  F_tube     │ │         │  Task 3 (CoVLA):            │  │
   │  Classification     │ │         │   "Describe behavior"       │  │
   │  5× sigmoid heads   │ │         │   "What to be careful of"   │  │
   │  (10/19/10/49/86)   │ │         │   → plain_caption + risk    │  │
   │       ↓             │ │         └─────────────────────────────┘  │
   │  T-norm loss        │ │                                           │
   └─────────────────────┘ │         ┌─────────────────────────────┐  │
                           │         │  Trajectory Head (Task 3)   │  │
                           │         │  mean-pool V → MLP(D→256    │  │
                           │         │  →128→30) → 10×3 waypoints  │  │
                           └─────────└─────────────────────────────┘  
```

---

## Hardware Requirements

| Config | VRAM | Notes |
|--------|------|-------|
| 7B base + LoRA r=16 + task heads | ~20–24 GB | Single A100 80GB or RTX 4090 feasible with gradient checkpointing |
| 3B base (ablation) | ~10–12 GB | RTX 3090 feasible |
| Joint training (all heads active) | ~24–28 GB | May need bf16 + gradient checkpointing |

---

## Related

- [[directions/qwen-multitask-finetuning|Approach 3 research direction]]
- [[directions/constrained-vlm-reasoning|Approach 4 — OpenMixer + DSDAG]] — causal head; this is its upstream
- [[datasets/bdd-x|BDD-X]] — Task 1
- [[datasets/covla|CoVLA]] — Task 3
- [[datasets/road-plusplus|ROAD++]] — Task 2
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — t-norm background
- [[papers/covla-2025|CoVLA paper]] — CoVLA-Agent baseline for trajectory comparison
