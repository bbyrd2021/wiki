---
type: direction
title: "Qwen2.5-VL Multi-Task Fine-Tuning (Approach 3)"
aliases: ["Approach 3", "Qwen multi-task", "VLM baseline"]
created: 2026-04-10
updated: 2026-04-10
sources: []
tags: [direction, approach3, qwen, vlm, multi-task, bdd-x, covla, road-r, road-reason]
status: draft
novelty: true
feasibility: workstation
datasets_required: [bdd-x, covla, road-plusplus]
---

# Qwen2.5-VL Multi-Task Fine-Tuning (Approach 3)

**Supervisor direction (Dr. Moradi)** — train a SOTA VLM separately on BDD-X, CoVLA, and ROAD-Waymo with task-specific I/O modules, then jointly. Established as Approach 3 in the research roadmap, preceding the full causal architecture (Approach 4).

## Motivation

Before building the novel DSDAG causal head (Approach 4), establish:
1. A strong SOTA VLM baseline on all three datasets
2. Language-grounded weights that can seed Approach 4's Stage 1 pre-training
3. A detection + t-norm ceiling on Qwen2.5-VL features to benchmark against OpenMixer
4. An independently publishable multi-task contribution

## Backbone: Qwen2.5-VL-7B

Selected over Qwen2.5-Omni (professor's initial suggestion) because Omni adds audio encoder + Talker speech synthesis — both irrelevant to this project, adding VRAM overhead with no benefit.

**Qwen2.5-VL-7B key properties:**
- ViT with dynamic resolution (window + 4 full attention layers), SwiGLU, RMSNorm, 2D-RoPE
- Video: dynamic FPS sampling, mRoPE with absolute temporal IDs → second-level event localization
- Native bounding box output in JSON: `{"bbox_2d": [x1,y1,x2,y2], "label": "..."}`
- Apache 2.0 license — open for research and publication
- ~16–20 GB VRAM for 7B; 3B variant available for ablations
- Mature fine-tune ecosystem: LLaMA-Factory, Qwen-VL-Finetune, 227+ community LoRA models

## Three Tasks

| Task | Dataset | Input | Output | Module |
|------|---------|-------|--------|--------|
| T1 | BDD-X | Video clip (~720p) | Action text + Justification text | LM head (standard) |
| T2 | ROAD-R | Video clip (ROAD-Waymo) | Tubes + multi-label triplet classifications (t-norm) | Detection + classification + tube-linking heads |
| T3 | CoVLA | Video frame / short clip | plain_caption + risk text + 10×3 waypoints | LM head + trajectory regression head |

Full architecture spec: [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task Architecture]]

## Training Strategy

Training is **sequential by dataset**, not parallel. Dr. Moradi confirmed order on 2026-04-10:

**Experiment 1 — ROAD-R, classification with GT boxes (running):**
- Input: video frames + ROAD-Waymo GT labels
- Backbone: Qwen2.5-VL-7B ViT encoder (frozen); merger output D=3584
- Localization: GT boxes used for ROI-average-pooling (no learned detection)
- Output: triplet + duplex + agent/action/loc classifications
- Loss: BCE (5 heads) + Łukasiewicz t-norm
- Trainable: tube-linking module + classification heads (~52M params)
- Purpose: validates ViT features + heads in isolation; produces warm-start weights for Exp 1b
- See [[methods/qwen25-vl-multitask#experiment-1-vs-1b--staged-detection-design|Staged detection rationale]]

**Experiment 1b — ROAD-R, + detection head (after Exp 1 converges):**
- Loads Exp 1 checkpoint; adds Linear(3584→4) detection head
- Trains detection (GIoU + L1 + Hungarian matching) with classification heads fine-tuning
- Full loss: L_det + L_cls + λ·L_tnorm
- Trainable: detection head + tube-linking + classification heads (+ LoRA r=8 on ViT shallow)

**Experiment 2 — BDD-X (after ROAD-R converges):**
- Input: video clip
- Output: action text + justification text
- Loss: L_lm; trainable: LoRA (LLM r=16) + projector

**Experiment 3 — CoVLA (after BDD-X):**
- Input: video frame / short clip
- Output: plain_caption + risk text + 10×3 trajectory waypoints
- Loss: L_lm + λ·L_traj; trainable: LoRA (LLM r=16) + projector + trajectory head

**Phase 4 — Joint training (after all three converge):**
```
L_joint = L_bddx + α·L_covla + β·L_road
Task sampling: BDD-X 20% · CoVLA 40% · ROAD-R 40%
```

## Evaluation

| Task | Primary metrics |
|------|----------------|
| BDD-X | BLEU-4, CIDEr, METEOR on action + justification (vs human captions) |
| CoVLA | ADE / FDE on 10-pt trajectory; BLEU/CIDEr on plain_caption + risk; compare to CoVLA-Agent (ADE 0.955 / FDE 2.239) |
| ROAD-R | frame-mAP / video-mAP per label type; constraint violation rate vs Approach 2 |

## Position in Research Roadmap

```
Approach 1 → Approach 2 → [Approach 3: this] → Approach 4 (causal) → Approach 5 → Approach 6
```

M_bddx and M_covla weights are reused as the starting point for Approach 4 Stage 1 pre-training — reducing compute and providing better initialization than random.

## Related

- [[methods/qwen25-vl-multitask|Architecture spec]]
- [[datasets/bdd-x|BDD-X]] — Task 1 dataset
- [[datasets/covla|CoVLA]] — Task 3 dataset
- [[datasets/road-plusplus|ROAD++]] — Task 2 dataset
- [[comparisons/bdd-x-vs-covla|BDD-X vs CoVLA]] — comparison of Task 1 and Task 3 sources
- [[directions/constrained-vlm-reasoning|Approach 4 (was Approach 3)]] — causal head; Approach 3 is its upstream
