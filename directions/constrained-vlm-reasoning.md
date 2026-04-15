---
type: direction
title: "Constrained VLM Reasoning (ROAD_Reason Approach 4)"
aliases: ["constrained VLM", "OpenMixer DSDAG", "Approach 4"]
created: 2026-04-07
updated: 2026-04-07
architecture_spec: "methods/multimodal-causal-driving"
sources: ["ROAD_Reason/docs/APPROACHES.md", "architecture-spec-2026-04-07"]
tags: [direction, road-plusplus, vlm, causal, t-norm, primary-contribution]
status: complete
novelty: true
feasibility: partial
datasets_required: [road-plusplus]
approach_number: 3
---

# Constrained VLM Reasoning (Primary Thesis Contribution)

**Approach 4** in [[projects/road-reason|ROAD_Reason]]. The primary novel thesis contribution.

## Core Insight

ROAD++ labels capture *what* happens (`Ped-Wait2X-RhtPav`) but not *why*. Identical triplets arise from different causal origins (genuine hazard stop vs. false positive stop). A causal reasoning head distinguishes these origins and produces grounded natural language explanations.

## Architecture

```
Video clip → CLIP-ViP (frozen) → OpenMixer backbone
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
          Structured head (raw triplet logits)    DSDAG causal head
                                                  VLT → reasoning embedding
                    │                                       │
                    └──────────── f(reasoning) ────────────┘
                                       │
                              reweighted triplet logits
                                       │
                              T-norm constraint loss
```

## Key Components

| Component | Source | Role |
|-----------|--------|------|
| OpenMixer | Bao et al., WACV 2025 | DETR-style detector on frozen CLIP-ViP |
| DSDAG | Cheng et al. (MCAM, ICCV 2025) | Causal DAG with hidden danger state W |
| VLT | Custom | Vision-Language Transformer, sparsity loss (anti-hallucination) |
| T-norm loss | Marconato et al. | Constraint compliance on valid duplexes/triplets |
| Entropy-gated fusion | Chlon et al. 2025 | Structured vs causal head selection at inference |

## Training Strategy

ROAD-Waymo has no explanation captions. Language supervision is handled in three stages:

1. **VLT pre-training on BDD-X + CoVLA** — grounds the VLT in language and driving vocabulary before touching ROAD-Waymo
2. **Structured head warm-up + VLT fine-tuning on ROAD-Waymo** — VLT uses structured label reconstruction (triplet prediction from r) as proxy supervision; no captions required; language output at inference is emergent from Stage 1
3. **Joint fine-tuning** — ROAD-Waymo GT labels + optional **Claude-generated pseudo-captions** (Opus 4.6 for corner cases, Sonnet 4.6 for bulk) via Anthropic Message Batches API for corner-case augmentation

## The "Right Label, Wrong Reason" Problem

Identical triplet labels arise from different causal origins:
- `LarVeh-Stop` caused by a plastic bag in the road ≠ `LarVeh-Stop` caused by a genuine hazard
- Constraints enforce *validity* but cannot resolve this — only the causal head can

## Three Representation Spaces

| Space | Contains |
|-------|---------|
| Perception | Boxes, tracks, queries |
| Structured Event | Agent, action, location, triplets |
| **Causal Reasoning** | Intent, hidden danger W, explanation |

## Key Equation

`L_final = L_raw ⊙ f(r)`

Reasoning embedding `r` is mapped to a weight vector `w = f(r)` that gates the raw triplet logits before t-norm constraint loss. The causal path is end-to-end differentiable.

## DSDAG Structure

```
Start State (Xs) → Action (Y) → End State (Xe)
                       ↑
                  Environment (Z)
```

- **Xs**: derived from early frames of clip
- **Xe**: derived from late frames of clip
- **Z**: global CLIP-ViP embedding `f_v` (produced for free alongside V; captures road geometry, weather, intersection type, traffic signal state — no separate module needed)

## Full Architecture

See [[methods/multimodal-causal-driving|Multimodal Causal Driving Model — Architecture Spec]] for complete module breakdown with inputs/outputs and defense preparation.

## Why Novel

First work combining:
- Open-vocabulary action detection (OpenMixer)
- Causal state modeling (DSDAG)
- Neuro-symbolic label constraints (t-norm)

on ROAD-Waymo. Introduces reasoning-driven logit modulation — causal reasoning *corrects* structured predictions, not a parallel explanation module.

## Related

- [[projects/road-reason|ROAD_Reason]] | [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[directions/qwen-multitask-finetuning|Qwen2.5-VL Multi-Task (Approach 3)]] — upstream baseline; its M_bddx + M_covla weights seed Stage 1
- [[directions/jepa-intent-head|V-JEPA 2 Intent Head (Approach 5)]]
- [[datasets/road-plusplus|ROAD++]]
