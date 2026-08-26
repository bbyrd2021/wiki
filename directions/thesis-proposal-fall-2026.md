---
type: direction
title: "Thesis Proposal — Hybrid Detection + Language-Geometry Composition (Fall 2026)"
aliases: []
created: 2026-08-26
updated: 2026-08-26
sources:
  - "ROAD_Reason/experiments/exp11_yolo"
tags: [direction, thesis, proposal, road-plusplus, contrastive, exp11]
status: complete
novelty: true
feasibility: workstation
datasets_required: [road-plusplus]
---

# Thesis Proposal — Hybrid Detection + Language-Geometry Composition

The proposal of record (skeleton drafted 2026-08-25, updated 08-26 after the
[[methods/stacked-composition-mlp|stacked-MLP]] sweep; full outline in
`~/Downloads/thesis_proposal_skeleton.md`; pending Moradi meeting).

## Three-act structure

1. **Attribution studies (done, exp5–exp9):** the language trilogy negative
   ([[findings/exp8-joint-lora-fusion|fusion]], text supervision, co-training
   — [[findings/exp9-attribution-grid|the grid]]); the
   [[findings/road-waymo-childs-mis-indexed|childs-bug]] discovery and
   corrected constraint sets.
2. **The hybrid platform (done, exp11):** YOLOv8x box source (35.24 agent;
   [[findings/road-waymo-schedule-overtraining|over-training finding]]);
   RoIAlign + stacked-MLP hybrid beating the official baseline on **all six
   heads** ([[findings/exp11-yolo-hybrid]]).
3. **Proposed: contrastive phrase head** (Moradi) — per-composition text
   embeddings in the InternVideo2 video-text space, on the same pooled boxes.
   Gate: beat the stacked-MLP sweep (duplex 15.52 / triplet 9.73) on
   identical rows; signature evidence = long-tail per-class gains (Amber,
   Ovtak, EmVeh). Supporting cells: encoder swap, temporal-clip features,
   WeDetect crop-vs-featuremap variant.

## Constraint arc (secondary contribution)

Three measured conditions: soft t-norm binds through a coupled backbone
(exp9), is structurally inert through independent linear heads (exp11 λ
sweep), and is superseded by hard vocabulary constraints (stacked MLP: zero
composed violations by construction). See
[[concepts/neuro-symbolic-constraints]].

## Timeline

Sept: proposal defense, phrases.json review, InternVideo2 dumps on YOLO rows.
Sept–Oct: phrase-head cells + ablations; optional test-set v-mAP submission.
Nov: writing (wiki findings pages are chapter drafts); defense. Fallback
claims if the phrase head fails its gate: the platform sweep, the constraint
arc, the over-training finding, the corrected constraint sets.
