---
type: direction
title: "Thesis Proposal — Hybrid Detection + Language-Geometry Composition (Fall 2026)"
aliases: []
created: 2026-08-26
updated: 2026-08-30
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

## Stretch goal — IDIL-shaped staged fine-tuning for the tail (Brandon 2026-08-30)

Adapt the IDIL idea from [[findings/sparse-temporal-pie|SparseTemporalPIE]] (Intention
Domain Incremental Learning: staged training along an ordered domain axis, which there
was time-to-event) to the thesis problem's ordered axis: **class frequency**. At the
fine-tuning rung (1B InternVideo2 + LoRA on crop inputs), train in staged tiers from
common to rare classes, protecting earlier tiers as rarer ones are admitted.

- Why it clears the evidence bar: staged/decoupled long-tail training has established
  literature (decoupled representation vs classifier, cRT: Kang et al. 2020) beyond the
  in-house IDIL result (pre-IL 0.854 to 0.926 accuracy on PIE).
- Why the fine-tuning rung: IDIL's gain in SparseTemporalPIE came from backbone
  adaptation across the staged curriculum; with frozen encoders and linear heads there
  is no backbone to adapt, so the idea only bites once LoRA unfreezes the encoder.
- Status: STRETCH — after the crop record row, the 1B encoder swap, and plain LoRA
  fine-tuning are secured; one controlled variable on top of the ladder.

## Timeline (compressed — defense by October, per Brandon 2026-08-26)

- Now → early Sept: Moradi meeting + **proposal defense**; writing starts
  immediately in parallel (wiki findings/method/concept pages are chapter
  drafts — attribution + platform chapters assemblable this week).
- Sept: InternVideo2 dumps + the two phrase-head cells, hard **mid-Sept
  go/no-go** on the phrase head.
- Early–mid Oct: draft complete to committee (~2 weeks before defense).
- **Late Oct: thesis defense.**
- Cut list (only if idle GPU + writing on schedule): coherence-penalty
  ablation, YOLO26 one-to-many re-dump, test-set v-mAP submission.
- The thesis stands without the phrase head: the six-head sweep, the
  attribution studies, the constraint arc, and the over-training finding are
  the committed claims; the phrase head is upside, not load-bearing.
