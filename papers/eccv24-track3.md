---
type: paper
title: "First Place: ECCV 2024 ROAD++ Challenge Track 3 — Atomic Activity Recognition (TACO)"
authors: "Li et al. (Inspur)"
year: 2024
venue: "ECCV 2024 Workshop (challenge tech report)"
arxiv: "2410.23092"
created: 2026-04-07
updated: 2026-08-27
sources: ["wiki/raw/ECCV2024_ROAD_Track3_TACO_FirstPlace.pdf"]
tags: [paper, eccv24-challenge, activity-recognition, taco, ensembling, internvideo]
status: complete
datasets_used: []
---

# ECCV 2024 ROAD++ Track 3 Winner — Atomic Activity Recognition on TACO

**Li, Zhang, Zhang, Liu, Wang, Li (Inspur, Beijing) · arXiv:2410.23092 · 69% test mAP (baseline Action-slot: 54%)**

**Correction (2026-08-27, from the primary PDF):** this page previously
conflated Track 3 with ROAD-Waymo. Track 3 is a *different task on a
different dataset*: multi-label **atomic activity recognition on TACO**
(simulated), where the 64 classes are region-to-region movements over road
topology — e.g. `Z1-Z4:C+` = a group of vehicles traveling from road Z1 to
road Z4. No detection; no ROAD-Waymo label vocabulary. The prior body's
claims (anonymous authors, Kinetics/ActivityNet pretraining, Wait2X/Xing
classes) were artifacts of a secondary source.

## Method (actual)

Base model: **Action-slot** (Kung et al. 2024, the TACO paper's own model).
Three challenge diagnoses — small objects, single-object vs object-group
confusion, and overfitting to the simulation's regularity — attacked with:

1. **Multi-branch framework**: separate models per object category and for
   single-vs-group (pedestrian / vehicle / two-wheeler × single / group),
   merged with the standard 64-class model. Same per-category divide-and-
   conquer the [[papers/eccv24-track1|Track-1 winners]] used for detection.
2. **Four-axis ensembling**: all frame-sampling sequences at test time (not
   just the middle one); sequence lengths 16/32/64; checkpoints from epochs
   60–100; two backbones — **X3D and InternVideo**.
3. **Augmentation**: Cutout + horizontal video flip with topology-aware label
   remapping (flipping swaps left/right road regions, so labels must remap).
4. **2× input upsampling** for small objects.

Training: AdamW, lr 5e-4, wd 0.07, batch 8, 100 epochs.

## Why the lab cares

- **InternVideo as a backbone in a winning road-scene system** — external
  precedent for the [[findings/exp12-phrase-head-attribution|exp12]]
  InternVideo2 direction (they used v1 as one of two ensemble backbones).
- **The resolution lesson, third independent confirmation**: their answer to
  small objects is upsampling the input — same bottleneck we measured in
  exp2 and again in the exp12 encoder swap.
- **Overfitting regime contrast** ([[findings/road-waymo-schedule-overtraining]]):
  on simulated TACO with AdamW they found best models at epochs 60–100 and
  ensembled across epochs; on real ROAD-Waymo with the ECCV recipe our
  detectors peaked at epochs 1–2. Different data, task, and optimizer — cite
  as contrast, never as contradiction.
- **Ensembling-heavy, architecture-light**: like Track 1, the winning margin
  is engineering; no representational contribution — the gap our thesis
  targets.

## Related

- [[papers/eccv24-track1|ECCV 2024 Track 1 winner]] — the agent-detection sibling report (ROAD-Waymo, our 31.6 anchor)
- [[papers/wang-2024-internvideo2|InternVideo2]] — successor of their ensemble backbone
- [[datasets/road-plusplus|ROAD++]] — for the challenge context only; TACO is a separate dataset
