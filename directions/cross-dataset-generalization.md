---
type: direction
title: "Cross-Dataset Generalization"
aliases: ["cross-dataset generalization", "domain adaptation", "dataset transfer"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [direction, jaad, pie, road-plusplus, domain-shift, generalization]
status: complete
novelty: true
feasibility: workstation
datasets_required: [jaad, pie, road-plusplus]
---

# Cross-Dataset Generalization

**Direction 3** from SYNTHESIS.md Part 7.

## Motivation

ROAD++ (USA Waymo, 798 clips, 712K ped boxes) → JAAD (Europe/NA, 686 behavioral peds) → PIE (Toronto, 1842 peds) represent complementary scales and annotation philosophies. All papers train and test on the same dataset. ROAD++ paper shows 3D-RetinaNet drops substantially when transferred from ROAD (UK) to ROAD-Waymo (USA). A similar gap should exist across all three datasets.

## Approach A: ROAD++ as Pretraining Source

Use ROAD++ 712K pedestrian boxes as large-scale pretraining for pedestrian detectors and scene encoders. Fine-tune zero-shot or few-shot on JAAD/PIE.

**Why promising:** ROAD++ dwarfs JAAD/PIE 400:1 in pedestrian box count. Pretraining on diverse Waymo footage should improve robustness.

## Approach B: Bridge Labels

Use JAAD/PIE intent labels to annotate the crossing-intent implied by ROAD++'s `Wait2X`/`Xing` action labels. Create bridge labels: `Wait2X` ≈ "about to cross", enabling ROAD++ to serve as an intent pretraining source.

## Key Challenges

- **Gaze reversal**: features learned on JAAD will encode the wrong gaze→intent mapping for PIE
- **Frame rate**: JAAD/PIE at 30 FPS vs. ROAD++ at 10 FPS
- **Camera**: fisheye (PIE) vs. standard (JAAD) vs. Waymo (ROAD++)

## What's Missing

No cross-dataset generalization study spans all three datasets. No paper uses ROAD++ for pedestrian pretraining in JAAD/PIE.

## Related

- [[concepts/domain-shift|Domain Shift]] | [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]]
- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]] | [[datasets/road-plusplus|ROAD++]]
