---
type: concept
title: "Crossing Intent"
aliases: ["intent", "pedestrian intent", "crossing intention"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD/JAAD_summary.md"
  - "PedestrianIntent++/PIE/PIE_Dataset_Summary.md"
tags: [concept, crossing-intent, jaad, pie]
status: complete
---

# Crossing Intent

Pedestrian crossing intent is the mental state predicting whether a pedestrian will cross the road in front of an ego-vehicle. Distinct from the **crossing action** (observed behavior).

## Intent vs. Action

| | Intent | Action |
|-|--------|--------|
| What | Mental state (will they cross?) | Observed behavior (are they crossing?) |
| When | Before crossing | During/after crossing |
| Labels | `intention_prob` (PIE), binary at decision_point (JAAD) | `cross=1` label, `XingFmLft` (ROAD++) |
| Challenge | Must be predicted from pre-crossing cues | Observable directly |

## JAAD Definition

Binary per-pedestrian outcome: does this pedestrian cross? Established at annotation time. The `decision_point` is the annotator-marked last viable decision moment — models must predict crossing outcome using only frames ≤ decision_point.

**Key signals at decision_point:**
- walking + looking → 95.7% cross (see [[findings/jaad-gaze-findings]])
- standing + not-looking → 44.9% cross

## PIE Definition

Binary crossing outcome plus `intention_prob` — crowd-sourced probability from up to 30 human judges who watched the clip ending at `critical_point` (before any crossing occurred). Models should use only frames ≤ critical_point.

**critical_point → crossing_point gap:** median 30 frames = 1 second. This is the prediction horizon.

See [[concepts/intention-probability|Intention Probability]] for the full distribution.

## ROAD++ Equivalent

No explicit intent label. Behavioral states closest to intent:
- `Wait2X` (54,928 instances) — actively waiting to cross
- `XingFmLft` / `XingFmRht` / `Xing` — mid-crossing actions (not intent, outcome)

ROAD++ cannot be used as a crossing-intent benchmark without additional annotation.

## Why Intent is Hard

1. Identical observable states (standing, looking) can reflect different mental states
2. Gaze toward the vehicle means **commitment** in JAAD but **hesitation** in PIE — context-dependent
3. No ground-truth access to mental states; all labels are proxy measures
4. The prediction horizon (1 second) requires the AV to respond well before the decision is unambiguous

## Related

- [[concepts/gaze-and-attention|Gaze and Attention]] | [[concepts/intention-probability|Intention Probability]]
- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]] | [[datasets/road-plusplus|ROAD++]]
- [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]] | [[comparisons/dataset-comparison|Dataset Comparison]]
