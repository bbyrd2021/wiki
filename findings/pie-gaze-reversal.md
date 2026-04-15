---
type: finding
title: "PIE Gaze Reversal at Critical Point"
aliases: ["PIE gaze reversal", "walking not-looking", "looking hesitation"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/PIE_analysis/cross_tab_results.json"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [finding, pie, gaze, cross-tabulation, critical-point, reversal]
status: complete
---

# PIE Gaze Reversal at Critical Point

Verified from all 1,842 PIE pedestrian tracks at critical_point, parsed from 6-set XML annotations.

## Cross-Tabulation at critical_point

| Action | Look | Crossed | Not Crossed | N | **Cross Rate** |
|--------|------|---------|-------------|---|---------------|
| walking | not-looking | 275 | 96 | 371 | **74.1%** |
| walking | looking | 65 | 51 | 116 | 56.0% |
| standing | not-looking | 152 | 545 | 697 | 21.8% |
| standing | looking | 27 | 163 | 190 | **14.2%** |

## The Reversal

In PIE: **walking+not-looking → 74.1% cross** — 18 percentage points *higher* than walking+looking (56.0%).

In JAAD: **walking+looking → 95.7% cross** — 33 percentage points *higher* than walking+not-looking (62.9%).

The sign of the gaze effect is opposite between the two datasets.

## Why the Reversal?

**Hypothesis:** In PIE's Toronto urban intersections, a pedestrian who has committed to crossing has already assessed the situation and is walking with purpose — no need to look at the vehicle. A pedestrian who is still looking toward the vehicle is **checking** — still deciding, still hesitant.

In JAAD's European/residential footage, looking at the vehicle reflects **social acknowledgment** — "I see you, you see me, I'm crossing." The cultural/geographic context and annotation protocol differ.

## Implications for Models

- **Never combine JAAD and PIE gaze features without controlling for dataset** — the same feature encodes opposite information
- Training on both datasets simultaneously may cause the model to learn a noisy gaze→intent mapping that is wrong for both
- This is one of the strongest arguments for [[directions/cross-dataset-generalization|cross-dataset generalization research]]

## Source

`PedestrianIntent++/PIE_analysis/cross_tab_results.json`

## Related

- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] | [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze Comparison]]
- [[concepts/gaze-and-attention|Gaze and Attention]] | [[datasets/pie|PIE Dataset]]
