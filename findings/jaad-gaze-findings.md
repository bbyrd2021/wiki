---
type: finding
title: "JAAD Gaze Findings at Decision Point"
aliases: ["JAAD gaze", "walking looking cross rate"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/JAAD_analysis/cross_tabulations.md"
  - "PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [finding, jaad, gaze, cross-tabulation, decision-point]
status: complete
---

# JAAD Gaze Findings at Decision Point

Verified from all 686 JAAD behavioral pedestrians, parsed from 346 XML annotation files.

## Cross-Tabulation at decision_point

| Action | Look | Crossed | Not Crossed | N | **Cross Rate** |
|--------|------|---------|-------------|---|---------------|
| walking | looking | 176 | 8 | 184 | **95.7%** |
| standing | looking | 67 | 12 | 79 | 84.8% |
| walking | not-looking | 90 | 53 | 143 | 62.9% |
| standing | not-looking | 22 | 27 | 49 | **44.9%** |

## Cross-Tabulation in Last 15 Frames

| Action × Look | Crossed | Not Crossed | N | Cross Rate |
|---------------|---------|-------------|---|-----------|
| walking+looking | 71 | 3 | 74 | **95.9%** |
| walking+not-looking | 330 | 59 | 389 | 84.8% |
| standing+looking | 42 | 10 | 52 | 80.8% |
| standing+not-looking | 33 | 28 | 61 | **54.1%** |

## Key Insights

1. **Walking+looking → near-certain crossing**: 95.7% cross rate at decision_point. The safest possible prediction for an AV to yield.
2. **Standing+not-looking → highest ambiguity**: 44.9% cross rate — essentially a coin flip. Additional context (traffic signal, crossing zone) needed.
3. **Gaze is a strong secondary signal** in JAAD: within each action category, looking shifts cross rate by ~33 percentage points.
4. **Action (walking vs. standing) is the primary discriminant** — a 33pp gap between walking+not-looking (62.9%) and standing+not-looking (44.9%) is smaller than the gap caused by action alone.

## Important Contrast

These findings **do not generalize to PIE**. In PIE, walking+not-looking has a *higher* cross rate than walking+looking. See [[findings/pie-gaze-reversal|PIE Gaze Reversal]].

## Source

`PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json`
Visualization: `JAAD_analysis/viz/analysis1_action_look_crosstab.png`

## Related

- [[concepts/gaze-and-attention|Gaze and Attention]] | [[concepts/action-look-crosstab|Action × Look Cross-Tabulation]]
- [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze Comparison]]
- [[datasets/jaad|JAAD Dataset]]
