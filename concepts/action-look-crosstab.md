---
type: concept
title: "Action × Look Cross-Tabulation"
aliases: ["cross-tabulation", "crosstab", "action-look analysis"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/JAAD_analysis/cross_tabulations.md"
  - "PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json"
  - "PedestrianIntent++/PIE_analysis/cross_tab_results.json"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [concept, cross-tabulation, jaad, pie, gaze, action]
status: complete
---

# Action × Look Cross-Tabulation

Analysis of how the combination of pedestrian action (walking vs. standing) and gaze (looking vs. not-looking) predicts crossing outcome. The primary empirical analysis in [[projects/pedestrian-intent|PedestrianIntent++]].

## Methodology

1. For each behavioral pedestrian in JAAD/PIE, extract the label state at the decision/critical point
2. Group by (action × look) combination
3. Compute crossing rate for each group
4. Analyze the interaction: does gaze matter, and how does it interact with action?

## JAAD Results (decision_point)

| Action | Look | Crossed | Not Crossed | N | Cross Rate |
|--------|------|---------|-------------|---|-----------|
| walking | looking | 176 | 8 | 184 | **95.7%** |
| standing | looking | 67 | 12 | 79 | 84.8% |
| walking | not-looking | 90 | 53 | 143 | 62.9% |
| standing | not-looking | 22 | 27 | 49 | **44.9%** |

Source: `JAAD_analysis/analysis/cross_tab_results.json`

## PIE Results (critical_point)

| Action | Look | Crossed | Not Crossed | N | Cross Rate |
|--------|------|---------|-------------|---|-----------|
| walking | not-looking | 275 | 96 | 371 | **74.1%** |
| walking | looking | 65 | 51 | 116 | 56.0% |
| standing | not-looking | 152 | 545 | 697 | 21.8% |
| standing | looking | 27 | 163 | 190 | **14.2%** |

Source: `PIE_analysis/cross_tab_results.json`

## Key Insight: The Gaze Reversal

In JAAD, looking+walking → highest cross rate. In PIE, not-looking+walking → highest cross rate. Action (walking vs. standing) is consistent across datasets as the dominant predictor. Gaze is a **secondary signal whose meaning reverses by dataset context**.

See [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze Comparison]] for full analysis.

## Visualizations

- `JAAD_analysis/viz/analysis1_action_look_crosstab.png`
- `JAAD_analysis/viz/analysis2_decision_point_labels.png`
- `JAAD_analysis/viz/analysis3_cross_label_timing.png`

## Related

- [[concepts/gaze-and-attention|Gaze and Attention]] | [[concepts/crossing-intent|Crossing Intent]]
- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] | [[findings/pie-gaze-reversal|PIE Gaze Reversal]]
