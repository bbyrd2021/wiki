---
type: concept
title: "Gaze and Attention in Pedestrian Intent"
aliases: ["gaze", "look label", "attention", "looking"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD_analysis/cross_tabulations.md"
  - "PedestrianIntent++/PIE_analysis/cross_tab_results.json"
tags: [concept, gaze, jaad, pie, crossing-intent]
status: complete
---

# Gaze and Attention

The `look` label in JAAD and PIE records whether a pedestrian is looking toward the ego-vehicle. Its predictive meaning **reverses between datasets** — this is one of the most important findings in [[projects/pedestrian-intent|PedestrianIntent++]].

## JAAD: Looking = Commitment

At the decision_point:

| Action | Look | Cross Rate |
|--------|------|-----------|
| walking | **looking** | **95.7%** |
| standing | looking | 84.8% |
| walking | not-looking | 62.9% |
| standing | **not-looking** | **44.9%** |

Interpretation: In JAAD footage (European/NA residential streets), gaze toward the vehicle signals **acknowledgment of the driver** — an implicit social negotiation meaning "I see you, I'm going to cross."

## PIE: Looking = Hesitation

At the critical_point:

| Action | Look | Cross Rate |
|--------|------|-----------|
| walking | **not-looking** | **74.1%** |
| walking | looking | 56.0% |
| standing | not-looking | 21.8% |
| standing | looking | 14.2% |

Interpretation: In PIE footage (Toronto urban intersections), looking toward the vehicle signals **caution or hesitation** — the pedestrian is checking whether it's safe, not committing to cross. A pedestrian who has already committed walks without looking.

See [[findings/pie-gaze-reversal|PIE Gaze Reversal]].

## Majority-Look Across Full Track (JAAD)

| Majority | Cross Rate |
|----------|-----------|
| majority looking | 85.3% |
| majority not-looking | 85.4% |

**Surprising: aggregate gaze predicts nothing.** Only the snapshot at the decision moment matters.

## ROAD++: No Gaze Annotation

ROAD++ has no `look` label. The dataset covers intent behavior through action classes (`Wait2X`, `XingFmLft`) rather than gaze.

## Implications for Models

1. **Do not transfer gaze features across datasets.** A model trained on JAAD gaze will learn the opposite mapping from one trained on PIE.
2. **Context is key.** The same visual signal (looking at vehicle) has opposite meanings depending on geographic/cultural context and protocol.
3. **Action dominates gaze.** Walking is a stronger crossing predictor than gaze in both datasets.
4. Papers (PCPA, PedFormer, GTransPDM) treat gaze as one feature in a multimodal model — gaze alone is insufficient.

## Related

- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] | [[findings/pie-gaze-reversal|PIE Gaze Reversal]]
- [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze Comparison]]
- [[concepts/crossing-intent|Crossing Intent]] | [[directions/cross-dataset-generalization|Cross-Dataset Generalization]]
