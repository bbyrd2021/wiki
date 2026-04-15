---
type: comparison
title: "JAAD vs PIE: The Gaze Reversal"
aliases: ["JAAD PIE gaze comparison", "gaze reversal", "look reversal"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json"
  - "PedestrianIntent++/PIE_analysis/cross_tab_results.json"
tags: [comparison, jaad, pie, gaze, reversal, domain-shift]
status: complete
---

# JAAD vs PIE: The Gaze Reversal

One of the most important findings in [[projects/pedestrian-intent|PedestrianIntent++]]. The meaning of the `look` label is **opposite** between JAAD and PIE.

## The Numbers

| Dataset | Condition | Cross Rate | N |
|---------|-----------|-----------|---|
| **JAAD** | walking + **looking** | **95.7%** | 184 |
| **JAAD** | walking + not-looking | 62.9% | 143 |
| **PIE** | walking + **not-looking** | **74.1%** | 371 |
| **PIE** | walking + looking | 56.0% | 116 |

In JAAD: looking → high cross rate.
In PIE: not-looking → high cross rate.
**The sign reverses.**

## Why This Happens

Two complementary hypotheses:

**Geographic/cultural context:** JAAD footage is predominantly European residential streets. In this context, a pedestrian looking at the approaching driver is engaged in a social negotiation: "I see you, you see me, I'm crossing." Looking is an act of commitment.

**PIE annotation protocol:** PIE's footage is Toronto urban intersections with heavy traffic. A pedestrian who has decided to cross is walking purposefully and doesn't need to look at the vehicle — they've already committed. A pedestrian still looking at the vehicle is actively checking whether it's safe — still deciding.

**Annotation moment difference:** JAAD's `decision_point` is annotator-marked (subjective). PIE's `critical_point` is defined by the human experiment clip boundary (more objective, crowd-sourced). Different moments capture different stages of the decision process.

## Majority-Look Across Full Track (JAAD)

Walking+looking and walking+not-looking cross rates at the decision_point diverge sharply (95.7% vs 62.9%). But aggregate gaze across the full track predicts nothing: majority-looking = 85.3% cross, majority-not-looking = 85.4% cross. Only the decision-moment snapshot matters.

## Implications for Research

1. **Do not pool gaze features across JAAD and PIE.** Any model trained on both datasets will learn a conflicted gaze signal.
2. **Feature importance scores will disagree.** Gaze importance will appear high in JAAD-specific ablations and negative/null in PIE-specific ones.
3. This reversal motivates [[directions/cross-dataset-generalization|cross-dataset generalization research]] — models must learn context-dependent feature semantics.
4. **Action (walking vs. standing) is the consistent primary predictor** across both datasets and is safe to use as a shared feature.

## Related

- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] | [[findings/pie-gaze-reversal|PIE Gaze Reversal]]
- [[concepts/gaze-and-attention|Gaze and Attention]] | [[concepts/domain-shift|Domain Shift]]
- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
