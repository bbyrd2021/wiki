---
type: direction
title: "Appearance-Conditioned Intent Prediction"
aliases: ["appearance-conditioned", "JAAD appearance attributes"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [direction, jaad, appearance, attributes, novel]
status: complete
novelty: true
feasibility: workstation
datasets_required: [jaad]
---

# Appearance-Conditioned Intent Prediction

**Direction 2** from SYNTHESIS.md Part 7.

## Motivation

JAAD's `annotations_appearance/` layer contains **24 binary per-frame appearance attributes** (pose, clothing type, accessories, carrying objects, etc.) that are **unused by every surveyed paper (2022–2025)**. The attributes have been used only for pedestrian re-identification, never for intent prediction.

## Why It Should Help

The cross-tabulation leaves substantial ambiguity for standing pedestrians (44.9% cross rate for standing+not-looking). Appearance attributes may resolve this:
- Carrying a stroller → likely heading toward crosswalk
- Carrying heavy luggage → less urgency
- Certain clothing or age groups may correlate with behavior patterns

## Approach

1. Add JAAD appearance attributes as additional input features to an existing architecture ([[methods/pedformer|PedFormer]] or [[methods/sparse-temporal-pie|SparseTemporalPIE]])
2. Ablate: which of the 24 attributes contribute to intent prediction?
3. Test whether appearance features reduce ambiguity specifically for standing pedestrians

## What's Missing

No paper uses JAAD's appearance layer for intent prediction. The 24 attributes are a completely untapped information source.

## Datasets

[[datasets/jaad|JAAD]] only — the only dataset with appearance annotations.

## Related

- [[datasets/jaad|JAAD]] | [[concepts/crossing-intent|Crossing Intent]]
- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] (ambiguity for standing pedestrians)
