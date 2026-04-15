---
type: concept
title: "Occlusion in Pedestrian Intent Research"
aliases: ["occlusion", "partial occlusion"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD/JAAD_summary.md"
tags: [concept, occlusion, jaad, pie, robustness]
status: complete
---

# Occlusion

Partial or full obscuring of a pedestrian from the ego-vehicle camera. A practical challenge for intent prediction systems.

## Occlusion Labels

**JAAD:** Per-frame occlusion class for each pedestrian (unoccluded, partial, full, heavy)

**PIE:** Per-frame occlusion annotation (similar classes)

**ROAD++:** No dedicated occlusion annotation; tube gaps implicitly indicate occlusion events

## Why It Matters for Intent Models

1. Pedestrian cues (gaze, action, body pose) become unreliable under heavy occlusion
2. Most published models assume full visibility — real deployments cannot
3. The `decision_point` (JAAD) or `critical_point` (PIE) may coincide with occlusion events
4. Models must either skip occluded frames or handle missing modalities gracefully

## Research Frontier

No surveyed paper (2022–2025) specifically addressed occlusion-robust pedestrian intent prediction. Papers generally filter heavily occluded samples from training/evaluation.

**Open question:** Can a model trained on non-occluded frames generalize to partially-occluded deployment conditions? This connects to [[directions/uncertainty-aware-intent|uncertainty quantification]] — high occlusion should produce high prediction uncertainty.

## Related

- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
- [[directions/uncertainty-aware-intent|Uncertainty-Aware Intent]]
