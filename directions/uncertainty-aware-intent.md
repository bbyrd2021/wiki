---
type: direction
title: "Uncertainty-Aware Intent Prediction"
aliases: ["uncertainty-aware intent", "intention_prob regression", "calibrated intent"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md", "PedestrianIntent++/ROAD_plusplus/APPROACHES.md"]
tags: [direction, pie, uncertainty, regression, calibration]
status: complete
novelty: true
feasibility: workstation
datasets_required: [pie]
---

# Uncertainty-Aware Intent Prediction

**Direction 1** from SYNTHESIS.md Part 7.

## Motivation

Most papers binarize PIE's `intention_prob` at 0.5 into a crossing/non-crossing label. This discards calibrated uncertainty. A pedestrian with `intention_prob=0.55` is very different from one at `0.95`:
- 0.55 → vehicle should slow and wait for more evidence
- 0.95 → vehicle should brake immediately

## Approach

Formulate intent estimation as **regression** (predict the probability value directly) or **probabilistic** (predict a distribution). Train against `intention_prob` as a soft target rather than binarizing.

**Loss options:**
- MSE against `intention_prob` value
- KL divergence against the underlying vote distribution (if recoverable)
- Beta distribution fitting (natural for values in [0,1])

## What's Missing

No published paper has trained a model to output calibrated uncertainty matching human annotator disagreement. All surveyed papers (2022–2025) binarize the label.

## Expected Benefit

Better-calibrated risk scores for downstream AV planners. [[methods/sparse-temporal-pie|SparseTemporalPIE v3]] already improves AUC by +0.030 — AUC measures calibration. The next step is explicit uncertainty quantification.

## Datasets

[[datasets/pie|PIE]] only — the only dataset with crowd-sourced uncertainty information.

## Related

- [[concepts/intention-probability|Intention Probability]] | [[findings/pie-intention-bimodality|PIE Intention Bimodality]]
- [[datasets/pie|PIE]] | [[projects/efficient-pie|EfficientPIE]]
