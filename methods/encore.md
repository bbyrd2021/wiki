---
type: method
title: "ENCORE"
aliases: ["ENCORE"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [method, pie, trajectory, scale-aware, evaluation]
status: complete
datasets_evaluated: [pie]
---

# ENCORE

Scale- and motion-aware model for egocentric pedestrian trajectory prediction on PIE (ICRA 2024).

## Key Contribution

**Stratified evaluation paradigm**: proposes evaluating separately by pedestrian bbox height (scale proxy for distance) and ego-vehicle speed. Exposes failure modes hidden by aggregate MSE.

## Architecture

- Scale-aware encoder: processes bbox sequences conditioned on pedestrian bounding box size
- Motion-conditioned Transformer decoder: conditions on ego-vehicle speed
- Outputs 45-frame (1.5s) future trajectory

## Evaluation Insight

Aggregate metrics mislead — models fail systematically at small scales and high ego-speeds. This insight applies to intent prediction (not just trajectory) and should inform evaluation protocols for [[projects/efficient-pie|SparseTemporalPIE]].

## Related

- [[papers/rasouli-2024-encore|ENCORE Paper]] | [[datasets/pie|PIE]]
