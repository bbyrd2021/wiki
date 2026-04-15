---
type: paper
title: "ENCORE: Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction"
authors: "Rasouli"
year: 2024
venue: "ICRA"
arxiv: "2310.10424"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, trajectory, scale-aware, evaluation]
status: complete
datasets_used: [pie]
---

# ENCORE

**Rasouli | ICRA 2024 | arXiv:2310.10424**

Proposes stratified evaluation by pedestrian scale (bbox height) and ego-vehicle speed, exposing evaluation blind spots in aggregate MSE metrics.

## Contribution

- **Stratified benchmarking paradigm**: reveals that aggregate MSE hides failure modes at small scales and high ego-speed
- Scale-aware encoder processes bbox sequences conditioned on pedestrian size
- Motion-conditioned Transformer decoder for 45-frame (1.5s) future trajectory generation
- Evaluated on PIE test set

## Key Finding

Intent/trajectory models fail systematically on distant pedestrians (small bbox) and at high ego-vehicle speeds. Aggregate accuracy metrics mislead — scenario-stratified evaluation is essential.

## Related

- [[methods/encore|ENCORE Method]] | [[datasets/pie|PIE]]
