---
type: direction
title: "ROAD++ Scene Context Transfer"
aliases: ["scene context transfer", "Direction 6"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [direction, road-plusplus, jaad, pie, transfer-learning, scene-context]
status: complete
novelty: true
feasibility: partial
datasets_required: [road-plusplus, jaad, pie]
---

# ROAD++ Scene Context Transfer to JAAD/PIE

**Direction 6** from SYNTHESIS.md Part 7.

## The Complementarity Gap

| | JAAD/PIE | ROAD++ |
|-|---------|--------|
| Intent labels | ✓ (rich) | ✗ |
| Scene context | ✗ (minimal) | ✓ (rich: 86-class triplets) |

ROAD++ can answer: "Is there a car moving toward the pedestrian? Is the pedestrian at a junction? Is the traffic light red?" — all contextual questions that JAAD/PIE lack automatic answers to.

## Approach

1. Train a scene encoder on ROAD++ to predict agent+action+location triplets from video clips
2. Freeze the encoder
3. Use it as a fixed scene-context feature extractor for JAAD/PIE intent models
4. The encoder provides implicit traffic signal state, crossing zone type, and multi-agent context

## Technical Challenge

ROAD++ encoder must generalize from Waymo footage to JAAD/PIE dashcam footage — a domain gap requiring adaptation. This is a specific instance of [[directions/cross-dataset-generalization|Direction 3]] focused on scene context rather than pedestrian detection.

## What's Missing

No paper uses ROAD++ compositional labels as a pretraining objective for downstream pedestrian intent prediction. Transfer between ROAD++ and JAAD/PIE is entirely unexplored.

## Related

- [[directions/cross-dataset-generalization|Cross-Dataset Generalization]]
- [[concepts/compositional-labels|Compositional Labels]] | [[datasets/road-plusplus|ROAD++]]
- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
