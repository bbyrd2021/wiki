---
type: direction
title: "LeWM Scene Prediction (ROAD_Reason Approach 6)"
aliases: ["LeWM", "Approach 6", "le-wm"]
created: 2026-04-07
updated: 2026-04-07
sources: ["ROAD_Reason/docs/APPROACHES.md"]
tags: [direction, road-plusplus, world-model, workstation-feasible]
status: complete
novelty: true
feasibility: workstation
datasets_required: [road-plusplus]
approach_number: 5
---

# LeWM Scene Prediction

**Approach 6** in [[projects/road-reason|ROAD_Reason]]. Exploratory — adaptation to driving video is novel.

## Key Advantage: Workstation-Feasible

- **~15M parameters**, trains in hours on a single RTX 3090/4090
- First JEPA to train end-to-end from raw pixels on a single GPU (arXiv:2603.19312)
- GitHub: `https://github.com/lucas-maes/le-wm`

## Method

1. Adapt LeWM to ROAD-Waymo clips: predict future latent scene states from current observations
2. Apply t-norm constraint loss across predicted future frames — future states must satisfy logic requirements
3. Use surprise detection (anomalous latent predictions) for unusual pedestrian behavior flagging

## Why Relevant

- World model predicting future scene states is a natural fit for intent prediction
- T-norm constraints on future predictions is unexplored
- Anomaly detection capability complements primary intent prediction

## What's Novel

Current LeWM benchmarks are robotics-only. Adaptation to driving video with constraint supervision on predictions is novel.

## Related

- [[projects/road-reason|ROAD_Reason]] | [[directions/jepa-intent-head|V-JEPA 2 (Approach 4)]]
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
