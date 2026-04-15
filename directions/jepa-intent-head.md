---
type: direction
title: "V-JEPA 2 + Intent Head (ROAD_Reason Approach 5)"
aliases: ["V-JEPA 2", "JEPA intent head", "Approach 5"]
created: 2026-04-07
updated: 2026-04-07
sources: ["ROAD_Reason/docs/APPROACHES.md"]
tags: [direction, road-plusplus, jepa, world-model, intent-prediction]
status: complete
novelty: true
feasibility: workstation
datasets_required: [road-plusplus]
approach_number: 4
---

# V-JEPA 2 + Intent Head

**Approach 5** in [[projects/road-reason|ROAD_Reason]]. Novel application of world model backbone to pedestrian crossing intent.

## Method

- Use pretrained **V-JEPA 2** encoder (frozen or lightly fine-tuned) as spatiotemporal feature extractor
- Add lightweight intent prediction head (MLP or Transformer) on top of JEPA features
- Train on ROAD-Waymo action/location labels + binary crossing intent
- V-JEPA learns physically predictable scene dynamics (motion, causality) without pixel reconstruction

## Why Relevant

- V-JEPA 2 is SOTA on action **anticipation** (Epic-Kitchens-100) — closest published analog to intent prediction
- Drive-JEPA (arXiv:2601.22032) demonstrates V-JEPA → AV pipeline for trajectory planning
- Pretrained weights are public; no pretraining compute required

## Extension

Add t-norm constraint loss to intent head outputs — combines Approaches 2 + 4.

## Key Papers

| Paper | arXiv | Relevance |
|-------|-------|-----------|
| V-JEPA 2 | 2506.09985 | Primary backbone |
| VL-JEPA | 2512.10942 | Vision-language extension |
| Drive-JEPA | 2601.22032 | AV pipeline demo |

## Why Novel

No published work applies JEPA-style predictive learning to pedestrian crossing intent prediction.

## Related

- [[projects/road-reason|ROAD_Reason]] | [[directions/constrained-vlm-reasoning|Approach 3 (Primary)]]
- [[directions/lewm-scene-prediction|LeWM Approach 6]]
