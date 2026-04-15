---
type: finding
title: "ROAD++ Pedestrian Tube Statistics"
aliases: ["ROAD++ tubes", "pedestrian tubes", "tube statistics"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/ROAD_plusplus/analysis/stats_full.json"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [finding, road-plusplus, tubes, scale-variation, statistics]
status: complete
---

# ROAD++ Pedestrian Tube Statistics

Verified from `road_waymo_trainval_v1.0.json` (798 annotated videos).

## Summary

| Metric | Value |
|--------|-------|
| Total pedestrian tubes | **9,573** |
| Median tube length | **58 frames** (~5.8 seconds at 10 FPS) |
| Total pedestrian boxes | 712,640 |
| Videos with ≥1 ped tube | majority of 798 annotated videos |

## Tube Length Distribution

Right-skewed:
- Many short tubes (1–10 frames) — transient pedestrian appearances, edge of frame
- Many medium tubes (~30–80 frames) — pedestrians tracked across a scene
- Few long tubes (>150 frames) — pedestrians visible across most of a clip

The median of 58 frames = the typical temporal context available before a pedestrian leaves the field of view.

## Scale Variation

Pedestrians range from:
- **Tiny** (10×20 pixels): distant pedestrians at far end of US urban intersection
- **Large** (hundreds of pixels): nearby pedestrians close to the Waymo front camera

This extreme scale variation is the primary detection challenge — naively scaled architectures fail on tiny distant pedestrians (see ECCV 2024 Track 1 finding).

## Short-Tube Challenge

Short tubes (1–10 frames) are common — a model must handle pedestrians with very few historical frames and no temporal context. This connects to [[concepts/occlusion|occlusion]] challenges: tubes end not because the pedestrian is gone, but because they became occluded or exited the scene.

## Crossing-Behavior Labels on Tubes

| Action | Instances on Ped tubes |
|--------|----------------------|
| Wait2X | 54,928 |
| XingFmLft | 72,395 |
| XingFmRht | 75,591 |
| Xing | 41,659 |

~32–35% of Ped action label instances are crossing-related.

## Visualization

`PedestrianIntent++/ROAD_plusplus/viz/ROAD_tube_timeline.png` — swimlane diagram showing tube durations for two example videos (train_00240 and train_00355).

## Related

- [[datasets/road-plusplus|ROAD++]] | [[concepts/occlusion|Occlusion]]
- [[papers/eccv24-track1|ECCV 2024 Track 1 Winner]] (addresses scale variation)
