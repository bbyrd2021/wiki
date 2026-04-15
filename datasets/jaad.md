---
type: dataset
title: "JAAD — Joint Attention in Autonomous Driving"
aliases: ["JAAD", "JAAD 2.0"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/JAAD/JAAD_summary.md"
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD_analysis/cross_tabulations.md"
tags: [dataset, jaad, crossing-intent, gaze, pedestrian]
status: complete
clips: 346
frames: 82032
primary_units: 686
annotation_format: xml
---

# JAAD — Joint Attention in Autonomous Driving

York University (2016/2019). Pedestrian-centric dashcam dataset for crossing intent prediction. European + North American urban and suburban scenes.

## Key Numbers

| Metric | Value |
|--------|-------|
| Videos | 346 |
| Frames | 82,032 |
| Behavioral pedestrians | 686 |
| Total bounding boxes | 391,038 |
| Frame rate | ~30 FPS |
| Geographic coverage | Europe (majority) + North America |

## Five Annotation Layers

| Directory | Content |
|-----------|---------|
| `annotations/` | Per-pedestrian bounding boxes + action, look, cross, gesture, nod, reaction |
| `annotations_attributes/` | Demographics (age, gender), crossing outcome, context |
| `annotations_appearance/` | 24 binary per-frame appearance attributes |
| `annotations_traffic/` | Scene-level traffic state (per frame) |
| `annotations_vehicle/` | Ego-vehicle action (5 classes, per frame) |

## Key Labels

- **`action`**: walking or standing
- **`look`**: looking toward ego-vehicle or not
- **`cross`**: 1 = crossing (starts at `crossing_point` frame)
- **`decision_point`**: annotator-marked last viable decision frame — use as prediction boundary
- **`crossing_point`**: frame when pedestrian steps off curb (median delta from cross=1: 0 frames)

## Critical Findings

- **[[findings/jaad-gaze-findings|Gaze at decision_point]]**: walking+looking → **95.7% cross**; standing+not-looking → 44.9%
- Gaze at decision_point is strongly predictive; majority-gaze across full track predicts almost nothing
- **24 appearance attributes entirely unused** in all surveyed papers (2022–2025) — research opportunity (see [[directions/appearance-conditioned-intent|Direction 2]])

## Cross-Tabulation (decision_point)

| Action | Look | Cross Rate | N |
|--------|------|-----------|---|
| walking | looking | **95.7%** | 184 |
| standing | looking | 84.8% | 79 |
| walking | not-looking | 62.9% | 143 |
| standing | not-looking | **44.9%** | 49 |

Source: `PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json`

## Visualizations

- Real frames: `PedestrianIntent++/JAAD_analysis/viz_real/JAAD_real_annotated_frames.png` (7 peds × 4 keyframes)
- Timeline: `PedestrianIntent++/JAAD_analysis/viz_real/JAAD_timeline_real_frames.png`
- Analysis charts: `PedestrianIntent++/JAAD_analysis/viz/`

## API

```python
from jaad_data import JAAD
db = JAAD(data_path='/data/datasets/JAAD/')
```

## Related

- [[datasets/pie|PIE]] (compare gaze findings — reversed) | [[datasets/road-plusplus|ROAD++]]
- [[concepts/crossing-intent|Crossing Intent]] | [[concepts/gaze-and-attention|Gaze and Attention]]
- [[comparisons/dataset-comparison|Dataset Comparison]] | [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]]
- [[projects/pedestrian-intent|PedestrianIntent++]] | [[projects/efficient-pie|EfficientPIE]]
