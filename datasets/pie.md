---
type: dataset
title: "PIE — Pedestrian Intention Estimation"
aliases: ["PIE"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/PIE/PIE_Dataset_Summary.md"
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/PIE_analysis/cross_tab_results.json"
  - "PedestrianIntent++/PIE_analysis/pie_stats.json"
tags: [dataset, pie, crossing-intent, intention-probability, pedestrian]
status: complete
clips: 53
frames: 740901
primary_units: 1842
annotation_format: xml
---

# PIE — Pedestrian Intention Estimation

York University (2019). Pedestrian intent prediction dataset with crowd-sourced `intention_prob` scores. Toronto urban intersections. Wide-angle (fisheye) dashcam at 30 FPS.

## Key Numbers

| Metric | Value |
|--------|-------|
| Video sets | 6 (set01–set06) |
| Clips | 53 |
| Frames | 740,901 |
| Pedestrian tracks | 1,842 |
| Frame rate | 30 FPS |
| Geographic coverage | Toronto, Canada only |

## Splits (from `pie_data.py`)

| Split | Sets |
|-------|------|
| Train | set01, set02, set04 |
| Val | set05, set06 |
| Test | set03 |

## Three Annotation Layers (XML per set)

| Layer | Content |
|-------|---------|
| Main | Bounding boxes, action, look, cross, gesture, occlusion |
| Attributes | Demographics, intent, ego-vehicle OBD data (speed, GPS, accel, gyro, heading) |
| Sets | Clip metadata |

## Key Labels and Unique Features

- **`intention_prob`**: crowd-sourced scalar per pedestrian (not per frame) — see [[concepts/intention-probability|Intention Probability]]
- **`critical_point`**: last frame of human experiment clip (definition boundary for intent prediction)
- **`exp_start_point`**: start of human experiment observation window
- **`crossing_point`**: frame when pedestrian steps onto road (median 30 frames AFTER critical_point)
- **OBD sensor data**: ego-vehicle speed, GPS, accelerometer, gyroscope, heading — unique in JAAD/PIE/ROAD++

## Intention Probability

Derived from crowd-sourced experiment: up to 30 humans watched each clip from exp_start_point → critical_point (ending BEFORE actual crossing) and voted "will cross? Yes/No". `intention_prob` = fraction saying Yes.

**Distribution:** mean=0.712, median=0.850 — **strongly bimodal**
- 42.5% of peds at 0.9–1.0 (clear crossers)
- 7.2% at 0.0–0.1 (clear non-crossers)

See [[findings/pie-intention-bimodality|PIE Intention Bimodality]].

## Critical Gaze Reversal

At critical_point, **walking + not-looking → 74.1% cross** (higher than walking+looking at 56.0%). In PIE, looking toward the vehicle signals hesitation/caution, not intent. This is **opposite** to [[datasets/jaad|JAAD]].

See [[findings/pie-gaze-reversal|PIE Gaze Reversal]] and [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]].

## Cross-Tabulation (at critical_point)

| Action | Look | Cross Rate | N |
|--------|------|-----------|---|
| walking | not-looking | **74.1%** | 371 |
| walking | looking | 56.0% | 116 |
| standing | not-looking | 21.8% | 697 |
| standing | looking | 14.2% | 190 |

Source: `PedestrianIntent++/PIE_analysis/cross_tab_results.json`

## Visualizations

- Real frames: `PedestrianIntent++/PIE_analysis/viz_real/PIE_real_annotated_frames.png` (6 peds × 4 keyframes)
- Timeline: `PedestrianIntent++/PIE_analysis/viz_real/PIE_timeline_real_frames.png`

## API

```python
from pie_data import PIE
db = PIE(data_path='/data/datasets/PIE/')
```

## Related

- [[datasets/jaad|JAAD]] (compare gaze — opposite findings) | [[datasets/road-plusplus|ROAD++]]
- [[concepts/intention-probability|Intention Probability]] | [[concepts/crossing-intent|Crossing Intent]]
- [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]] | [[comparisons/dataset-comparison|Dataset Comparison]]
- [[projects/efficient-pie|EfficientPIE]] (trained primarily on PIE)
