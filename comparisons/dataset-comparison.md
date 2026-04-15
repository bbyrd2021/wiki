---
type: comparison
title: "JAAD vs PIE vs ROAD++ — Dataset Comparison"
aliases: ["dataset comparison", "three dataset comparison"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [comparison, jaad, pie, road-plusplus, datasets]
status: complete
---

# JAAD vs PIE vs ROAD++ — Dataset Comparison

Side-by-side capabilities table from SYNTHESIS.md Part 3.

## Scale

| Metric | JAAD | PIE | ROAD++ |
|--------|------|-----|--------|
| Clips | 346 | 53 | 798 (+202 test) |
| Frames | 82,032 | 740,901 | 153,534 |
| Primary units | 686 behavioral peds | 1,842 ped tracks | 41,935 agent tubes |
| Bounding boxes | 391,038 | — | 3,304,353 |
| Frame rate | ~30 FPS | 30 FPS | 10 FPS |
| Geography | Europe + N. America | Toronto only | USA (Waymo) |

## Annotation Features

| Capability | ROAD++ | JAAD | PIE |
|------------|--------|------|-----|
| **Primary task** | Agent/action/event detection | Pedestrian crossing intent | Pedestrian intent + trajectory |
| **Per-frame behavioral labels** | agent type, action (22), location (16) | action, look, cross, gesture, nod | action, look, cross, gesture |
| **Intent / crossing label** | ✗ | Binary per-pedestrian | Binary + intention_prob (0–1) |
| **Decision moment** | ✗ | decision_point | critical_point (crowd experiment) |
| **Gaze annotation** | ✗ | ✓ look label | ✓ look label |
| **Compositional event labels** | ✓ 49 duplexes, 86 triplets | ✗ | ✗ |
| **Logic constraints** | ✓ 243 requirements (ROAD-R) | ✗ | ✗ |
| **Ego-vehicle sensor** | AV action (9 classes) | 5-class action label | OBD speed/GPS/accel/gyro |
| **Appearance annotations** | ✗ | ✓ 24 per-frame binary attrs | ✗ |
| **Multi-agent / scene** | ✓ All agents simultaneous | Bystanders labeled (no behavior) | Bystanders labeled (no behavior) |
| **Traffic light as agent** | ✓ Red/Amber/Green state | Traffic layer only | Scene attribute only |
| **License** | CC BY-NC-SA 4.0 (Waymo restriction) | MIT + CC-BY 4.0 | Custom research |

## Gaze Finding Summary

| Dataset | walking+looking | walking+not-looking | Interpretation |
|---------|----------------|--------------------|-|
| **JAAD** | **95.7%** cross | 62.9% cross | Looking = commitment |
| **PIE** | 56.0% cross | **74.1%** cross | Looking = hesitation |

The gaze effect **reverses** between datasets. See [[comparisons/jaad-vs-pie-gaze|detailed comparison]].

## What Each Dataset Is Best For

| Use Case | Best Dataset |
|----------|-------------|
| Crossing intent classification | [[datasets/pie|PIE]] (intention_prob + crowd validation) |
| Crossing behavior + appearance | [[datasets/jaad|JAAD]] (24 appearance attrs, 5 annotation layers) |
| Scene-level multi-agent reasoning | [[datasets/road-plusplus|ROAD++]] (compositional labels, constraints) |
| Pretraining scale (pedestrian detection) | ROAD++ (712K ped boxes vs. <400K in JAAD/PIE) |
| Calibrated uncertainty | PIE (only dataset with uncertainty information) |

## Related

- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]] | [[datasets/road-plusplus|ROAD++]]
- [[comparisons/jaad-vs-pie-gaze|Gaze Comparison]] | [[comparisons/model-comparison|Model Comparison]]
