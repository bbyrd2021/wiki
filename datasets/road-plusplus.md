---
type: dataset
title: "ROAD++ / ROAD-Waymo"
aliases: ["ROAD++", "ROAD-Waymo", "Road-Waymo"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/ROAD_plusplus/ROAD_plusplus_summary.md"
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/ROAD_plusplus/analysis/stats_full.json"
  - "ROAD_Reason/docs/ROAD_plusplus_summary.md"
tags: [dataset, road-plusplus, compositional-labels, neuro-symbolic, tube-detection]
status: complete
clips: 798
frames: 153534
primary_units: 41935
annotation_format: json
---

# ROAD++ / ROAD-Waymo

Oxford Brookes / Waymo (2023–2024). Scene-level event detection dataset for autonomous driving. Rich compositional label system covering agents, actions, locations, and logic constraints. Largest of the three datasets, focused on multi-agent scene understanding rather than pedestrian-only intent.

**Warning:** Paper claims differ from verified JSON stats — use verified numbers below.

## Key Numbers (VERIFIED from JSON)

| Metric | Paper Claims | **Verified from JSON** |
|--------|-------------|----------------------|
| Annotated videos | 1000 | **798** |
| Test videos (unannotated) | — | **202** |
| Annotated frames | 198K | **153,534** |
| Agent tubes | 54K | **41,935** |
| Bounding boxes | 3.9M | **3.3M** |

Annotation file: `/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.0.json` (~1 GB)

## Splits

| Split | Videos |
|-------|--------|
| Train | 600 |
| Val | 198 |
| Test | 202 (no labels) |

## Five-Level Label Hierarchy

| Level | Classes | Multi-label? | Example |
|-------|---------|-------------|---------|
| Agent | 10 | No | `Ped` |
| Action | 22 | Yes | `Wait2X` |
| Location | 16 | Yes | `RhtPav` |
| Duplex (agent+action) | **49 task** | Derived | `Ped-Wait2X` |
| Triplet (agent+action+loc) | **86 task** | Derived | `Ped-Wait2X-RhtPav` |

### Agent Classes (10)
Ped, Car, Cyc, Mobike, SmalVeh, MedVeh, LarVeh, Bus, EmVeh, TL

### Pedestrian-Specific Action Classes
- `Wait2X` — waiting to cross (54,928 instances)
- `XingFmLft` — crossing from left (72,395 instances)
- `XingFmRht` — crossing from right (75,591 instances)
- `Xing` — crossing unspecified direction (41,659 instances)
- `PushObj` — pushing an object (5,322 instances)

### Agent Distribution (top classes)

| Agent | Boxes |
|-------|-------|
| Car | 2,197,049 |
| **Ped** | **712,640** |
| MedVeh | 238,522 |
| TL | 57,722 |
| LarVeh | 39,889 |

## Compositional Constraints

- **49 valid duplexes** (from 220 possible agent+action combinations)
- **86 valid triplets** (from 3,520 possible combinations)
- Stored in `duplex_childs` + `triplet_childs` in annotation JSON
- Basis for t-norm constraint loss in [[projects/road-reason|ROAD_Reason]]
- ROAD-R (Marconato et al. 2022) formalized 243 propositional-logic requirements

## Pedestrian Tube Statistics

- 9,573 pedestrian tubes total
- Median tube length: 58 frames (~5.8 seconds at 10 FPS)
- High scale variation: pedestrians range from tiny (10×20px distant) to large (nearby)

See [[findings/road-ped-tube-statistics|ROAD++ Pedestrian Tube Statistics]].

## What ROAD++ Has vs JAAD/PIE

| Feature | ROAD++ | JAAD | PIE |
|---------|--------|------|-----|
| Intent/crossing label | ✗ | ✓ | ✓ + intention_prob |
| Decision/critical point | ✗ | ✓ | ✓ |
| Gaze/look annotation | ✗ | ✓ | ✓ |
| Multi-agent simultaneous | ✓ | Partial | Partial |
| Compositional labels | ✓ | ✗ | ✗ |
| Logic constraints | ✓ | ✗ | ✗ |
| Traffic light as agent | ✓ | Layer only | Attribute |
| AV action labels | ✓ 9 classes | 5 classes | OBD sensor |

## Visualizations

- Real frames: `PedestrianIntent++/ROAD_plusplus/viz/ROAD_real_annotated_frames.png` (5 videos × 4 frames)
- Tube timelines: `PedestrianIntent++/ROAD_plusplus/viz/ROAD_tube_timeline.png`

## Data Location

Videos: `/data/datasets/ROAD_plusplus/videos/` | Test videos: `.../test_videos/`
GCS: `gs://waymo_open_dataset_road_plus_plus/`
gsutil: `/home/brandon/google-cloud-sdk/bin/gsutil`

## Related

- [[concepts/compositional-labels|Compositional Labels]] | [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[comparisons/dataset-comparison|Dataset Comparison]]
- [[projects/road-reason|ROAD_Reason]] | [[projects/pedestrian-intent|PedestrianIntent++]]
