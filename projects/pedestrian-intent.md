---
type: project
title: "PedestrianIntent++ — 3-Dataset Synthesis"
aliases: ["PedestrianIntent++", "pedestrian intent synthesis"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/MEMORY.md"
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/JAAD/JAAD_summary.md"
  - "PedestrianIntent++/PIE/PIE_Dataset_Summary.md"
  - "PedestrianIntent++/ROAD_plusplus/ROAD_plusplus_summary.md"
tags: [project, pedestrian-intent, jaad, pie, road-plusplus, synthesis]
status: complete
---

# PedestrianIntent++

Comprehensive research synthesis of three pedestrian intent / scene-understanding datasets. Goal: professor-level review covering annotation structure, label semantics, cross-dataset comparison, literature survey (2022–2025), and proposed research directions.

## Scope

| Dataset | Clips | Frames | Primary Units |
|---------|-------|--------|---------------|
| [[datasets/jaad\|JAAD]] | 346 | 82,032 | 686 behavioral peds |
| [[datasets/pie\|PIE]] | 53 | 740,901 | 1,842 ped tracks |
| [[datasets/road-plusplus\|ROAD++]] | 798+202 test | 153,534 | 41,935 agent tubes |

## What Was Done

- All three repos cloned with full datasets downloaded
- All annotation layers parsed (XML for JAAD/PIE, JSON for ROAD++)
- Cross-tabulations: action × look → crossing outcome at decision/critical points
- Real dashcam frame visualizations with annotation overlays
- Literature survey: 6–8 papers per dataset (2022–2025)
- `SYNTHESIS.md` (75KB): 7-part synthesis report
- `MEMORY.md`: verified statistics and project tracking

## Key Findings

- **[[findings/jaad-gaze-findings|JAAD gaze]]**: walking+looking → 95.7% cross at decision_point
- **[[findings/pie-gaze-reversal|PIE gaze reversal]]**: walking+not-looking → 74.1% cross — opposite to JAAD
- **[[findings/pie-intention-bimodality|PIE intention_prob]]**: bimodal, mean=0.712, median=0.850
- **[[findings/road-ped-tube-statistics|ROAD++ tubes]]**: 9,573 ped tubes, median 58 frames, high scale variation
- JAAD appearance attributes (24 binary per-frame) unused by all surveyed papers

## SYNTHESIS.md Structure

| Part | Content |
|------|---------|
| Part 1 | Real-frame annotation visualizations (JAAD, PIE, ROAD++) |
| Part 2 | Label clarifications Q1–Q7 (cross timing, intention_prob, gaze, gestures, ROAD++ labels) |
| Part 3 | [[comparisons/dataset-comparison\|Side-by-side capability table]] |
| Part 4 | Literature survey (6–8 papers per dataset) |
| Part 5 | [[comparisons/jaad-vs-pie-gaze\|Cross-tabulation deep dive]] |
| Part 6 | [[comparisons/model-comparison\|Input/output patterns across models]] |
| Part 7 | 5 proposed research directions |

## Research Directions

1. [[directions/uncertainty-aware-intent|Uncertainty-aware intent]] — regression on intention_prob
2. [[directions/appearance-conditioned-intent|Appearance-conditioned intent]] — JAAD's 24 unused attributes
3. [[directions/cross-dataset-generalization|Cross-dataset generalization]] — JAAD → PIE → ROAD++
4. [[directions/logic-constrained-intent|Logic-constrained intent]] — ROAD-R methodology on JAAD/PIE
5. [[directions/state-sequence-modeling|State-sequence modeling]] — Transformer over discrete states

## Related Projects

- [[projects/efficient-pie|EfficientPIE]] — implements intent prediction on PIE/JAAD
- [[projects/road-reason|ROAD_Reason]] — scene reasoning on ROAD++

## Data Locations

- JAAD videos: `/data/datasets/JAAD/` (~3.1 GB)
- PIE videos: `/data/datasets/PIE/` (~74 GB)
- ROAD++ videos + annotations: `/data/datasets/ROAD_plusplus/` (~18 GB)
- GCS bucket: `gs://waymo_open_dataset_road_plus_plus/`
