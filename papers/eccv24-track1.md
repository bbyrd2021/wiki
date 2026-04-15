---
type: paper
title: "First Place: ECCV 2024 ROAD++ Challenge — Spatiotemporal Agent Detection"
authors: "(anonymous)"
year: 2024
venue: "ECCV Workshop"
arxiv: "2410.23077"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, tube-detection, small-object]
status: complete
datasets_used: [road-plusplus]
---

# ECCV 2024 ROAD++ Track 1 Winner — Spatiotemporal Agent Detection

**arXiv:2410.23077 | ECCV 2024 Workshop**

ECCV 2024 Track 1 winner (30.82% video-mAP). Addresses extreme-size objects, low-light scenarios, and class imbalance.

## Contribution

- Large-backbone detector with multi-scale feature pyramids + temporal aggregation
- Extensive augmentation for tail classes (Cyc, Mobike, EmVeh)
- Shows that naive backbone scaling is insufficient — explicit small-object handling required

## Key Finding

The 9,573 pedestrian tubes in ROAD++ are particularly challenging due to scale variation (tiny distant peds to nearby large peds). Small-object detection is as important as temporal modeling.

## Related

- [[datasets/road-plusplus|ROAD++]] | [[findings/road-ped-tube-statistics|ROAD++ Tube Statistics]]
- [[papers/eccv24-track3|ECCV 2024 Track 3 winner]]
