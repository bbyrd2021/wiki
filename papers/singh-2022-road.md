---
type: paper
title: "ROAD: The ROad Event Awareness Dataset for Autonomous Driving"
authors: "Singh, Cuzzolin et al."
year: 2022
venue: "IEEE TPAMI"
arxiv: "2102.11585"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, 3d-retinanet, tube-detection, event-detection]
status: complete
datasets_used: [road-plusplus]
---

# ROAD: The ROad Event Awareness Dataset

**Singh, Cuzzolin et al. | IEEE TPAMI 2022 | arXiv:2102.11585**

Foundational paper introducing the ROAD dataset (UK), the five-level compositional annotation framework (agent × action × location → duplex → event triplet), and the 3D-RetinaNet baseline.

## Contribution

- Defines the ROAD annotation paradigm: each bounding box receives simultaneous agent, action, and location labels; higher-level duplex and triplet labels are compositional
- Benchmarks frame-mAP and video-mAP for agent, action, duplex, and event detection
- Kinetics-pretrained 3D-RetinaNet serves as the baseline for all subsequent ROAD/ROAD++ work

## Key Results

- Establishes 3D-RetinaNet as the ROAD baseline (specific mAP numbers from UK ROAD dataset)
- All subsequent ROAD++ challenge entries are compared against this baseline

## Significance

The ROAD annotation paradigm is directly inherited by [[datasets/road-plusplus|ROAD++]] (ROAD-Waymo). The five-level label hierarchy, duplex/triplet structure, and 3D-RetinaNet baseline are central to [[projects/road-reason|ROAD_Reason]] Approaches 1–2.

## Related

- [[methods/3d-retinanet|3D-RetinaNet]] | [[datasets/road-plusplus|ROAD++]]
- [[papers/marconato-2022-road-r|ROAD-R]] | [[papers/salmank-2024-road-waymo|ROAD-Waymo]]
