---
type: method
title: "3D-RetinaNet"
aliases: ["3D-RetinaNet", "3DRetinaNet"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md", "ROAD_Reason/docs/APPROACHES.md"]
tags: [method, road-plusplus, tube-detection, anchor-based, kinetics]
status: complete
datasets_evaluated: [road-plusplus]
---

# 3D-RetinaNet

Anchor-based spatiotemporal tube detector, Kinetics-pretrained. The baseline model for all ROAD and ROAD++ evaluations.

## Architecture

- 3D convolutional backbone pretrained on Kinetics action recognition
- Anchor-based detection with temporal stride
- Multi-label classification heads: agent, action, location, duplex, triplet
- Evaluated on frame-mAP and video-mAP per label type

## Role in Research Stack

- **ROAD paper** (Singh et al. 2022): introduced as the ROAD baseline
- **ROAD++ paper** (Salmank et al. 2024): same architecture, benchmarked on ROAD-Waymo
- **ROAD_Reason Approach 1**: replication baseline (starting point before novel contributions)
- **ROAD_Reason Approach 2**: extended with t-norm constraint loss (neuro-symbolic baseline)

## Performance

- ECCV 2024 Track 1 winner: 30.82% video-mAP (shows significant room for improvement)
- Primary challenge: small objects (distant pedestrians), class imbalance (Car dominates)

## Baseline Repo

`https://github.com/salmank255/ROAD_plus_plus_Baseline`

## Related

- [[papers/singh-2022-road|ROAD Paper]] | [[papers/salmank-2024-road-waymo|ROAD-Waymo Paper]]
- [[projects/road-reason|ROAD_Reason]] | [[datasets/road-plusplus|ROAD++]]
