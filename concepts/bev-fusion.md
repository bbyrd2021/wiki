---
type: concept
title: "Bird's-Eye View (BEV) Fusion"
aliases: ["BEV fusion", "bird's eye view", "camera-LiDAR fusion"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "bevfusion/README.md"
tags: [concept, bev, lidar, camera-fusion, 3d-detection]
status: complete
---

# Bird's-Eye View (BEV) Fusion

Projecting multi-modal sensor features (camera images + LiDAR point clouds) into a shared bird's-eye view representation for 3D object detection and map segmentation.

## Core Idea

Both camera and LiDAR data are projected into a unified BEV space. BEV is natural for autonomous driving — it preserves spatial relationships between objects on the road plane and removes the depth ambiguity of perspective image projections.

## Why Fusion?

| Modality | Strengths | Weaknesses |
|----------|-----------|------------|
| Camera | Rich semantics, color, texture, small object detection | Depth ambiguity, sensitive to lighting |
| LiDAR | Accurate depth, 360° coverage, lighting invariant | No texture, sparse at distance, expensive |

BEVFusion combines both: LiDAR provides accurate geometry, camera provides semantic detail.

## BEVFusion Architecture (ICRA 2023)

1. Camera features extracted via image backbone → projected to BEV using depth estimation
2. LiDAR point cloud voxelized → processed by sparse 3D CNN → BEV features
3. Camera BEV + LiDAR BEV concatenated → joint BEV backbone
4. Task heads: 3D object detection + BEV map segmentation

## Results (at publication)

- 1st on Waymo Open Dataset
- 1st on nuScenes detection and segmentation
- 1st on Argoverse 3D detection

## Relevance to Research Stack

- [[projects/bevfusion|BEVFusion repo]] is available locally for experiments
- Integrated into [[projects/mmdetection3d|mmdetection3d]] as a config
- Represents the current state-of-the-art backbone approach for multi-sensor 3D perception

## Related

- [[projects/bevfusion|BEVFusion]] | [[projects/mmdetection3d|mmdetection3d]]
- [[concepts/multi-task-perception|Multi-Task Perception]]
