---
type: project
title: "BEVFusion — Camera + LiDAR BEV Fusion"
aliases: ["BEVFusion", "bevfusion"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "bevfusion/README.md"
tags: [project, 3d-detection, bev, lidar, camera-fusion, foundation]
status: complete
---

# BEVFusion

Multi-sensor fusion framework for 3D object detection and BEV map segmentation, unifying camera and LiDAR features in bird's-eye view space. ICRA 2023. Former SOTA on Waymo, nuScenes, and Argoverse leaderboards (2022–2023).

**Repo:** `/data/repos/bevfusion/` | **Published:** ICRA 2023

## Core Contribution

Projects both camera image features and LiDAR point cloud features into a shared BEV space, then fuses them. Avoids the depth-estimation bottleneck of camera-only methods while retaining rich semantic features from images.

## Performance Highlights

- 1st on Waymo open dataset (at publication)
- 1st on nuScenes detection and segmentation (at publication)
- 1st on Argoverse 3D detection

## Role in Research Stack

Foundational reference for 3D multi-sensor perception. Integrated into [[projects/mmdetection3d|mmdetection3d]] as a config. Represents the upper bound for camera+LiDAR fusion approaches.

See [[concepts/bev-fusion|BEV Fusion Concept]] for architectural details.
