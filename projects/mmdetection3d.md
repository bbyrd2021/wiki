---
type: project
title: "mmdetection3d — 3D Detection Toolbox"
aliases: ["mmdetection3d", "OpenMMLab 3D"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "mmdetection3d/README.md"
tags: [project, 3d-detection, openmmlab, foundation, toolbox]
status: complete
---

# mmdetection3d

OpenMMLab's general-purpose 3D object detection toolbox. Supports 40+ detection methods across 300+ pretrained models. Actively maintained with ~6-month release cycle.

**Repo:** `/data/repos/mmdetection3d/`

## Supported Methods (selection)

PointPillars, VoteNet, SECOND, Part-A2, DSVT, BEVFusion, CenterPoint, 3DSSD, ImVoxelNet, MonoDETR, PETR, BEVDet

## Supported Datasets

KITTI, nuScenes, Waymo, Lyft, ScanNet, SUNRGB-D, S3DIS, Semantic KITTI

## Role in Research Stack

Foundation library for any 3D perception experiments. Provides pretrained weights, config system, and evaluation scripts. [[projects/bevfusion|BEVFusion]] is integrated here as a config.

See `mmdetection3d/configs/` for method-specific configurations.
