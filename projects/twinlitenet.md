---
type: project
title: "TwinLiteNet — Lightweight Dual-Task Perception"
aliases: ["TwinLiteNet"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "TwinLiteNet/README.md"
tags: [project, segmentation, lane, bdd100k, lightweight, embedded]
status: complete
---

# TwinLiteNet

Lightweight CNN for simultaneous drivable area segmentation and lane detection. Designed for efficient inference on embedded autonomous driving hardware. BDD100K-trained.

**Repo:** `/data/repos/TwinLiteNet/` | **Published:** Pattern Recognition 2024

## Key Characteristics

- Dual-task head: drivable area segmentation + lane detection
- Optimized for embedded hardware (low parameter count)
- BDD100K benchmark: competitive vs. YOLOP and HybridNets
- Minimal training artifacts — focused on inference efficiency

## Role in Research Stack

Lightweight alternative to [[projects/yolopx|YOLOPX]] for deployments requiring lower compute. If only segmentation+lane is needed (no object detection), TwinLiteNet is the leaner option.

See [[tools/bdd100k-detection|BDD100K Task Comparison]] for cross-method results.
