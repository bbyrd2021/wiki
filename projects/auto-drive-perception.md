---
type: project
title: "AutoDrivePerception2026 — ROS Perception Pipeline"
aliases: ["AutoDrivePerception2026", "ROS pipeline"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "AutoDrivePerception2026/PLAN.md"
tags: [project, detection, ros, yolov10, pipeline]
status: complete
---

# AutoDrivePerception2026

ROS 1 (Noetic) inference pipeline for autonomous vehicle perception. Integrates YOLOv10 object detection with downstream traffic light and sign classifiers into a multi-node ROS architecture.

**Repo:** `/data/repos/AutoDrivePerception2026/`

## Architecture

```
Camera topic → YOLOv10 ROS node (detection)
                      │
          ┌───────────┴───────────┐
          │                       │
Traffic light ROIs          Traffic sign ROIs
          │                       │
eff_light_detection         eff_sign_detection
(7-class EfficientNet)      (15-class EfficientNet)
          │                       │
          └───────────┬───────────┘
                ROS output topics
```

## Components

- **YOLOv10 ROS node** — `yolov10_ros/` — real-time detection publishing bounding boxes + classes
- **YOLOv5 ROS node** — `yolov5_ros/` — legacy fallback detector
- **Custom ROS message definitions** — typed bounding box messages with class and confidence fields
- Downstream classifiers: see [[projects/eff-light-detection|eff_light_detection]] and [[projects/eff-sign-detection|eff_sign_detection]]

## Related

- [[projects/yolo-bdd|YOLO_BDD]] | [[tools/ros-perception-pipeline|Pipeline Diagram]]
- [[tools/traffic-light-classification|Traffic Light Classification]]
- [[tools/traffic-sign-classification|Traffic Sign Classification]]
