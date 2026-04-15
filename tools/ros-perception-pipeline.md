---
type: tool
title: "ROS Perception Pipeline"
aliases: ["ROS pipeline", "AutoDrivePerception pipeline"]
created: 2026-04-07
updated: 2026-04-07
sources: ["AutoDrivePerception2026/PLAN.md"]
tags: [tool, ros, detection, classification, pipeline]
status: complete
---

# ROS Perception Pipeline

Multi-node ROS 1 (Noetic) pipeline integrating detection and classification for autonomous vehicle perception. Implemented in [[projects/auto-drive-perception|AutoDrivePerception2026]].

## Architecture

```
Camera topic
     │
YOLOv10 ROS node
(detection — 10 classes)
     │
     ├── Traffic Light ROIs ──► eff_light_detection (7-class EfficientNet)
     │                                   │
     └── Traffic Sign ROIs ───► eff_sign_detection (15-class EfficientNet)
                                         │
                            Combined ROS output topics
                        (bbox + class + state + confidence)
```

## Node Descriptions

| Node | Repo | Output |
|------|------|--------|
| YOLOv10 detector | `yolov10_ros/` | Bounding boxes + 10 classes |
| Light classifier | `eff_light_detection/` | 7-class traffic light state |
| Sign classifier | `eff_sign_detection/` | 15-class traffic sign category |

## ROS Messages

Custom typed bounding box messages with class label and confidence score. Published on separate topics per detector.

## Related

- [[projects/auto-drive-perception|AutoDrivePerception2026]] | [[projects/yolo-bdd|YOLO_BDD]]
- [[tools/traffic-light-classification|Traffic Light Classification]]
- [[tools/traffic-sign-classification|Traffic Sign Classification]]
