---
type: tool
title: "Traffic Sign Classification"
aliases: ["traffic sign classifier", "eff_sign_detection pipeline"]
created: 2026-04-07
updated: 2026-04-07
sources: ["eff_sign_detection/README.md", "eff_sign_detection/SETUP.md"]
tags: [tool, traffic-sign, classification, efficientnet, mapillary]
status: complete
---

# Traffic Sign Classification

EfficientNet classifier for traffic sign categories, trained on Mapillary Traffic Sign Dataset (MTSD) v2.

## Categories (15)

Speed limit, Pedestrian crossing, Yield, Stop, No entry, One way, No parking, School zone, Construction, Do not pass, Keep right/left, Roundabout, Traffic light ahead, Railway crossing, Other regulatory

See `eff_sign_detection/results/CLASS_MAPPING_SUMMARY.md` for full class definitions.

## Pipeline Role

Downstream stage in [[tools/ros-perception-pipeline|ROS Perception Pipeline]]:
1. [[projects/yolopx|YOLOPX]] or [[projects/yolo-bdd|YOLO_BDD]] detects traffic sign bounding boxes
2. ROIs are cropped and fed to this classifier
3. Sign category output is published on ROS topic

## Setup

See `eff_sign_detection/SETUP.md` for dataset and training setup.

## Related

- [[projects/eff-sign-detection|eff_sign_detection]] | [[tools/ros-perception-pipeline|ROS Pipeline]]
- [[tools/traffic-light-classification|Traffic Light Classification]]
