---
type: tool
title: "Traffic Light Classification"
aliases: ["traffic light classifier", "eff_light_detection pipeline"]
created: 2026-04-07
updated: 2026-04-07
sources: ["eff_light_detection/README.md", "eff_light_detection/TRAFFIC_LIGHT_PIPELINE_SETUP.md"]
tags: [tool, traffic-light, classification, efficientnet, lisa]
status: complete
---

# Traffic Light Classification

EfficientNet-B0 classifier for traffic light state, trained on the LISA Traffic Light Dataset (~109K images).

## Classes (7)

| Class | Description |
|-------|-------------|
| Red circular | Standard red stop signal |
| Yellow circular | Standard yellow caution |
| Green circular | Standard green go |
| Red arrow | Directional red |
| Yellow arrow | Directional yellow |
| Green arrow | Directional green |
| No signal | No visible traffic light |

## Pipeline Role

Downstream stage in [[tools/ros-perception-pipeline|ROS Perception Pipeline]]:
1. [[projects/yolo-bdd|YOLO]] or [[projects/yolopx|YOLOPX]] detects traffic light bounding boxes
2. ROIs are cropped and fed to this classifier
3. State + confidence output is published on ROS topic

## Setup

See `eff_light_detection/TRAFFIC_LIGHT_PIPELINE_SETUP.md` for ROS integration guide.
Dataset: `eff_light_detection/KAGGLE_SETUP.md` for LISA dataset download.

## Related

- [[projects/eff-light-detection|eff_light_detection]] | [[tools/ros-perception-pipeline|ROS Pipeline]]
- [[tools/traffic-sign-classification|Traffic Sign Classification]]
