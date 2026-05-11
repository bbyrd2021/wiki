---
type: tool
title: "Traffic Light Classification"
aliases: ["traffic light classifier", "eff_light_detection pipeline"]
created: 2026-04-07
updated: 2026-05-07
sources: ["eff_light_detection/README.md", "eff_light_detection/TRAFFIC_LIGHT_PIPELINE_SETUP.md"]
tags: [tool, traffic-light, classification, efficientnet, lisa, bstld]
status: complete
---

# Traffic Light Classification

EfficientNet-B0 classifier for traffic light state. Trained 2026-05-07 on a
**merged LISA + BSTLD** patch set to address domain-shift failures observed
on the workshop USB-camera rig and to add a real `off` class so the ROS HSV
heuristic gate can be retired. See [[findings/traffic-light-domain-shift]] for
the symptom, root-cause analysis, and retrain protocol.

## Classes (7, alphabetical = ImageFolder index order)

| Idx | Class | Source(s) | Description |
|---|---|---|---|
| 0 | green_left | LISA + BSTLD | Green left-arrow |
| 1 | green_light | LISA + BSTLD | Standard green circular |
| 2 | **off** | **BSTLD only** | **Dark / unlit housing — replaces HSV gate** |
| 3 | red_left | LISA + BSTLD | Red left-arrow |
| 4 | red_light | LISA + BSTLD | Standard red circular |
| 5 | yellow_left | LISA only | Yellow left-arrow (rare, 297 train) |
| 6 | yellow_light | LISA + BSTLD | Standard yellow circular |

Dropped from previous: `green_straight` (205 LISA samples, none in BSTLD).

## Pipeline Role

Downstream stage in [[tools/ros-perception-pipeline|ROS Perception Pipeline]]:
1. [[projects/yolo-bdd|YOLO]] or [[projects/yolopx|YOLOPX]] detects traffic light bounding boxes
2. ROIs are cropped and fed to this classifier
3. State + confidence output is published on ROS topic

## Setup

See `eff_light_detection/TRAFFIC_LIGHT_PIPELINE_SETUP.md` for ROS integration guide.
Dataset: `eff_light_detection/KAGGLE_SETUP.md` (LISA), `tools/bstld_to_patches.py`
(BSTLD), `tools/merge_patch_datasets.py` (merge).

## Related

- [[findings/traffic-light-domain-shift]] — domain-shift retrain (2026-05-07)
- [[projects/eff-light-detection|eff_light_detection]] | [[tools/ros-perception-pipeline|ROS Pipeline]]
- [[tools/traffic-sign-classification|Traffic Sign Classification]]
