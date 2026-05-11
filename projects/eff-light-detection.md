---
type: project
title: "eff_light_detection — Traffic Light Classifier"
aliases: ["eff_light_detection", "traffic light classifier"]
created: 2026-04-07
updated: 2026-05-07
sources:
  - "eff_light_detection/README.md"
  - "eff_light_detection/results/experiment_summary.md"
  - "eff_light_detection/tools/bstld_to_patches.py"
  - "eff_light_detection/tools/merge_patch_datasets.py"
tags: [project, classification, traffic-light, efficientnet, lisa, bstld]
status: complete
---

# eff_light_detection

EfficientNet-B0 traffic light classifier. Originally trained on LISA only
(7-class incl. `green_straight`); retrained 2026-05-07 on **LISA + BSTLD** to
fix red→yellow domain-shift failures observed on `sensor_data.bag` and to add
an explicit **`off`** class so the ROS HSV off-signal gate can be retired.

**Repo:** `/data/repos/eff_light_detection/`

## Current taxonomy (7 classes, retrain 2026-05-07)

`green_left, green_light, off, red_left, red_light, yellow_left, yellow_light`

Dropped: `green_straight` (only 205 samples in LISA, none in BSTLD).
Added: `off` (BSTLD-only source, 435 train patches).

## Training data — merged

| Source | Train | Val | Notes |
|---|---|---|---|
| LISA (dayTrain + nightTrain) | 49,720 | 23,596 | min bbox 10px, 1.2× expansion |
| BSTLD (90/10 image split) | 7,667 | 863 | min bbox 6px (median dim 8.5px) |
| **Merged** | **57,387** | **24,459** | hard-linked, alphabetical 7 dirs |

## Role in Pipeline

Downstream stage in [[projects/auto-drive-perception|AutoDrivePerception2026]]:
YOLOv10 detects traffic light ROIs → EfficientNet classifies state. See
`TRAFFIC_LIGHT_PIPELINE_SETUP.md` for ROS integration.

## Related

- [[findings/traffic-light-domain-shift|Domain-shift retrain (2026-05-07)]]
- [[tools/traffic-light-classification|Traffic Light Classification]]
- [[projects/eff-sign-detection|eff_sign_detection]] (sister classifier)
