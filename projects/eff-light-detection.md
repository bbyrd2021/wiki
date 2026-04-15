---
type: project
title: "eff_light_detection — Traffic Light Classifier"
aliases: ["eff_light_detection", "traffic light classifier"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "eff_light_detection/README.md"
  - "eff_light_detection/results/experiment_summary.md"
tags: [project, classification, traffic-light, efficientnet]
status: complete
---

# eff_light_detection

7-class traffic light classifier using EfficientNet-B0, trained on the LISA Traffic Light Dataset (~109K images).

**Repo:** `/data/repos/eff_light_detection/`

## Classes (7)

Red circular, Yellow circular, Green circular, Red arrow, Yellow arrow, Green arrow, No signal

## Role in Pipeline

Downstream stage in [[projects/auto-drive-perception|AutoDrivePerception2026]]: YOLO detects traffic light ROIs → EfficientNet classifies the specific state. See `TRAFFIC_LIGHT_PIPELINE_SETUP.md` for ROS integration.

## Related

- [[tools/traffic-light-classification|Traffic Light Classification]]
- [[projects/eff-sign-detection|eff_sign_detection]] (sister classifier)
