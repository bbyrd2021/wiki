---
type: project
title: "eff_sign_detection — Traffic Sign Classifier"
aliases: ["eff_sign_detection", "traffic sign classifier"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "eff_sign_detection/README.md"
tags: [project, classification, traffic-sign, efficientnet, mapillary]
status: complete
---

# eff_sign_detection

15-class traffic sign classifier using EfficientNet, trained on the Mapillary Traffic Sign Dataset (MTSD) v2.

**Repo:** `/data/repos/eff_sign_detection/`

## Classes (15 categories)

Speed limit, Pedestrian crossing, Yield, Stop, No entry, One way, No parking, School zone, Construction, Do not pass, Keep right/left, Roundabout, Traffic light ahead, Railway crossing, Other regulatory

## Role in Pipeline

Downstream stage in [[projects/auto-drive-perception|AutoDrivePerception2026]]: YOLO detects sign ROIs → EfficientNet classifies the sign category. See `SETUP.md` for dataset and training setup.

## Related

- [[tools/traffic-sign-classification|Traffic Sign Classification]]
- [[projects/eff-light-detection|eff_light_detection]] (sister classifier)
