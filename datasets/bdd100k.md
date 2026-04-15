---
type: dataset
title: "BDD100K — Berkeley DeepDrive"
aliases: ["BDD100K", "BDD-100K"]
created: 2026-04-07
updated: 2026-04-07
sources: []
tags: [dataset, bdd100k, detection, segmentation, lane]
status: complete
---

# BDD100K

Large-scale diverse driving dataset from UC Berkeley. 100K dashcam video clips with diverse weather, lighting, and scene conditions across US cities.

## Annotation Tasks

- Object detection: 10 classes (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign)
- Drivable area segmentation
- Lane detection
- Semantic segmentation (full pixel labels)
- Instance segmentation
- Multiple object tracking (MOT)

## Role in Research Stack

Primary benchmark for [[projects/yolopx|YOLOPX]], [[projects/yolo-bdd|YOLO_BDD]], and [[projects/twinlitenet|TwinLiteNet]]. All driving perception models are evaluated here.

See [[tools/bdd100k-detection|BDD100K Detection Comparison]] for cross-model results.
