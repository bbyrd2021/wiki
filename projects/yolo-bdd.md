---
type: project
title: "YOLO_BDD — YOLOv10 on BDD100K"
aliases: ["YOLO_BDD"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "YOLO_BDD/README.md"
  - "YOLO_BDD/evaluation_summary_20260128_152537.md"
tags: [project, detection, yolov10, bdd100k]
status: complete
---

# YOLO_BDD

YOLOv10 object detection baseline and fine-tuned models evaluated on the BDD100K dataset (10 classes).

**Repo:** `/data/repos/YOLO_BDD/`

## Detection Classes (10)

Pedestrian, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Traffic Light, Traffic Sign

## Key Points

- Baseline YOLOv10 + fine-tuned variant with per-class analysis
- BDD100K evaluation with precision/recall metrics per class
- Outputs feed downstream classifiers for traffic lights/signs
- See evaluation summary: `YOLO_BDD/evaluation_summary_20260128_152537.md`

## Related

- [[projects/yolopx|YOLOPX]] (end-to-end panoptic alternative)
- [[projects/auto-drive-perception|AutoDrivePerception2026]] (uses YOLO output)
- [[tools/bdd100k-detection|BDD100K Detection Comparison]]
