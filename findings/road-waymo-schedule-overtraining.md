---
type: finding
title: "ROAD-Waymo Over-Trains Long Detector Schedules: Effective n = Scenes, Not Frames"
aliases: []
created: 2026-08-24
updated: 2026-08-24
sources:
  - "ROAD_Reason/experiments/exp11_yolo/runs/v8x_agent_1280_eccv/results.csv"
tags: [finding, road-plusplus, yolo, training-schedule, video-redundancy, exp11]
status: complete
---

# ROAD-Waymo Over-Trains Long Detector Schedules

**Effective dataset size is the scene count, not the frame count.** 115,602
"training images" are 10 Hz samples of **600 twenty-second videos**; the
verified 3.3M boxes collapse to 41,935 agent tubes (~79 boxes per physical
object). A 50-epoch schedule shows each object ~4,000 times.

## The measurement (exp11 YOLOv8x, ECCV Track-1 recipe)

- Epoch-1 checkpoint (COCO warm start + one epoch): **35.24 agent f-mAP**
- Epoch-50 checkpoint (full recipe, aug-off final 5): **29.92** — a **5.3-point
  loss**, landing *below* the ECCV report's 31.6 reference
- Ultralytics val mAP50 never re-reached its epoch-1 value in 49 subsequent
  epochs; the aug-off phase produced no jump

Val is 198 *different* videos, so the mechanism is scene memorization: the
model trades transferable COCO features for the specific appearance of 600
scenes, which pays nothing on unseen ones.

## Implications

1. On video-redundant driving data, early checkpoints with strong pretrained
   inits are competitive-to-superior; fitness-selected `best.pt` is essential.
2. Hypothesis (unverifiable from the report): the ECCV winners' 75/25 split of
   the training data, if frame-level rather than video-level, would *reward*
   scene memorization and explain why the long recipe looked good internally.
3. YOLO26x on the identical recipe shows a **rising** early curve (0.366 →
   0.375 by epoch 3, unlike v8x's decline) — architecture changes the
   trade-off; watch its best-epoch index. Cite the final pair when it lands.

## Related

- [[findings/exp11-yolo-hybrid]] — the run this was measured in
- [[papers/eccv24-track1]] — the recipe source
- [[datasets/road-plusplus]] — the verified tube/box counts behind the arithmetic
