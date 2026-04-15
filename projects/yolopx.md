---
type: project
title: "YOLOPX — Panoptic Driving Perception"
aliases: ["YOLOPX"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "YOLOPX/README.md"
  - "YOLOPX/eval_readme.md"
tags: [project, detection, segmentation, lane, bdd100k, panoptic]
status: complete
---

# YOLOPX

Anchor-free multi-task learning network for panoptic driving perception. Single end-to-end model handling object detection, drivable area segmentation, and lane detection simultaneously. BDD100K SOTA (2024).

**Repo:** `/data/repos/YOLOPX/` | **Published:** Pattern Recognition 2024

## Performance (BDD100K)

| Task | Metric | Score |
|------|--------|-------|
| Object detection | Recall / mAP50 | 93.7% / 83.3% |
| Drivable area seg | mIoU | 93.2% |
| Lane detection | Accuracy | 88.6% |
| Inference | fps (RTX) | 47 fps |
| Model size | params | 32.9M |

## Key Characteristics

- Anchor-free detection head (no anchor box tuning needed)
- Shared backbone + three task-specific heads
- End-to-end training on BDD100K
- Sign classification head extension (see `sign_head_fixes.md`)

## Related

- [[tools/bdd100k-detection|BDD100K Detection Comparison]] | [[projects/yolo-bdd|YOLO_BDD]]
- [[projects/twinlitenet|TwinLiteNet]] (alternative for segmentation+lane only)
- [[projects/auto-drive-perception|AutoDrivePerception2026]] (integrates detection output)
