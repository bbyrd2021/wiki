---
type: tool
title: "BDD100K Detection — Multi-Model Comparison"
aliases: ["BDD100K detection", "BDD100K comparison"]
created: 2026-04-07
updated: 2026-04-07
sources: ["YOLOPX/README.md", "YOLO_BDD/README.md", "TwinLiteNet/README.md"]
tags: [tool, bdd100k, detection, segmentation, comparison]
status: complete
---

# BDD100K Multi-Model Comparison

Models trained on [[datasets/bdd100k|BDD100K]] for driving perception tasks.

## Detection Models

| Model | Task | Key Metric | Notes |
|-------|------|------------|-------|
| [[projects/yolopx\|YOLOPX]] | Detection + seg + lane | 93.7% recall, 83.3% mAP50 | Anchor-free panoptic, 32.9M params, 47fps |
| [[projects/yolo-bdd\|YOLO_BDD]] | Detection only | See evaluation summary | YOLOv10 fine-tuned |
| [[projects/twinlitenet\|TwinLiteNet]] | Seg + lane (no detection) | Competitive vs YOLOP | Lightweight, embedded-optimized |

## Segmentation + Lane Models

| Model | Drivable mIoU | Lane Acc |
|-------|---------------|---------|
| YOLOPX | 93.2% | 88.6% |
| TwinLiteNet | Competitive | Competitive |
| HybridNets (reference) | ~90% | ~85% |

## Trade-offs

| Approach | Best For |
|----------|---------|
| YOLOPX | Max accuracy, single model for all tasks |
| TwinLiteNet | Embedded deployment, seg+lane only |
| YOLO_BDD | Detection only, modular pipeline with downstream classifiers |

## Related

- [[projects/yolopx|YOLOPX]] | [[projects/yolo-bdd|YOLO_BDD]] | [[projects/twinlitenet|TwinLiteNet]]
- [[datasets/bdd100k|BDD100K Dataset]]
