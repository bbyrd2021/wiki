---
type: method
title: "3D-RetinaNet"
aliases: ["3D-RetinaNet", "3DRetinaNet"]
created: 2026-04-07
updated: 2026-05-11
sources: ["PedestrianIntent++/SYNTHESIS.md", "ROAD_Reason/docs/APPROACHES.md"]
tags: [method, road-plusplus, tube-detection, anchor-based, kinetics]
status: complete
datasets_evaluated: [road-plusplus]
---

# 3D-RetinaNet

Anchor-based spatiotemporal tube detector, Kinetics-pretrained. The baseline model for all ROAD and ROAD++ evaluations.

## Architecture

- **Backbone:** ResNet-50 I3D (~46M params) — 3D convolutional, pretrained on Kinetics action recognition
- Anchor-based detection with temporal stride and FPN
- Multi-label classification heads: agent, action, location, duplex, triplet
- **Total trainable params:** ~50-55M (all trainable, no frozen components)
- Evaluated on frame-mAP (f-mAP) and video-mAP per label type
- Training validation computes full f-mAP at IoU=0.5 every `VAL_STEP` epochs via `modules.evaluation.evaluate()`

## Role in Research Stack

- **ROAD paper** (Singh et al. 2022): introduced as the ROAD baseline
- **ROAD++ paper** (Salmank et al. 2024): same architecture, benchmarked on ROAD-Waymo
- **ROAD_Reason Approach 1**: replication baseline (starting point before novel contributions)
- **ROAD_Reason Approach 2**: extended with t-norm constraint loss (neuro-symbolic baseline)

## Performance

- **ROAD-Waymo f-mAP:** agent=17.76% (our replication baseline target)
- ECCV 2024 Track 1 winner: 30.82% video-mAP (shows significant room for improvement)
- Primary challenge: small objects (distant pedestrians), class imbalance (Car dominates)

## Size Comparison

The ResNet-50 I3D backbone is ~9x larger than EfficientNet-B0 (~46M vs ~5.3M). It handles both spatial precision and semantic understanding in a single large 3D CNN. Exp2c's dual-backbone approach (small CNN + frozen CLIP) splits this work — testing whether frozen VLM semantics can compensate for a lighter spatial backbone.

## Baseline Repo

`https://github.com/salmank255/ROAD_plus_plus_Baseline`

## Related

- [[papers/singh-2022-road|ROAD Paper]] | [[papers/salmank-2024-road-waymo|ROAD-Waymo Paper]]
- [[projects/road-reason|ROAD_Reason]] | [[datasets/road-plusplus|ROAD++]]
