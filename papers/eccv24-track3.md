---
type: paper
title: "First Place: ECCV 2024 ROAD++ Challenge — Atomic Activity Recognition"
authors: "(anonymous)"
year: 2024
venue: "ECCV Workshop"
arxiv: "2410.23092"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, activity-recognition, action-classification]
status: complete
datasets_used: [road-plusplus]
---

# ECCV 2024 ROAD++ Track 3 Winner — Atomic Activity Recognition

**arXiv:2410.23092 | ECCV 2024 Workshop**

ECCV 2024 Track 3 winner (69% mAP on 64-class atomic activity recognition). Uses video Transformer with multi-label classification.

## Contribution

- Video Transformer fine-tuned from Kinetics + ActivityNet pretraining
- Multi-label classification head for 64 fine-grained action classes
- Distinguishes `Wait2X`, `XingFmLft`, `XingFmRht`, `Xing`, `MovAway` for pedestrian agents

## Key Finding

Pretraining on action recognition datasets (Kinetics, ActivityNet) before ROAD++ fine-tuning is essential. Track 3 evaluation is the most directly relevant to pedestrian crossing behavior recognition.

## Related

- [[datasets/road-plusplus|ROAD++]] | [[papers/eccv24-track1|ECCV 2024 Track 1 winner]]
