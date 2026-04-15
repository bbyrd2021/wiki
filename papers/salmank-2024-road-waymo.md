---
type: paper
title: "ROAD-Waymo: Action Awareness at Scale for Autonomous Driving"
authors: "Salmank et al."
year: 2024
venue: "arXiv"
arxiv: "2411.01683"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, road-plusplus, domain-shift, tube-detection]
status: complete
datasets_used: [road-plusplus]
---

# ROAD-Waymo: Action Awareness at Scale for Autonomous Driving

**Salmank et al. | arXiv:2411.01683 | Nov 2024**

Formal introduction of the ROAD++ / ROAD-Waymo dataset. Benchmarks domain shift between ROAD (UK) and ROAD-Waymo (USA) and proposes ROAD++ as a multi-domain benchmark.

## Contribution

- Introduces ROAD++ (ROAD-Waymo): 10 agent classes, 22 action classes, 16 location classes
- Demonstrates domain shift: 3D-RetinaNet trained on ROAD (UK) drops substantially on ROAD-Waymo without adaptation
- Proposes neuro-symbolic baseline (t-norm constraint loss) — replication of ROAD-R methodology

## ⚠ Paper vs. Verified Numbers

| Metric | Paper Claims | **Verified from JSON** |
|--------|-------------|----------------------|
| Videos | 1000 | **798 + 202 unannotated test** |
| Frames | 198K | **153,534** |
| Agent tubes | 54K | **41,935** |
| Bounding boxes | 3.9M | **3.3M** |

## Domain Shift Finding

US vs. UK road structure (right-hand vs. left-hand driving, US intersection geometry) causes `XingFmLft`/`XingFmRht` label distributions to shift. Naive transfer fails — domain adaptation is required.

## Related

- [[datasets/road-plusplus|ROAD++]] | [[concepts/domain-shift|Domain Shift]]
- [[papers/singh-2022-road|ROAD foundational paper]]
