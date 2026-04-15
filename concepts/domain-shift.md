---
type: concept
title: "Domain Shift in Pedestrian Intent Research"
aliases: ["domain shift", "cross-dataset generalization", "distribution shift"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/ROAD_plusplus/ROAD_plusplus_summary.md"
tags: [concept, domain-shift, generalization, jaad, pie, road-plusplus]
status: complete
---

# Domain Shift

The distributional differences between datasets that cause models trained on one to underperform on another. A major challenge in deploying pedestrian intent models across real-world conditions.

## Sources of Shift Across JAAD, PIE, ROAD++

| Factor | JAAD | PIE | ROAD++ |
|--------|------|-----|--------|
| Geography | Europe + N. America | Toronto only | USA (Waymo) |
| Camera | Standard dash cam | Wide-angle fisheye | Waymo front cam |
| Frame rate | ~30 FPS | 30 FPS | 10 FPS |
| Scene density | Residential, urban | Urban intersections | Urban US intersections |
| Protocol | Annotator decision_point | Crowd experiment critical_point | No intent protocol |
| Gaze meaning | Looking = commitment | Looking = hesitation | No gaze label |

## Empirical Evidence of Shift

**ROAD++ paper (Salmank et al., arXiv:2411.01683):** 3D-RetinaNet trained on ROAD (UK) drops substantially when tested on ROAD-Waymo (USA) without domain adaptation. US vs. UK road structure (right-hand vs. left-hand driving, different intersection geometry) causes significant label distribution shift.

**EfficientPIE cross-dataset (v3):** Trained on PIE, evaluated on JAAD:
- Accuracy: 0.926 (PIE) → 0.878 (JAAD) — expected drop
- AUC: 0.947 (PIE) → 0.885 (JAAD) — still above original EfficientPIE paper

**Gaze reversal:** The opposite meaning of the `look` label across JAAD and PIE is itself a form of domain shift — the feature distribution is similar but the label semantics reverse.

## Approaches to Addressing Shift

1. **Data augmentation**: simulate distribution of target domain during training
2. **Domain adaptation**: fine-tune on target domain data (few-shot)
3. **Domain-invariant features**: learn representations that are invariant to geography/protocol
4. **Multi-domain training**: train simultaneously on JAAD + PIE; forces shared representations

See [[directions/cross-dataset-generalization|Cross-Dataset Generalization]] for the proposed research direction.

## Related

- [[concepts/gaze-and-attention|Gaze and Attention]] (gaze reversal as domain shift)
- [[directions/cross-dataset-generalization|Cross-Dataset Generalization]]
- [[comparisons/dataset-comparison|Dataset Comparison]]
