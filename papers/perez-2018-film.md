---
type: paper
title: "FiLM: Visual Reasoning with a General Conditioning Layer"
aliases: ["film", "feature-wise linear modulation"]
authors: "Perez et al."
year: 2018
venue: "AAAI"
arxiv: "1709.07871"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/film_aaai2018.pdf"
tags: [paper, conditioning, feature-modulation, visual-reasoning, multimodal-fusion]
status: complete
---

# FiLM — Visual Reasoning with a General Conditioning Layer

Perez, Strub, de Vries, Dumoulin, Courville. AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)

## Key Idea

Feature-wise Linear Modulation: a simple, general-purpose conditioning mechanism.

```
FiLM(F | x) = gamma(x) * F + beta(x)
```

Where `gamma` and `beta` are per-channel scale and shift parameters predicted from conditioning information `x` (e.g., language, another modality). Each feature map channel gets its own independent affine transformation.

Key properties:
- **Per-channel**: 256 channels → 512 parameters (256 gamma + 256 beta) per layer
- **Agnostic to spatial location**: same transform everywhere (channel-only, not spatial)
- **Generalizes Conditional Normalization**: can be seen as learned conditional batch/instance norm
- **Computational cost**: does not scale with image resolution — just two parameters per channel

## Results

- SOTA on CLEVR visual reasoning (97.7% accuracy, halving error rate)
- Modulates features coherently — learns to count, compare, identify objects
- Generalizes to unseen combinations and zero-shot scenarios
- Widely adopted: image stylization, speech recognition, visual QA, detection

## Relevance to ROAD_Reason

**Foundation for VMCNet-style modulation** — replacing Exp2b's scalar gate with FiLM conditioning:

| Current | FiLM upgrade |
|---------|-------------|
| `fused = fpn + gate * vlm` | `fused = gamma(vlm) * fpn + beta(vlm)` |
| 1 param per level | 512 params per level (256 gamma + 256 beta) |
| Same weight everywhere | Per-channel adaptive |

The VLM features predict gamma/beta via a small MLP, allowing channel-selective modulation of EfficientNet's spatial features. Zero-initialize beta so the model starts as identity (pure FPN features), then learns to incorporate VLM semantics.

**Limitation**: FiLM is channel-only, no spatial selectivity. Combine with [[papers/woo-2018-cbam|CBAM]]'s spatial attention for both channel and spatial control.

## Related

- [[papers/gao-2025-vmcnet|VMCNet]] — applies FiLM-style modulation for CNN+ViT detection
- [[papers/woo-2018-cbam|CBAM]] — adds spatial attention that FiLM lacks
- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — target model
