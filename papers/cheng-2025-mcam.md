---
type: paper
title: "MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding"
aliases: ["mcam", "dsdag", "driving state directed acyclic graph"]
authors: "Cheng et al."
year: 2025
venue: "ICCV"
arxiv: "2507.06072"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/mcam-dsdag_iccv2025.pdf"
  - "ROAD_Reason/papers/Cheng_MCAM_Multimodal_Causal_Analysis_Model_for_Ego-Vehicle-Level_Driving_Video_Understanding_ICCV_2025_paper.pdf"
tags: [paper, causal-reasoning, video-understanding, autonomous-driving, dag, bdd-x, covla, approach-4]
status: complete
---

# MCAM — Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding

Cheng, Li, Xiong, Zhang, Wang, Liu. ICCV 2025. [arXiv:2507.06072](https://arxiv.org/abs/2507.06072) | [GitHub](https://github.com/SixCorePeach/MCAM)

## Key Idea

Introduces the **Driving State Directed Acyclic Graph (DSDAG)** to decompose driving behavior into discrete causal states, enabling fine-grained causal reasoning and dynamic inter-action modeling. Three components:

1. **Multi-level Feature Extractor (MFE)**: Dual-branch — 3DResNet for local features + VidSwin for global features. Captures both local details and long-range dependencies.
2. **Causal Analysis Module (CAM)**: Constructs causal relationships using DSDAG. Status transformation graph: Xs (start safe state) → W (unsafe/danger state) → Y (action) → Xe (end safe state), mediated by Z (environment). Uses feature disentanglement to establish direct, indirect, and confounding causal relationships.
3. **Vision-Language Transformer (VLT)**: Integrates visual features and causal relationships to generate coherent textual descriptions and reasoning.

## DSDAG Structure

```
Xs (initial state: safe/moving/stopped)
  → W (danger: traffic light change, obstacle, pedestrian)
    → Y (action: brake, stop, turn, accelerate)
      → Xe (end state: stopped, moving, changed lane)
Z (environment) mediates all transitions
```

The key insight: identical surface actions (e.g., "vehicle stops") can have different causal origins (red light vs. obstacle vs. traffic jam). DSDAG models this hidden danger state W that standard label-only approaches miss.

## Results

- **SOTA on BDD-X**: action description (CIDEr 2.349) and justification (CIDEr 1.744)
- **SOTA on CoVLA**: risk description BLEU-4 27.3, CIDEr 1.892
- Superior causal characteristic capture within video sequences
- Mitigates interference of spurious correlations

## Relevance to ROAD_Reason

**Core causal reasoning component for Approach 4 (MCDM)**. The DSDAG maps to ROAD_Reason's architecture:

| MCAM component | MCDM adaptation |
|---------------|----------------|
| MFE (3DResNet + VidSwin) | Replaced by OpenMixer on CLIP-ViP features |
| CAM (DSDAG) | Applied downstream on detected tubes |
| VLT | Generates "why" reasoning for each tube's action |

DSDAG presupposes that objects/scenes have already been perceived — the causal module operates on extracted features, not raw pixels. This validates the decoupled design: OpenMixer detects agents → DSDAG reasons about causal transitions → VLT generates explanations.

Pre-training plan: Stage 1 trains DSDAG + VLT on BDD-X + CoVLA (both have action + justification annotations), then Stage 3 applies to ROAD-Waymo tubes.

## Related

- [[papers/bao-2025-openmixer|OpenMixer]] — detection backbone feeding into DSDAG
- [[methods/multimodal-causal-driving|MCDM Architecture]] — full Approach 4 design
- [[papers/pearl-2009-causality|Pearl 2009 — Causality]] — theoretical foundation for DAG modeling
- [[datasets/bdd-x|BDD-X]] — training data for DSDAG pre-training
- [[datasets/covla|CoVLA]] — additional training data for causal reasoning
- [[comparisons/bdd-x-vs-covla|BDD-X vs CoVLA]] — Stage 1 pre-training source comparison
