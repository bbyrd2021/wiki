---
type: paper
title: "PiShield: A PyTorch Package for Learning with Requirements"
aliases: ["pishield", "pi-shield", "shield layers"]
authors: "Stoian et al."
year: 2024
venue: "IJCAI"
arxiv: "2402.18285"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/pishield_ijcai2024.pdf"
tags: [paper, neuro-symbolic, constraints, guaranteed-compliance, shield-layers, road-r, autonomous-driving]
status: complete
---

# PiShield — A PyTorch Package for Learning with Requirements

Stoian, Tatomir, Lukasiewicz, Giunchiglia. IJCAI 2024. [arXiv:2402.18285](https://arxiv.org/abs/2402.18285) | [GitHub](https://github.com/mihaela-stoian/PiShield)

## Key Idea

**Shield Layers** that architecturally **guarantee** constraint satisfaction, regardless of input. Unlike t-norm losses (which penalize violations during training but cannot guarantee compliance), PiShield injects requirements directly into the network topology as a final layer.

Requirements are expressed as propositional logic in CNF (conjunctive normal form) or as linear inequalities. The Shield Layer takes a tensor of (possibly violating) predictions and returns a corrected tensor that is **guaranteed compliant**.

Usage is simple:
```python
from pishield.shield_layer import build_shield_layer
shield = build_shield_layer(num_dim, requirements_path)
corrected = shield(predictions)  # guaranteed compliant
```

Can be applied at inference time (post-hoc correction) or integrated into training (differentiable, gradients flow through the shield layer).

## Results

- Demonstrated on three domains: functional genomics, **autonomous driving (ROAD-R)**, tabular data generation
- On ROAD-R: integrating Shield Layers into DNNs yields increased performance across all metrics — not just compliance but actual mAP improvement
- Autonomous driving f-mAP: 0.225 → **0.241** (unconstrained → PiShield)

## Relevance to ROAD_Reason

**Drop-in replacement for t-norm constraint loss** in Exp2b and future experiments. Key advantages:

| Aspect | T-norm loss (current) | PiShield |
|--------|----------------------|----------|
| Compliance | Soft penalty — violations still occur | **Hard guarantee — zero violations** |
| Training | Added loss term, hyperparameter λ | Shield layer in forward pass, differentiable |
| Inference | No guarantee | Guaranteed compliant |
| Memory | High for complex constraints (see [[papers/stoian-2023-efficient-tnorms|efficient t-norms]]) | Efficient — operates on prediction tensor |

For ROAD++ duplex constraints (49 valid / 220 possible) and triplet constraints (86 valid / 3,520 possible), PiShield could eliminate the 13-38% duplex violation rates observed across all models. Especially valuable in the decoupled Approach 4 pipeline: detect with OpenMixer, classify with heads, shield the outputs.

Requirements file format for ROAD++ duplexes would encode mutual exclusivity constraints (e.g., "LarVeh cannot perform Xing") as CNF clauses.

## Related

- [[papers/marconato-2022-road-r|ROAD-R]] — defines the 243 logic requirements PiShield can enforce
- [[papers/stoian-2023-efficient-tnorms|Efficient T-norms]] — memory-efficient soft alternative
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — t-norm methodology PiShield replaces
- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — shows 33% duplex violation rates that PiShield could eliminate
