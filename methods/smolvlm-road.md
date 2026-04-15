---
type: method
title: "SmolVLM Baseline (ROAD_Reason)"
aliases: ["SmolVLM", "SmolVLM baseline"]
created: 2026-04-07
updated: 2026-04-07
sources: ["ROAD_Reason/README.md", "ROAD_Reason/docs/APPROACHES.md"]
tags: [method, road-plusplus, vlm, baseline, generative]
status: complete
datasets_evaluated: [road-plusplus]
---

# SmolVLM Baseline (ROAD_Reason)

The initial generative VLM baseline used in [[projects/road-reason|ROAD_Reason]] before the full Approach 3 architecture.

## Role

Starting point for understanding how a generative VLM (without constraint training) performs on ROAD++ scene reasoning. Three variants:
1. **Zero-shot**: off-the-shelf SmolVLM, no fine-tuning
2. **Constraint-aware prompting**: system prompt embeds ROAD++ valid duplex/triplet constraints
3. **GT-conditioned reasoning**: given ground-truth labels, generate natural language reasoning explanation

## Usage

```bash
python baseline/run_smolvlm.py
```

## Related

- [[projects/road-reason|ROAD_Reason]] | [[datasets/road-plusplus|ROAD++]]
- [[directions/constrained-vlm-reasoning|Constrained VLM Reasoning (Approach 3)]]
