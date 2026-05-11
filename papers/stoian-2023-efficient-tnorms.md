---
type: paper
title: "Exploiting T-norms for Deep Learning in Autonomous Driving"
aliases: ["efficient t-norms", "memory-efficient t-norms"]
authors: "Stoian et al."
year: 2023
venue: "NeSy"
arxiv: "2402.11362"
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/efficient-tnorms_nesy2023.pdf"
tags: [paper, neuro-symbolic, t-norms, memory-efficient, road-r, autonomous-driving, constraints]
status: complete
---

# Efficient T-norms for Deep Learning in Autonomous Driving

Stoian, Giunchiglia, Lukasiewicz. NeSy 2023. [arXiv:2402.11362](https://arxiv.org/abs/2402.11362)

## Key Idea

Standard t-norm-based constraint losses require enormous memory (>25 GiB for ROAD-R, estimated >100 GiB for full constraint sets) because they materialize the full constraint satisfaction matrix G of size [D × |Π|] (D anchor boxes × number of constraints per frame).

This paper reformulates the t-norm loss using **sparse tensors**. Constraints are expressed as positive/negative constraint matrices C+ and C- (sparse binary matrices indicating which labels appear positively/negatively in each constraint). The satisfaction matrix G is computed efficiently via sparse matrix operations without materializing the full dense tensor.

Result: t-norm losses that fit in <25 GiB GPU memory, making them practical for complex autonomous driving constraint sets.

## Results

- Runs on GPUs with <25 GiB memory (vs >100 GiB for standard implementation)
- T-norm losses improve performance in all settings, especially with limited labeled data (+1.85% with 10% labels, +3.95% with 20% labels)
- T-norm losses on unlabeled data (semi-supervised) provide further improvement (+2.75% over fully supervised baseline)
- Evaluated on ROAD-R dataset

## Relevance to ROAD_Reason

ROAD_Reason's current t-norm implementation (`tnorm_loss.py`) uses the buffer-based approach with invalid pair/triple buffers. For ROAD-Waymo's constraint set (49 valid duplexes, 86 valid triplets), memory is manageable.

However, if constraints are applied in the decoupled Approach 4 pipeline (post-detection on tube features), the constraint tensor is much smaller (one prediction per tube, not per anchor box), making memory less of a concern. The efficient formulation becomes important if:
- Scaling to more complex constraint sets
- Applying constraints during training with many detection proposals
- Semi-supervised scenarios with unlabeled ROAD-Waymo data

## Related

- [[papers/marconato-2022-road-r|ROAD-R]] — defines the constraint set
- [[papers/stoian-2024-pishield|PiShield]] — hard guarantee alternative (same research group)
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — t-norm methodology
