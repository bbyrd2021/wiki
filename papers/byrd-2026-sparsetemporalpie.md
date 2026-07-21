---
type: paper
title: "SparseTemporalPIE: Cross-Attention over Sparse Temporal Context for Pedestrian Crossing Intention Prediction"
aliases: [ST-PIE, SparseTemporalPIE paper]
created: 2026-07-20
updated: 2026-07-20
sources:
  - "wiki/raw/SparseTemporalPIE_IEEE_draft.pdf"
tags: [paper, pedestrian-intent, cross-attention, pose, lab-paper]
status: complete
authors: "Byrd et al. (anonymous submission)"
year: 2026
venue: "IEEE (under review; draft hosted by Dr. Moradi)"
arxiv: ""
datasets_used: [pie, jaad]
---

# SparseTemporalPIE: Cross-Attention over Sparse Temporal Context for Pedestrian Crossing Intention Prediction

**The lab's own paper** — the IEEE-format writeup of [[methods/sparse-temporal-pie|SparseTemporalPIE v3/v4]], developed in [[projects/efficient-pie|EfficientPIE / SparseTemporalPIE]]. 8 pages, anonymous submission, draft circulated by Dr. Moradi 2026-07-20. Full experimental narrative lives in [[findings/sparse-temporal-pie-results]].

## Claimed contributions

1. Sparse multi-frame cross-attention: current frame queries up to K ≤ 4 evenly spaced context frames from the 15-frame observation window through a single cross-attention layer.
2. 68-d ViTPose-B pose representation (34-d static keypoints + 34-d frame-to-frame velocity), fused additively into the 1280-d backbone embeddings via a biasless linear projection.
3. Late fusion of structured bbox-trajectory (12-d) and ego/behavioral (5-d) features at the classifier.
4. Trained under the inherited IDIL curriculum of [[papers/efficientpie-ijcai-2025|EfficientPIE]] (step 0: 50 epochs; steps 2–14: 30 epochs each, adaptive distillation).

## Headline numbers

- [[datasets/pie|PIE]] test: **0.926 accuracy / 0.947 AUC / 0.957 F1** (893 samples, 92 held-out pedestrians) — new SOTA; AUC +0.030 over EfficientPIE, +0.037 over prior best published (VMI 0.910).
- [[datasets/jaad|JAAD]]: 0.878 acc / **0.885 AUC** / 0.633 F1 — trades −0.012 accuracy for +0.025 AUC vs EfficientPIE.
- Latency: 2.50 ms model-only, 6.38 ms end-to-end incl. online ViTPose-B (RTX 3090, batch 128). ~9.0M params.
- Framing: AUC is the headline metric — the PIE test set is 86% positive, so accuracy is prior-inflated; AUC reflects calibrated risk scores for downstream planners (ties to [[directions/uncertainty-aware-intent]]).

Ablations: **Table III** (v3 vs v4, cross-attention removed → v4 peaks at IL step 2 with 0.9194 and *regresses* to 0.9127 by step 14; cross-attention is what keeps later IL steps productive). **Table IV** (EfficientPIE-init vs ImageNet-init full v3 → ImageNet falls short by 0.0045 acc / 0.0257 AUC; val set too small (~92 peds) to select on).

## Reviewer feedback — Dr. Qingge, 2026-07-20 (code-verified)

Three points, checked against `EfficientPIE/models/SparseTemporalPIE_v3.py` and `EfficientPIE/utils/sparse_dataset_v3.py`:

### 1. The t = 0 case: Eq. (1) and Section VII-B disagree; both need precision

- **Eq. (1)** `I_ctx(t) = linspace(0, t−1, min(K,t))` yields an *empty* context set at t = 0. Implemented literally, all key slots would be masked padding and PyTorch `MultiheadAttention` returns NaN. The code instead special-cases t = 0 (`sparse_dataset_v3.py:112–117`) to `I_ctx(0) = [0]` — **the current frame is its own sole key/value**, with a valid mask. Fix: state "Eq. (1) applies for t ≥ 1; at t = 0, I_ctx(0) = {0}."
- **"Query and keys/values functionally identical" — correct.** Same frame, same shared backbone, same pose fusion; pose velocity is zero at index 0. At inference the vectors are literally identical.
- **"Collapses to a residual passthrough" — imprecise.** With one unmasked key the softmax weight is identically 1 *independent of the query*, so `attn_out = W_O(W_V·e₀ + b_V) + b_O` — a learned, query-independent affine transform of the current embedding, added residually and LayerNormed. What collapses is the *selection mechanism*, not the output. Precise wording: at t = 0 the block contributes no cross-frame information and no input-dependent attention; it degenerates to an extra learned linear residual layer, not an identity.
- Training-time nuance: attention-weight dropout (p = 0.1) zeroes the single unit weight for ~10% of step-0 samples — a stochastic-depth-like effect on the branch.
- **Bonus erratum found during verification:** paper states I_ctx(14) = [0, 4, **9**, 13] (Sections III-E, VI); `np.linspace(0, 13, 4, dtype=int)` truncates to [0, 4, **8**, 13]. Figure 4's step-14 row should be re-checked too.

### 2. Sparse vs dense sampling ablation

Suggested "if time permits." Deferred for now (user decision 2026-07-20). Infrastructure note: the absolute-frame-ID keypoint cache was designed so any sampling protocol runs without recomputation, and `n_context` is a dataset parameter — the run is cheap to set up if the paper gets a revision cycle.

### 3. Are Tables III and IV repetitive?

No — orthogonal ablation axes (III: architecture at fixed init; IV: initialization at fixed architecture) — but they share the full-v3 anchor row (0.9261 / 0.9468), which reads as repetition. Only three distinct runs span both tables, so merging into one 3-row ablation table (variant | params | best step | test acc | AUC) is the cleanest response; a caption cross-reference is the minimal fix.

## Relation to wiki pages

- Method details + v3/v4 architecture: [[methods/sparse-temporal-pie]]
- Full results narrative incl. IL trajectories: [[findings/sparse-temporal-pie-results]]
- Upstream baseline: [[papers/efficientpie-ijcai-2025]]
- Datasets: [[datasets/pie]], [[datasets/jaad]] — note the paper is careful that PIE labels encode crossing *intention* while JAAD labels encode crossing *action* (models trained independently per dataset; no cross-dataset transfer claimed)
- Calibration framing: [[directions/uncertainty-aware-intent]]
