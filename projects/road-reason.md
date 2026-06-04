---
type: project
title: "ROAD_Reason — Logic-Constrained Scene Reasoning"
aliases: ["ROAD_Reason", "ROAD Reason"]
created: 2026-04-07
updated: 2026-06-04
sources:
  - "ROAD_Reason/docs/CLAUDE.md"
  - "ROAD_Reason/docs/APPROACHES.md"
  - "ROAD_Reason/README.md"
tags: [project, road-plusplus, neuro-symbolic, vlm, thesis-core]
status: complete
---

# ROAD_Reason

Master's research on **logic-constrained scene understanding and reasoning** for autonomous driving, built on the [[datasets/road-plusplus|ROAD++]] dataset. Goal: produce natural language reasoning about scene intent grounded in propositional logic constraints from the dataset's compositional label structure.

**Supervisor:** Dr. Moradi | **Repo:** `/data/repos/ROAD_Reason/`

## Core Idea

ROAD-Waymo labels capture *what* happens (`Ped-Wait2X-RhtPav`) but not *why*. Identical triplets can arise from different causal origins (genuine hazard stop vs. false positive stop). The novel contribution (Approach 4) adds a causal reasoning head that distinguishes these origins and produces grounded natural language explanations alongside structured predictions.

## Constraint Background

ROAD++ encodes implicit constraints via its compositional label structure:
- **49 valid duplexes** (agent + action) from 220 possible combinations
- **86 valid triplets** (agent + action + location) from 3,520 possible combinations
- Stored in `duplex_childs` and `triplet_childs` in the annotation JSON

**T-norm constraint loss** (Łukasiewicz): `violation = max(0, p(LarVeh) + p(Xing) - 1)` — penalizes co-prediction of invalid label combinations.

## Research Roadmap

| Approach | Model | Novel? | Status |
|----------|-------|--------|--------|
| 1. 3D-RetinaNet baseline | 3D-RetinaNet | No | Starting point |
| 2. Neuro-symbolic RetinaNet | 3D-RetinaNet + t-norm | No (in paper) | Replication |
| **3. Qwen2.5-VL multi-task** | **Qwen2.5-VL-7B + task heads** | **Yes** | **New — in design** |
| **4. Constrained VLM reasoning** | **OpenMixer + DSDAG + VLT** | **Yes** | **Primary** |
| 5. V-JEPA 2 intent head | V-JEPA 2 + MLP | Yes | Novel application |
| 6. LeWM scene prediction | LeWM (~15M params) | Yes | Workstation-feasible |
| 7. JEPA + VLM hybrid | VL-JEPA + t-norm | Yes | Long-term |
| **8. VLM Reasoning Layer** | **Frozen 3D-RetinaNet + VLM (cached JSON, late fusion)** | **Yes** | **New — Moradi 2026-06-02 pivot, staged** |

## Approach 3: Qwen2.5-VL Multi-Task (New)

- Shared Qwen2.5-VL-7B backbone fine-tuned separately on BDD-X, CoVLA, ROAD-R, then jointly
- BDD-X: LM head → action + justification text
- ROAD-R: detection + tube-linking + 5× sigmoid heads + t-norm (LLM bypassed)
- CoVLA: LM head → plain_caption + risk + trajectory MLP → 10×3 waypoints
- Phase 1: three separate LoRA adapter sets; Phase 2: joint with task sampling 20/40/40%
- Produces SOTA VLM baseline + seeds Approach 4 Stage 1 weights

See [[directions/qwen-multitask-finetuning|Approach 3 direction]] | [[methods/qwen25-vl-multitask|Architecture spec]]

## Approach 4 Architecture (Primary Contribution)

```
Video clip → CLIP-ViP (frozen) → OpenMixer backbone
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
          Structured head                        DSDAG causal head
          raw triplet logits                     VLT → reasoning embedding
                    │                                      │
                    └─────────── f(reasoning) ────────────┘
                                      │
                             reweighted triplet logits
                                      │
                             T-norm constraint loss
                                      │
                          Triplet mAP  +  "Why" reasoning
```

**Key components:**
- **OpenMixer** (Bao et al., WACV 2025) — DETR-style detector on frozen CLIP-ViP features; S-OMB (spatial) + T-OMB (temporal) + DFA (open vocab)
- **DSDAG** (Cheng et al., MCAM, ICCV 2025) — causal DAG: Xs → Action(Y) → Xe mediated by Environment(Z); Xs from early frames, Xe from late frames
- **VLT** — Vision-Language Transformer with sparsity loss (anti-hallucination); outputs reasoning embedding r
- **Logit reweighting** — `L_final = L_raw ⊙ f(r)`; causal reasoning gates structured predictions end-to-end
- **Entropy-gated fusion** — at inference, trusts causal head more when structured head is uncertain

See [[methods/multimodal-causal-driving|Full Architecture Spec]] for module-by-module breakdown with math and defense notes.

**Training stages:**
1. Causal head pre-training on BDD-X + CoVLA (real language supervision)
2. Structured head warm-up on ROAD-Waymo
3. Joint fine-tuning — ROAD-Waymo GT labels + Claude-generated pseudo-labels (Opus 4.6 for corner cases, Sonnet 4.6 for bulk) via Anthropic Message Batches API

**Why novel:** First combination of open-vocabulary action detection (OpenMixer) + causal state modeling (DSDAG) + neuro-symbolic label constraints (t-norm) on ROAD-Waymo.

## Approach 5: V-JEPA 2 + Intent Head

- V-JEPA 2 (arXiv:2506.09985) as frozen spatiotemporal encoder
- Lightweight intent head (MLP or Transformer) on JEPA features
- Train on ROAD-Waymo action/location + binary crossing intent
- Extension: add t-norm constraint loss to head outputs
- Reference: Drive-JEPA (arXiv:2601.22032) demonstrates V-JEPA → AV pipeline

## Approach 6: LeWM (Workstation-feasible)

- ~15M params, trains in hours on single RTX 3090/4090
- Predicts future latent scene states from current observations
- T-norm constraints on future predictions — unexplored
- Anomaly detection via surprise in latent predictions

## Experiment Status

| Experiment | Description | Status | Best result |
|------------|-------------|--------|-------------|
| Exp1 | Frozen ViT + GT boxes + BCE + Łukasiewicz t-norm | Complete (ep6) | action mAP=22.2%, duplex=12.3%, triplet=8.8% |
| **Exp1b** | **LoRA + FCOS dense detection + focal loss + Gödel t-norm** | **Complete (ep15, Apr 20)** | Internal: agent=60.6%, action=32.4% (macro-mAP, fg tokens) · Baseline-compat: agent=3.2%, action=1.6% (f-mAP) |
| **Exp2** | **DETR-style set prediction: 100 learnable queries + Hungarian matching + L1+GIoU + Gödel t-norm** | **Complete (ep30, Apr 24)** | f-mAP: agent=0.63% (28x below RetinaNet); localization bottleneck |
| **Exp2b** | **Deformable DETR + EfficientNet-B0/FPN + iterative refinement + auxiliary losses** | **Complete (ep27/30, May 4)** | f-mAP: agent=1.71% (10x below RetinaNet); VLM localization bottleneck confirmed |
| **Exp2c** | **Frozen-DETR: EfficientNet-FPN + DETR encoder (4 scales) + CLIP ViT-L/14 (frozen)** | **Training ep24/30 (May 12)** | val action mAP 43.72% (ep23 best); f-mAP: agent 1.76%, recall 59% (ep15 eval) |
| **Exp2d v1** | **Swin-L backbone (replaces EfficientNet-B0) + Frozen-DETR** | **Complete (ep4, May 12)** | val action mAP 44.55% (ep2 best); overfitting by ep3 |
| **Exp2d v2** | **Swin-L + DINO COCO pretrained encoder (192 keys) + DETR augmentations + drop path 0.2** | **Training (May 12)** | Anti-overfitting relaunch; resolution bottleneck identified; see [[findings/exp2d-swin-detr-v2]] |
| **Exp2e** | **R50 @ 800×1333 (paper's exact config) + CLIP ViT-L/14 + t-norm** | **Training (May 17)** | f-mAP: agent 5.5%, agent_ness 5.5% (ep11); resolution helps but score-localization decorrelation identified; see [[findings/exp2e-r50-frozen-detr]] |
| **Exp2f** | **Flat 184-dim sigmoid head (baseline classification design) — fixes score-decorrelation** | **Training ep19/30 (May 26)** | f-mAP: agent=4.40% (ep14); matched action mAP=0.199 (ep18 best); see [[findings/exp2f-flat-head]] |
| **Exp2g** | **MS-DETR: two-stage proposals (900 queries) + O2M (k=6) + full encoder loss + softmax agent** | **Training ep1/30 (May 26)** | Full Frozen-DETR/MS-DETR replication; 358.7M params (54.2M trainable); see [[findings/exp2g-msdetr]] |
| **Exp3** | **BDD-X captioning: LoRA r=16 on LLM + merger; CE on response tokens; 3 epochs** | **Ready to train (Apr 21)** | TBD |

**Exp1b** redesigns Exp1 from oracle-box classification to FCOS dense detection: every spatial ViT token predicts agentness + box + labels. Internal macro-mAP is strong, but baseline-compatible f-mAP at IoU=0.5 is only 3.2% — FCOS box quality is the bottleneck, not classification. See [[findings/exp1b-fcos-detection|Exp1b finding page]].

**Exp2** replaces FCOS with a DETR-style clip-level decoder: 100 learnable queries attend to all T×H'×W'=2048 spatiotemporal tokens, each query predicts a full tube (T boxes + tube-level labels). Hungarian matching, L1+GIoU supervision, and real learned agentness head. Warm-starts ViT+LoRA from Exp1b. See [[findings/exp2-detr-detection|Exp2 finding page]].

**Exp2b** redesigns the decoder as standard Deformable DETR with three fixes: (1) per-frame decoding with temporal self-attention (replaces temporal stacking), (2) iterative box refinement with per-layer box heads, (3) auxiliary losses at every decoder layer. Adds EfficientNet-B0 + FPN as spatial backbone alongside Qwen ViT, fused via learned gates. 300 queries, 692M total params (15.6M trainable). Warm-starts from Exp2. See [[findings/exp2b-deformable-detr|Exp2b finding page]].

**Exp2c** faithfully implements [[papers/fu-2024-frozen-detr|Frozen-DETR]] (Fu et al., NeurIPS 2024) to fix Exp2b's two gaps: (1) adds a 6-layer deformable encoder so CNN and VLM tokens fuse through self-attention rather than a scalar gate, and (2) replaces Qwen2.5-VL ViT with CLIP ViT-L/14@336px (Dr. Moradi approved, 2026-05-07). CLIP patch tokens enter as 4th encoder scale, are stripped after encoding; CLS token is injected per-layer into the decoder. ~445M total params (~15.7M trainable), saves ~5 GB GPU vs Exp2b. See [[findings/exp2c-frozen-detr|Exp2c finding page]].

**Exp2d** scales the spatial backbone from EfficientNet-B0 (~5.3M) to Swin-L (~195M) following the [[comparisons/yolov8x-vs-swin-l-backbone|backbone comparison]]. v1 showed fast learning (val action mAP 44.55% at ep2, beating exp2c's 43.72% at ep23) but overfitted by epoch 3 due to zero augmentation + small dataset (7K clips) + large model (214M trainable). v2 applies three anti-overfitting fixes: (1) DINO COCO pretrained encoder/decoder (192 keys from 51.9 AP checkpoint), (2) DETR-standard augmentations + strong color augmentation, (3) Swin-L drop path 0.2 + weight decay 0.05. See [[findings/exp2d-swin-detr-v2|Exp2d v2 finding page]].

**Exp2e** addresses the root cause of low f-mAP identified across exp2c/2d: **input resolution** (384-448px vs paper's 800×1333). At low resolution, small objects become sub-pixel and can't achieve IoU≥0.5. Exp2e faithfully replicates the paper: ResNet-50 with FrozenBatchNorm at variable aspect ratio (train multi-scale [480..800], val 800×max1333). 457 DINO COCO keys transfer cleanly (R50 architecture matches exactly). The only difference from the paper: 5 ROAD multi-label heads + Gödel t-norm loss. **Post-eval finding:** resolution improved f-mAP ~4× (1.4% → 5.5% agent) but a deeper issue remains — **score-localization decorrelation** (top-scoring detections have terrible boxes because unmatched queries get zero gradient on classification heads). See [[findings/exp2e-r50-frozen-detr|Exp2e finding page]].

**Exp2f** fixes the score-localization decorrelation by replacing exp2e's 6 separate classification heads with a **single flat `nn.Linear(256, 184)`** — the same design used by the 3D-RetinaNet baseline. Sigmoid + focal loss on ALL 300 queries × 184 dims means unmatched queries get explicit target=0 supervision on every class dimension, not just agentness. Result: agent f-mAP jumped from 1.4% to 4.4% (3x improvement). Still 4x below baseline due to query coverage limitations. See [[findings/exp2f-flat-head|Exp2f finding page]].

**Exp2g** implements the full MS-DETR recipe from the Frozen-DETR paper to close the remaining gap. Three core additions: (1) two-stage query init from encoder proposals (900 queries, up from 300), (2) one-to-many auxiliary loss (k=6 matches per GT, 6x more positive training signals), (3) full encoder proposal supervision matching the reference exactly. Also adds softmax on agent dims (single-label) and 4D iterative box refinement. Systematic audit closed all remaining gaps vs reference repo. See [[findings/exp2g-msdetr|Exp2g finding page]].

**Exp3** fine-tunes Qwen2.5-VL-7B on BDD-X (16,553 train examples): LoRA r=16 on LLM attention + MLP layers, merger trainable, ViT frozen. Input: single BDD100K keyframe. Output: "Action: X\nJustification: Y." Loss: CE on assistant tokens only (label mask via prompt-length comparison). Evaluated with BLEU-4 action / justification / combined.

## Running

```bash
# Experiment 3 — BDD-X captioning (ready to train)
cd /data/repos/ROAD_Reason
bash experiments/exp3_bddx/run.sh           # train + BLEU-4 eval

# Experiment 2 — DETR-style tube detection (training)
cd /data/repos/ROAD_Reason
bash experiments/exp2_detr_qwen/run.sh      # train + frame-mAP + video-mAP

# Experiment 1b — FCOS dense detection (complete)
python -u experiments/exp1b_road_r/train.py
python -u experiments/exp1b_road_r/eval.py --out experiments/exp1b_road_r/logs/eval_results.json
python -u experiments/exp1b_road_r/eval_baseline_compat.py

# Experiment 1 — oracle-box classification (complete)
python experiments/exp1_road_r/train.py
python experiments/exp1_road_r/eval.py

# SmolVLM baselines (reference only)
python baseline/smolvlm_constrained.py

# Dataset analysis
python analysis/compute_stats.py
```

## Related

- [[datasets/road-plusplus|ROAD++ Dataset]] | [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[directions/qwen-multitask-finetuning|Approach 3: Qwen2.5-VL Multi-Task]] | [[directions/constrained-vlm-reasoning|Approach 4: Constrained VLM]] | [[directions/jepa-intent-head|Approach 5: V-JEPA 2]] | [[directions/vlm-reasoning-layer|Approach 8: VLM Reasoning Layer]]
- [[projects/pedestrian-intent|PedestrianIntent++]] (dataset documentation)
