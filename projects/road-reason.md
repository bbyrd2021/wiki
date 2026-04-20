---
type: project
title: "ROAD_Reason — Logic-Constrained Scene Reasoning"
aliases: ["ROAD_Reason", "ROAD Reason"]
created: 2026-04-07
updated: 2026-04-20
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
| **Exp1b** | **LoRA + FCOS dense detection + focal loss + Gödel t-norm** | **Complete (ep15, Apr 20)** | agent=60.6%, action=32.4%, loc=50.0%, duplex=23.1%, triplet=17.5% (macro-mAP, fg tokens) |

**Exp1b** redesigns Exp1 from oracle-box classification to paper-analogous FCOS dense detection: every spatial ViT token predicts agentness + box + labels; no GT boxes needed at inference. Agentness is a real learned score (replaces hardcoded 1.0). See [[findings/exp1b-fcos-detection|Exp1b design page]] for full architecture, loss breakdown, and expected results.

## Running

```bash
# Experiment 1b — FCOS dense detection (current)
cd /data/repos/ROAD_Reason
python -u experiments/exp1b_road_r/train.py
python -u experiments/exp1b_road_r/eval.py --out experiments/exp1b_road_r/logs/eval_results.json

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
- [[directions/qwen-multitask-finetuning|Approach 3: Qwen2.5-VL Multi-Task]] | [[directions/constrained-vlm-reasoning|Approach 4: Constrained VLM]] | [[directions/jepa-intent-head|Approach 5: V-JEPA 2]]
- [[projects/pedestrian-intent|PedestrianIntent++]] (dataset documentation)
