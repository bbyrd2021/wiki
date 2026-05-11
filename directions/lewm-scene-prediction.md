---
type: direction
title: "LeWM Scene Prediction (ROAD_Reason Approach 6)"
aliases: ["LeWM", "Approach 6", "le-wm"]
created: 2026-04-07
updated: 2026-05-11
sources: ["ROAD_Reason/docs/APPROACHES.md", "wiki/raw/LeWM.pdf"]
tags: [direction, road-plusplus, world-model, workstation-feasible]
status: complete
novelty: true
feasibility: workstation
datasets_required: [road-plusplus]
approach_number: 5
---

# LeWM Scene Prediction

**Approach 6** in [[projects/road-reason|ROAD_Reason]]. Exploratory — adaptation to driving video is novel.

## Key Advantage: Workstation-Feasible

- **~15M parameters** (ViT-tiny encoder ~5M + 6-layer transformer predictor ~10M), trains in hours on a single A6000 / RTX 3090 / 4090
- First JEPA to train **stably end-to-end from raw pixels** with no EMA, no stop-gradient, no pretrained encoder freeze — see [[papers/maes-2026-lewm|Maes et al. 2026 (arXiv:2603.19312)]]
- **Only one hyperparameter** (`λ` for the SIGReg anti-collapse term), bisection search vs PLDM's `O(n^6)` 6-hyperparameter grid
- Plans **~48× faster than DINO-WM** at inference (0.98s vs 47s full planning on Push-T)
- Reference implementation: `stable-worldmodel-v1` package from the paper authors

## Method

1. Adapt LeWM to ROAD-Waymo clips: encoder maps frames → latent z_t, predictor maps (z_t, action_t) → ẑ_{t+1}. Per the paper, action conditioning enters via AdaLN.
2. Apply t-norm constraint loss across predicted future frames — future states must satisfy logic requirements
3. Use surprise detection (anomalous latent predictions) for unusual pedestrian behavior flagging — the paper's violation-of-expectation framework (paired t-test p < 0.01 sensitivity to physical perturbations on Push-T / Two-Room / OGBench-Cube) is the direct evaluation primitive

### Action labels — open question

ROAD-Waymo doesn't ship explicit ego-action labels per frame. Two options:
- Use [[datasets/covla|CoVLA]] instead — it has full CAN bus + 60-point trajectory, which is the natural fit for LeWM-style training. Cross-evaluate on ROAD-Waymo for intent.
- Learn an inverse dynamics model from consecutive frames first, then feed predicted actions into LeWM. The paper flags this as a future direction.

## Why Relevant

- World model predicting future scene states is a natural fit for intent prediction
- T-norm constraints on future predictions is unexplored
- Anomaly detection capability complements primary intent prediction

## What's Novel

Current LeWM benchmarks are robotics-only. Adaptation to driving video with constraint supervision on predictions is novel.

## Related

- [[papers/maes-2026-lewm|Maes 2026 — LeWorldModel]] — the reference paper
- [[projects/road-reason|ROAD_Reason]] | [[directions/jepa-intent-head|V-JEPA 2 (Approach 5)]]
- [[papers/chen-2026-vl-jepa|VL-JEPA]] — sibling JEPA family (1.6B params, InfoNCE anti-collapse, language-conditioned)
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[concepts/encoder-collapse|Encoder collapse]] — SIGReg is LeWM's principled alternative to the usual EMA/stop-grad heuristics
