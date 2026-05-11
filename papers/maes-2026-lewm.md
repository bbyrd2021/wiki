---
type: paper
title: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
aliases: ["LeWM", "LeWorldModel", "Maes 2026 LeWM"]
created: 2026-05-11
updated: 2026-05-11
sources:
  - "wiki/raw/LeWM.pdf"
tags: [paper, jepa, world-model, sigreg, anti-collapse, end-to-end, control, planning, mpc]
status: complete
authors: "Maes et al."
year: 2026
venue: "arXiv preprint, Mar 2026"
arxiv: "2603.19312"
datasets_used: []
---

# LeWorldModel — Stable End-to-End JEPA from Pixels

Maes, Le Lidec, Scieur, **LeCun**, **Balestriero**. [arXiv:2603.19312v2](https://arxiv.org/abs/2603.19312) (24 Mar 2026). Mila / Université de Montréal / NYU / Samsung SAIL / Brown.

The first JEPA that **trains stably end-to-end from raw pixels** without stop-gradient, EMA target encoders, pretrained backbones, or multi-term auxiliary losses. The whole training objective is two terms: a next-embedding MSE prediction loss + a single anti-collapse regularizer (**SIGReg**) tuned by one scalar. The resulting ~15M-parameter ViT-tiny + small predictor is trainable on a single GPU in a few hours, and at inference time plans **48× faster** than DINO-WM (0.98s vs 47s full planning) while matching or beating it on continuous control tasks.

---

## Why This Matters for the Lab

LeWM is the published basis for [[directions/lewm-scene-prediction|Approach 6 (LeWM scene prediction)]] in [[projects/road-reason|ROAD_Reason]]. Until now that direction was speculative — referenced only as a workstation-feasible JEPA option. This page makes it concrete:

1. **Workstation-feasible JEPA.** 15M params, ViT-tiny encoder + 6-layer transformer predictor, *single A6000 in hours*. Compare to [[papers/chen-2026-vl-jepa|VL-JEPA]] (1.6B params, 24 nodes × 8× H200 × 4 weeks). LeWM is the realistic JEPA option for an MS thesis budget.
2. **One hyperparameter.** SIGReg's `λ` is the only knob. Bisection search is `O(log n)` vs PLDM's 6 hyperparameters with `O(n^6)` cost. For a thesis where every experiment costs days of GPU time, this is decisive.
3. **Action-conditioned latent prediction.** Pred(z_t, a_t) → ẑ_{t+1}. Maps cleanly to driving: condition on (steer, brake, throttle) and predict the next scene latent. The existing direction already calls out this mapping; LeWM gives the architecture template.
4. **Violation-of-expectation evaluation.** Surprise = ‖ẑ_{t+1} − z_{t+1}‖² on perturbed trajectories. LeWM detects physically-implausible events (teleportation) more strongly than visual changes (color shift). This is a clean primitive for driving anomaly detection — "the pedestrian appeared from nowhere" → high surprise.
5. **Anti-collapse without heuristics.** Contrasts with VL-JEPA's InfoNCE-based anti-collapse (which needs an LR multiplier of ×0.05 on the Y-Encoder + careful scheduling). SIGReg is a single regularizer term derived from a normality test — no asymmetric encoder roles, no stop-gradient.

---

## Architecture

Two components, both learned jointly end-to-end:

| Component | Implementation | Params | Notes |
|---|---|---:|---|
| **Encoder** `enc_θ(o_t) → z_t` | ViT-tiny (patch 14, 12 layers, 3 heads, hidden 192) + projection MLP w/ BatchNorm | ~5M | CLS token of last layer → 1-layer MLP w/ BN. BN is necessary because ViT's final LayerNorm interferes with SIGReg's anti-collapse target. |
| **Predictor** `pred_φ(z_t, a_t) → ẑ_{t+1}` | Transformer, 6 layers, 16 heads, 10% dropout, **AdaLN** for action conditioning | ~10M | History length N input. Causal temporal mask. Autoregressive future rollout. AdaLN params init at zero so action conditioning ramps in progressively. |

```
o_1, ..., o_T  (raw pixel frames)
    │
    ▼
[ ViT-tiny encoder ] ──── z_1, ..., z_T  (latent embeddings)
                              │
                              │  ◄─── a_1, ..., a_T (actions, via AdaLN)
                              ▼
                  [ 6-layer Transformer predictor ]
                              │
                              ▼
                       ẑ_2, ..., ẑ_{T+1}  (next-step predictions)
                              │
                              ▼
            L_LeWM = ‖ẑ_{t+1} − z_{t+1}‖² + λ · SIGReg(Z)
```

**Total ~15M trainable parameters.** ResNet-18 also works as a drop-in encoder replacement (Tab. 8) — the method is largely encoder-agnostic.

### Training Objective (Eq. 3)

```
L_LeWM = L_pred + λ · SIGReg(Z)
       = ‖ẑ_{t+1} − z_{t+1}‖²  +  λ · SIGReg(Z)
```

- **L_pred** (teacher-forcing MSE): standard next-embedding prediction error.
- **SIGReg** (Sketched-Isotropic-Gaussian Regularizer): the only anti-collapse term. Projects the embedding tensor Z ∈ ℝ^{N×B×d} onto M=1024 random unit-norm directions, applies the **Epps–Pulley univariate normality test** to each one-dim projection, averages the test statistics. By the **Cramér–Wold theorem**, matching every 1D marginal to N(0,1) ⟺ matching the full joint distribution to an isotropic Gaussian.

Default settings: `λ = 0.1`, `M = 1024 projections`. The paper shows performance is largely insensitive to M (ablation, App. G) — λ is effectively the only knob.

### What LeWM does NOT do (vs other JEPAs)

- **No exponential moving average (EMA) target encoder** (unlike I-JEPA, V-JEPA 2)
- **No stop-gradient** between encoder and predictor (unlike I-JEPA, V-JEPA, Brain-JEPA)
- **No pretrained encoder freeze** (unlike DINO-WM, which freezes DINOv2)
- **No multi-term VICReg-style loss** (unlike PLDM, which has 7 loss terms)
- **No auxiliary supervision** (no proprioception, no rewards, no task labels — purely pixel + action)

---

## Results

### Planning performance (Fig. 6 — success rate %)

| Env | LeWM (15M, pixel) | PLDM | DINO-WM (124M, frozen DINOv2) | Goal-cond. BC | GCIVL / GCIQL |
|---|---:|---:|---:|---:|---:|
| Push-T | **90** | 76 | 73 | — | — |
| OGBench-Cube | 74 | 65 | **80** | 65 | 50 / 48 |
| Two-Room | 87 | **100** | 100 | 100 | 100 / 100 |
| Reacher | **86** | 78 | 79 | — | — |

LeWM matches PLDM on harder tasks (+14 to +18 on Push-T) and stays competitive with the foundation-model-based DINO-WM despite using ~8× fewer parameters and no pretrained features. On simpler environments (Two-Room) LeWM is slightly worse — the paper notes SIGReg's isotropic-Gaussian prior may be a poor fit when intrinsic environment dimensionality is very low.

### Planning speed (Fig. 3 — full planning, averaged over 50 runs)

| Method | Full planning time |
|---|---:|
| LeWM | **0.98 s** |
| DINO-WM | 47 s |

**~48× speedup.** Comes from ~200× fewer tokens per observation than DINO-WM. Under fixed-FLOPs budgets LeWM still wins on Push-T (90 vs 13) and on OGBench-Cube (74 vs 48).

### Physical structure of the latent space (Tab. 1, Push-T probing)

Linear / MLP probes predicting physical quantities from frozen latents:

| Property | LeWM (linear / MLP MSE↓) | PLDM | DINO-WM (124M) |
|---|---|---|---|
| Agent location | **0.052** / 0.004 | 0.090 / 0.014 | 1.888 / 0.003 |
| Block location | 0.029 / **0.001** | 0.122 / 0.011 | **0.006** / 0.002 |
| Block angle | 0.187 / 0.021 | 0.446 / 0.056 | **0.050** / **0.009** |

LeWM beats PLDM on every quantity and is close to DINO-WM despite DINO-WM's encoder having seen ~124M images during pretraining (Push-T training would never expose LeWM to that volume).

### Violation-of-Expectation (Fig. 10)

Surprise = prediction MSE on perturbed trajectories. Across Two-Room / Push-T / OGBench-Cube, LeWM assigns **significantly higher surprise to physical perturbations (object teleportation) than to visual perturbations (color shifts)** — paired t-test p < 0.01. The model is more sensitive to physics violations than appearance violations, which is the desired behavior for a world model.

### Ablations (App. G)

- SIGReg projections M ∈ [256, 4096]: largely insensitive. M=1024 is fine.
- Number of SIGReg quadrature knots: insensitive.
- Embedding dim: needs to be "sufficiently large", then saturates.
- Encoder swap (ViT-tiny → ResNet-18): both architectures work, method is largely backbone-agnostic.

---

## Connections to the Lab

### Direct beneficiaries

- [[directions/lewm-scene-prediction|Approach 6 (LeWM scene prediction)]] — this is the **reference paper**. The direction can now be specified concretely: 15M-param JEPA, single-GPU training, action-conditioned latent prediction, MPC at planning time.
- [[projects/road-reason|ROAD_Reason]] — Approach 6 becomes the most workstation-realistic of the JEPA-family options (compare to Approach 5's V-JEPA 2, which assumes a frozen pretrained encoder of size ~300M).

### Sibling JEPA papers

- [[papers/chen-2026-vl-jepa|VL-JEPA (Chen 2026)]] — vision-language JEPA. Same LeCun co-author; same JEPA family; very different scale (1.6B vs 15M) and anti-collapse strategy (InfoNCE vs SIGReg). VL-JEPA targets *language-conditioned answer prediction*; LeWM targets *action-conditioned future-state prediction*. They are complementary, not redundant.
- [[directions/jepa-intent-head|V-JEPA 2 + Intent Head (Approach 5)]] — frozen pretrained V-JEPA 2 encoder + small head. LeWM is the **end-to-end** alternative: don't freeze anything, train a small encoder from scratch.

### Concept connections

- [[concepts/encoder-collapse|Encoder collapse]] — LeWM's whole reason for being is to prevent collapse without the usual heuristic toolkit (EMA, stop-grad, freezing). SIGReg is a principled alternative grounded in the Cramér–Wold theorem.

### Related work (per the paper's References section)

Generative world models compared: IRIS, DIAMOND, Δ-IRIS, OASIS, DreamerV3, Genie, HunyuanWorld, WorldGym. JEPA family: I-JEPA, V-JEPA, V-JEPA 2, Brain-JEPA, Echo-JEPA, DINO-WM (frozen-encoder JEPA WM), PLDM (end-to-end VICReg-based JEPA WM), OSVI-WM, **Causal-JEPA** (object-level latent interventions — could become a research direction). Anti-collapse theory: VICReg, **LeJEPA** (the paper that introduces SIGReg). MPC tradition: Hafner's Dreamer line, TD-MPC2, Navigation World Models.

---

## Limitations (per the paper's Conclusion)

- **Short planning horizons.** Auto-regressive latent rollouts accumulate prediction error; the paper uses MPC to mitigate (replan after K steps) but long-horizon planning remains open. Hierarchical world modeling is flagged as the natural next direction.
- **Offline data required.** Like all current JEPA WMs, training needs trajectories with action labels. Inverse-dynamics models that learn actions could remove this dependency.
- **Action labels required.** Same point — `a_t` must be available at training time.
- **Performance dips in low-intrinsic-dimensionality environments.** Two-Room is the example: forcing an isotropic Gaussian over a high-d latent for a structurally low-d environment hurts.

---

## Practical Notes for Lab Implementation

If/when Approach 6 is attempted in ROAD_Reason:

1. **Start from the paper's repo.** Maes et al. publish a `stable-worldmodel-v1` package (Ref [50]) for reproducible JEPA world modeling — the lab should clone and reuse rather than reimplement.
2. **Single-GPU training is plausible.** 15M params + ViT-tiny + small batch fits in A6000 memory comfortably.
3. **Action space.** ROAD-Waymo doesn't have explicit ego-action labels at every frame, but CAN-bus data exists in [[datasets/covla|CoVLA]] (60-point trajectory, steer/throttle). CoVLA is the more natural fit for action-conditioned LeWM training than ROAD-Waymo.
4. **Surprise as anomaly signal.** The violation-of-expectation framework gives a free anomaly detector. For driving: train LeWM on normal trajectories, deploy MSE(ẑ_{t+1}, z_{t+1}) as a real-time surprise score. Spikes correlate with novel events.

---

## Related

- [[directions/lewm-scene-prediction|LeWM Scene Prediction (Approach 6)]]
- [[directions/jepa-intent-head|V-JEPA 2 + Intent Head (Approach 5)]]
- [[papers/chen-2026-vl-jepa|VL-JEPA (Chen 2026)]]
- [[projects/road-reason|ROAD_Reason]]
- [[concepts/encoder-collapse|Encoder collapse]]
