---
type: paper
title: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
aliases: ["VL-JEPA", "Chen 2026 VL-JEPA", "JEPA for VLM"]
created: 2026-05-11
updated: 2026-05-11
sources:
  - "wiki/raw/VL-JEPA.pdf"
tags: [paper, jepa, vlm, embedding-prediction, v-jepa, world-model, selective-decoding, infonce, latent-space]
status: complete
authors: "Chen et al."
year: 2026
venue: "arXiv preprint, Feb 2026"
arxiv: "2512.10942"
datasets_used: []
---

# VL-JEPA — Joint Embedding Predictive Architecture for Vision-language

Chen, Shukor, Moutakanni, Chung, Yu, Kasarla, Bang, Bolourchi, **LeCun**, Fung. [arXiv:2512.10942v2](https://arxiv.org/abs/2512.10942) (Feb 2 2026). Meta FAIR / HKUST / Sorbonne / NYU.

The first JEPA-style VLM: instead of autoregressively generating answer **tokens**, VL-JEPA predicts a continuous **embedding** of the target answer in a shared latent space. A small text decoder (`Y-Decoder`) is invoked only when text is actually needed. On matched training data the 1.6B model beats CLIP / SigLIP2 / Perception-Encoder on 8 classification + 8 retrieval benchmarks and matches Qwen-VL / InstructBLIP on four VQA datasets — at **half the trainable params** of a comparable VLM and ~3.7× lower end-to-end inference latency for streaming use.

---

## Why This Matters for the Lab

VL-JEPA is the same backbone family ([[directions/jepa-intent-head|V-JEPA 2]]) the lab is already considering for the ROAD_Reason Approach 5 intent head, now augmented with a language-aware predictor. Three direct implications:

1. **Alternative to token-space VLMs for Approach 3/4.** [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task]] and [[methods/multimodal-causal-driving|MCDM]] are both token-generative. VL-JEPA shows latent-space prediction is competitive on VQA / classification / retrieval at half the trainable parameters — relevant if the user wants a leaner backbone for [[projects/road-reason|ROAD_Reason]].
2. **Motion-centric gains.** VL-JEPA particularly shines on SSv2, EK-100, EgoExo4D — motion-heavy benchmarks. The V-JEPA 2 X-Encoder captures dynamics better than CLIP/SigLIP2. This is the same regime that matters for pedestrian crossing intent.
3. **Selective decoding for streaming.** A ~2.85× reduction in decoder invocations on EgoExo4D while maintaining CIDEr — interesting primitive for the [[projects/auto-drive-perception|AutoDrive ROS pipeline]] where text-emitting modules (captions, alerts) should fire only on semantic change.

---

## Architecture

Four components (Fig. 1, Fig. 2):

| Module | Init | Params | Trainable? |
|---|---|---:|:---:|
| **X-Encoder** (vision → S_V) | V-JEPA 2 ViT-L | 304M | ❌ frozen |
| **Predictor** ((S_V, X_Q) → Ŝ_Y) | Llama-3.2-1B layers 8–16 | 490M | ✅ |
| **Y-Encoder** (Y → S_Y) | EmbeddingGemma-300M | 300M | ✅ (LR mult ×0.05) |
| **Y-Decoder** (Ŝ_Y → Ŷ) | EmbeddingGemma decoder | — | inference-only readout |

Total: **1.6B params** (~0.5B trainable predictor + Y-encoder; X-encoder frozen).

```
X_V (video/frames) ──→ X-Encoder (V-JEPA 2, frozen) ──→ S_V
                                                          │
X_Q (text query) ──→ tokenize+embed (Llama-3) ────────────┤
                                                          ↓
                                                       Predictor
                                                    (Llama-3 layers 8-16)
                                                          ↓
                                                         Ŝ_Y  ←─── L_VL-JEPA = D(Ŝ_Y, S_Y)
                                                                              ↑
                                                            Y (target text) ──→ Y-Encoder ──→ S_Y
                                                                              (EmbeddingGemma-300M)
```

Predictor + Y-Encoder are trained jointly with bi-directional **InfoNCE** in a shared 1536-dim embedding space. The causal mask is disabled in the predictor so visual and query tokens can attend jointly. Llama-3 init beats random init for VQA but is slightly worse for retrieval/classification (Tab. 7d).

### Why predict embeddings, not tokens?

The paper makes the cleanest argument I've seen for JEPA at the VLM level. Two valid answers to *"What will happen if I flip this light switch?"* — "the lamp turns off" vs "the room goes dark" — share **no overlapping tokens**, so in raw token space their distributions are orthogonal point-masses. The training signal forces the VLM to model both as separate disjoint high-density regions in sparse token space. In embedding space, both can sit near a single semantic point, giving a unimodal, simpler distribution.

This is the same anti-modality argument LeCun makes for I-JEPA / V-JEPA, now applied to language targets.

---

## Training

Two stages:

| Stage | Data | Steps | Batch | Resources |
|---|---|---|---|---|
| **Pretraining → VL-JEPA_BASE** | Datacomp + YFCC-100M (image), Action100M (video) | 70k iters (60k img + 10k vid) | 24k effective (img), large for vid | 24 nodes × 8× H200, 4 weeks |
| **SFT → VL-JEPA_SFT** | PLM mixture: 25M VQA + 2.8M caption + 1.8M class + downsampled pretrain | 83k steps | 3,072 | 2.5 days on 24 nodes |

InfoNCE alignment + uniformity loss; cosine annealing. After image-only pretraining (1 frame/sample), continues with 8 frames/sample for 60k iters, 32 frames for last 10k. The Y-Encoder uses an **0.05× LR multiplier** — surprisingly important; ×1.0 multiplier degrades retrieval by 1.4 mAP (Tab. 7b).

### Ablation highlights (Tab. 7)

- Drop pretraining stage → −21.7 acc / −17.3 R@1 / −3.6 VQA. Pretraining is decisive.
- InfoNCE beats Cosine / L1 / L2 losses for representation alignment (handles anti-collapse without needing freeze/EMA tricks).
- Bi-directional attention in predictor adds **+1.9 VQA** vs causal mask.
- Llama-3 init helps VQA (+1.9) but slightly hurts classification (−2.7) — vision-text alignment benefits from random init in early epochs.
- The Y-Encoder choice matters: vision-aligned text encoders (PE-Core, SigLIP2) substantially outperform pure text encoders (Qwen3-Embedding) on classification/retrieval; on average EmbeddingGemma-300M is the chosen default but PE-Core-G (539M, vision-aligned) is the best on classification (33.9% vs 19.5%).

---

## Headline Results

### Strict zero-shot vision-text (Tab. 1)

| Model | # params | Samples seen | Avg Class top-1 (8 sets) | Avg Retrieval R@1 (8 sets) |
|---|---:|---:|---:|---:|
| CLIP ViT-L | 124M | 12.8B | 25.4 | 31.0 |
| SigLIP2 ViT-L | 882M | 40B | 38.6 | 41.6 |
| PE-Core ViT-L | 671M | 58B | 42.9 | 46.8 |
| **VL-JEPA_BASE** ViT-L | **1.6B** | **3.3B** | **52.5** | **63.7** |
| **VL-JEPA_SFT** ViT-L | 1.6B | 3.6B | 75.4 (vs 77.5 prev SoTA) | 63.8 (vs 62.8) |

VL-JEPA_BASE wins on average despite ~4× less training data than PE-Core. Particularly strong on **SSv2 (52.5 vs 9.3 PE-Core), EK-100, EgoExo4D, COIN-Step/Task, CrossTask** — all motion- and procedure-centric. Weaker on appearance-only Kinetics-400.

### VQA (Tab. 2 — VL-JEPA_SFT, 1.6B vs much larger baselines)

| Benchmark | VL-JEPA_SFT | Best competing |
|---|---:|---:|
| GQA | 61.5 | InternVL-Chat-13B 66.6 |
| TallyQA | 69.9 | PaliGemma-3B 76.8 |
| POPE | 85.7 | Qwen2-VL-7B 87.5 |
| POPEv2 | 86.3 | Qwen2-VL-7B 88.8 |

Competitive across the board at a tenth the parameter count.

### WorldPrediction-WM (Tab. 3)

Identify the action that explains a transition between two world-state frames. VL-JEPA_SFT: **65.7%**, beating GPT-4o (53.6), Claude-3.5-sonnet (55.6), Gemini-2.0 (52.0), and InternVL2.5-78B (50.3). New SOTA.

### Inference cost (Fig. 3, right)

| Stage | VLM (1B LLM) | VL-JEPA (0.5B predictor) |
|---|---:|---:|
| Query embedding | — | 16 ms |
| Encoder + (LLM \| predictor) | 330 ms | 126 ms |
| Text decoding | 203 ms (every prompt) | only when decoded |
| **End-to-end (prompted)** | ~533 ms | ~142 ms |

Per-prompt encoder/predictor pass is repeated 100× over the same video at zero cost in the JEPA pipeline because the visual encoding is computed once and reused.

### Selective decoding (Fig. 4)

On EgoExo4D, agglomerative temporal clustering in embedding space (Ward distance on Ŝ_Y) yields decoding points that **Pareto-dominate uniform sampling**. At 0.35 Hz (∼2.85 s interval) it matches uniform decoding at 1 Hz — a 2.85× reduction.

---

## Connections to Existing Wiki

### Direct architectural relatives

- [[directions/jepa-intent-head|V-JEPA 2 + Intent Head (Approach 5)]] — uses the same V-JEPA 2 ViT-L X-Encoder. VL-JEPA shows that the V-JEPA 2 backbone is competitive at VLM-scale tasks, not just action anticipation.
- [[methods/multimodal-causal-driving|MCDM (Approach 4)]] — token-generative pipeline with CLIP-ViP + OpenMixer + DSDAG. VL-JEPA is the latent-space alternative: replace the LLM head with embedding prediction; selective decoding for live alerts.
- [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task (Approach 3)]] — direct generative baseline. VL-JEPA outperforms a matched-data token-generative VLM at half the trainable parameters in the controlled comparison (§4.5).

### Concepts

- [[concepts/vlm-architectures|VLM Architectures]] — VL-JEPA is a third pattern beyond contrastive alignment (CLIP) and generative LLM-coupling (LLaVA): **embedding prediction with optional decode**. The contrastive InfoNCE loss is the same family as Pattern 1 but the target is a *predicted answer embedding*, not an aligned input pair.

### Comparison baselines mentioned

- CLIP ([[papers/perez-2018-film|FiLM]]-era contrastive)
- SigLIP2, Perception-Encoder (PE-Core) — current state-of-the-art contrastive baselines that VL-JEPA beats on average
- InstructBLIP, Qwen-VL — generative VLM baselines VL-JEPA matches on VQA

---

## Limitations (per the paper)

- Pretraining data is only 3.6B samples — less than 1/20 of PE-Core's 86B. Authors acknowledge that scaling the predictor and Y-Encoder is left to future work.
- Worse than V-JEPA 2 (g-384px) on Epic-Kitchens-100 1-second anticipation. The smaller L-256px encoder is the bottleneck.
- SFT data quality dominates: the WorldPrediction SOTA depends on the PLM-curated 25M VQA mixture.
- No exploration of advanced anti-collapse losses (VICReg, SIGReg) — InfoNCE only.

---

## Practical Takeaways for Lab Work

1. **For Approach 5 intent head**, V-JEPA 2 ViT-L (frozen) is now validated at the VLM scale — building on top of it is safer than I had assumed in [[directions/jepa-intent-head]].
2. **For Approach 4 (MCDM)** the JEPA framing offers an alternative *prediction head*: instead of feeding the reasoning embedding `r` into a language decoder, predict the answer embedding directly and decode only when text is needed (alerts, captions). Could compose with the existing OpenMixer + DSDAG front-end.
3. **For the ROS pipeline** the selective-decoding primitive — segment an embedding stream by Ward distance, decode only at midpoints — generalises beyond captioning. Could trigger downstream classifier dispatch only when the scene actually changes.
4. **For SLURP-style hierarchical SLU** ([[methods/wavlm-hier|WavLM-Hier]]): VL-JEPA's bi-directional InfoNCE in a shared 1536-dim space is the cleanest single-objective formulation of "predict the labels' embedding, not the labels' tokens" — relevant to imagining future single-objective alternatives to the hierarchical CE+CE+CE stack.

---

## Related

- [[directions/jepa-intent-head|V-JEPA 2 + Intent Head (Approach 5)]]
- [[methods/multimodal-causal-driving|MCDM Architecture (Approach 4)]]
- [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task (Approach 3)]]
- [[concepts/vlm-architectures|VLM Architectures]]
