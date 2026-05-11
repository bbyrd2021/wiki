---
type: method
title: "WavLM-Hier — Hierarchical SLU on WavLM-Large"
aliases: ["WavHeir", "WavLM-Hier", "wavlm_large_hier"]
created: 2026-05-08
updated: 2026-05-08
sources:
  - "SLURP/contrib/run_whisper_hier.py"
  - "SLURP/contrib/README.md"
  - "SLURP/contrib/build_ontology.py"
  - "SLURP/logs/wavlm_large_hier_v3/model_config.json"
tags: [method, slurp, wavlm, hierarchical, slu]
status: complete
---

# WavLM-Hier

Hierarchical SLU model for SLURP combining a WavLM-Large encoder with attention pooling, scenario→action conditioning, curriculum teacher forcing, and inference-time ontology masking. Best result on SLURP test set: **scenario F1w 0.877 / action F1w 0.833 / intent acc 0.820** ([[findings/slurp-wavlm-hier-results|full results]]).

The script filename is `contrib/run_whisper_hier.py` because Whisper was the original encoder choice; the architecture is encoder-agnostic and currently uses `microsoft/wavlm-large` via the `ENCODER_NAME` constant.

## Five-component novelty stack

Each component can be ablated independently. The flat ablation (#3 + #4 + #5 disabled) is run as `wavlm_large_flat_ablation`.

| # | Component | What it does |
|---|---|---|
| 1 | **WavLM-Large encoder** | Replaces the frozen Wav2Vec2 that collapsed in [[findings/slurp-collapse-e1|E1]]. Pretrained with masked unit prediction + denoising on noisy speech; SUPERB Intent Classification SOTA |
| 2 | **Attention pooling** | Learned temporal query over encoder frames (vs. mean pool). 1 query × 1024-dim |
| 3 | **Hierarchical conditioning** | Action head receives the full scenario soft-probability vector (18-dim) concatenated to pooled audio features before the action classifier |
| 4 | **Curriculum teacher forcing** | Scenario probabilities fed to the action head are linearly interpolated between ground-truth one-hot (epoch 1) and predicted softmax (epoch 15). `teacher_p` decays 1.0 → 0.0 over training |
| 5 | **Ontology masking** | At inference, scenario→action pairs absent in training data (only 60 of 18×46 = 828 pairs are valid) are masked to −∞ before action softmax |

## Architecture

```
audio (waveform, max 10s)
  ↓
WavLM-Large encoder (315.5M params, fine-tuned)
  ↓ last_hidden_state [B, T, 1024]
attention pooling (1 query, 1024-dim)
  ↓ [B, 1024]
  ├──→ scenario classifier (1024 → 18) → P(scenario | audio)
  └──→ [audio_pool ⊕ teacher_blend(scenario_probs)] → action classifier (1042 → 46) → P(action | audio, scenario)
       └─ (inference) ontology mask: −∞ where pair invalid
```

| Param block | Count |
|---|---|
| Encoder (WavLM-Large) | 315,453,120 |
| Heads (pooling + scenario + action + projections) | 1,094,745 |
| **Total** | **316,547,865** |

## Training configuration (v3, best run)

| Hyperparameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 16 |
| LR (heads) | 3e-4 |
| LR (encoder) | 5e-6 |
| Warmup fraction | 10% |
| LR schedule | Linear warmup → cosine annealing |
| Gradient clip | max_norm 1.0 |
| Weight cap (class weights) | [0.1, 10.0] |
| Loss | Class-weighted CE (sum of scenario + action losses) |
| Teacher forcing | `teacher_p = 1.0 − (epoch / num_epochs)` |
| Seed | 42 |
| Hardware | 1× A6000, ~14 hours |

Class weights are inverse-frequency on the training set, capped to [0.1, 10.0] to prevent any single rare class from dominating gradient signal.

## Ontology

`build_ontology.py` extracts the 60 valid scenario→action pairs from the training set (out of 18 × 46 = 828 possible). Stored as a binary mask `[18, 46]` in `valid_mask.npy` and applied to action logits at inference:

```python
action_logits = action_logits.masked_fill(~valid_mask[scenario_pred], float("-inf"))
```

Training does not use the mask — it learns from supervised signal — so the model still allocates capacity to invalid pairs but never predicts them at test time.

## Variants run

| Run | Encoder | Hier | Test Sc F1w | Test Act F1w | Intent Acc | Notes |
|---|---|---|---|---|---|---|
| `whisper_hier` | WavLM-base-plus | yes | 0.816 | 0.734 | 0.711 | Smaller encoder ablation (run via the same script) |
| `wavlm_large_hier` | WavLM-Large | yes | 0.852 | 0.788 | 0.772 | First Large run (v1) |
| `wavlm_large_hier_v2` | WavLM-Large | yes | — | — | — | Tuning run, no final test eval |
| `wavlm_large_hier_v3` | WavLM-Large | yes | **0.877** | **0.833** | **0.820** | Best |
| `wavlm_large_flat_ablation` | WavLM-Large | **no** | 0.874 | 0.823 | 0.796 | Hierarchy ablation: removes #3 + #4 + #5 |

The hierarchy contributes +0.003 scenario F1w / +0.010 action F1w / +0.024 intent acc over flat WavLM-Large. Marginal on the per-task metrics; meaningful on the joint intent metric where scenario and action must both be correct.

## Encoder swap

The script supports any HuggingFace audio encoder exposing `last_hidden_state` via the standard `(input_values, attention_mask)` interface. Tested:

```python
ENCODER_NAME = "microsoft/wavlm-large"        # 315M, current best
ENCODER_NAME = "microsoft/wavlm-base-plus"    # 94M, smaller
ENCODER_NAME = "openai/whisper-medium"        # original plan, not run
ENCODER_NAME = "facebook/hubert-large-ll60k"  # untested
```

## Related

- [[findings/slurp-wavlm-hier-results|WavLM-Hier full results]] — per-class breakdown, ablations, dev curves
- [[findings/slurp-collapse-e1|SLURP E1 Collapse]] — failure baseline this method recovers from
- [[comparisons/slurp-audio-vs-text-oracle|Audio vs Text Oracle]] — why audio beats RoBERTa on gold transcripts
- [[concepts/encoder-collapse|Encoder Collapse]] — generalized failure pattern
- [[projects/slurp|SLURP project]]
