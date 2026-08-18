---
type: paper
title: "Kimi-VL / MoonViT: Native-Resolution Vision Encoder"
aliases: ["MoonViT", "MoonViT-3D", "Kimi-VL"]
created: 2026-05-27
updated: 2026-05-27
sources:
  - "https://arxiv.org/abs/2504.07491"
  - "https://arxiv.org/abs/2602.02276"
tags: [paper, vlm, vision-encoder, native-resolution, moe, video]
status: complete
authors: "Kimi Team"
year: 2025
venue: "arXiv (Tech Report)"
arxiv: "2504.07491"
datasets_used: []
---

# Kimi-VL / MoonViT: Native-Resolution Vision Encoder

**Kimi Team (2025)** — Moonshot AI

Two papers cover MoonViT: the original **Kimi-VL Technical Report** (arXiv 2504.07491, Apr 2025) introduces MoonViT, and **Kimi K2.5** (arXiv 2602.02276, Feb 2026) extends it to MoonViT-3D for video.

## Problem

Standard VLM vision encoders (CLIP ViT, SigLIP) process images at fixed resolution (224px, 336px, etc.), requiring either resizing (loses detail) or complex sub-image tiling (introduces artifacts). For video, per-frame processing is token-expensive and discards temporal structure.

## MoonViT Architecture

| Property | Value |
|----------|-------|
| Base | SigLIP-SO-400M (continual pre-training) |
| Params | ~400M |
| Patch size | Standard ViT patches |
| Resolution | **Native** — variable per image, no resizing |
| Positional encoding | Interpolated SigLIP absolute PE + **2D RoPE** (height × width) |
| Packing | NaViT "patch n' pack" — variable-size images packed into 1D sequences |
| Attention | FlashAttention on variable-length packed sequences |
| Projector | Pixel shuffle (2×2 spatial compression) → 2-layer MLP → LLM embedding dim |

### NaViT Packing

Instead of padding all images to a fixed grid, MoonViT:
1. Divides each image into patches at its **native resolution**
2. Flattens patches into a 1D sequence
3. Concatenates multiple images into a single packed sequence
4. Runs FlashAttention with appropriate masking

This means a 384×384 image produces fewer tokens than a 1024×768 image — compute scales with actual content, not a fixed budget.

### 2D RoPE

Standard ViT uses absolute position embeddings tied to a fixed grid. MoonViT adds **2D Rotary Position Embeddings** across height and width dimensions, allowing the model to generalize to resolutions never seen during training. The interpolated SigLIP absolute PEs provide a coarse prior; RoPE provides fine-grained relative positioning.

### Training

| Stage | Data | Objective |
|-------|------|-----------|
| Stage 1 | 2T tokens, image-text pairs | SigLIP contrastive loss + caption generation |
| Stage 2 | 0.1T tokens | Alignment with MoE LLM |

Progressive resolution sampling: gradually increases maximum resolution during training so the model learns to handle larger inputs without distribution shock.

## MoonViT-3D (Kimi K2.5, 2026)

Extends MoonViT to video with minimal architectural change:

- **Temporal packing:** Up to 4 consecutive frames treated as a spatiotemporal volume — 2D patches from these frames are jointly flattened and packed into a single 1D sequence
- **4× temporal compression:** Lightweight temporal pooling at patch level before the MLP projector, extending feasible video length
- **Unified parameters:** Fully shared weights between image and video processing — no separate video module
- **Training:** ~15T mixed visual-text tokens in joint pretraining

### Video Benchmarks (K2.5)

| Benchmark | Score |
|-----------|-------|
| VideoMMMU | 86.6% |
| MMVU | 80.4% |
| LongVideoBench | 79.8% |
| LVBench | 75.9% |

## Kimi-VL System Architecture

MoonViT is the vision component of a larger VLM:

| Component | Spec |
|-----------|------|
| Vision encoder | MoonViT (~400M) |
| Language model | Moonlight MoE (2.8B active / 16B total) |
| Projector | Pixel shuffle + 2-layer MLP |
| Context window | 128K tokens |
| Training | 4.4T tokens across 4 stages |

### Key Kimi-VL Benchmarks

| Benchmark | Kimi-VL-A3B | GPT-4o | Notes |
|-----------|-------------|--------|-------|
| MMMU | 57.0% | — | Beats DeepSeek-VL2 (51.1%) |
| InfoVQA | **83.2%** | 80.7% | Ultra-high-res understanding |
| MathVista | **68.7%** | 63.8% | Mathematical reasoning |
| OSWorld | **8.22%** | 5.03% | Agent tasks |
| LongVideoBench | 64.5% | 66.7% | Close to GPT-4o |
| EgoSchema | **78.5%** | 72.2% | Egocentric video |

With only 2.8B active LLM parameters — far smaller than GPT-4o.

## Relevance to This Research

### Used by LocateAnything

[[papers/wang-2026-locate-anything|LocateAnything]] uses MoonViT as its vision encoder paired with Qwen2.5-3B for parallel box decoding. The native-resolution capability is critical for LocateAnything's grounding accuracy — it can process up to 2.5K resolution inputs, meaning small objects retain full detail rather than being downsampled.

### Comparison with Our Vision Encoders

| Property | MoonViT | CLIP ViT-L/14 (our exp2) | CLIP-ViP (OpenMixer) |
|----------|---------|--------------------------|----------------------|
| Resolution | Native (any) | Fixed 336px | Fixed (224/336px) |
| Params | ~400M | 304M | ~400M (video-adapted) |
| Temporal | 3D packing (K2.5) | None (per-frame) | Temporal prompting |
| Positional | 2D RoPE + absolute | Fixed absolute | Fixed absolute |
| Open-vocab | Via LLM | CLIP text encoder | CLIP text encoder |
| Detection use | LocateAnything (PBD) | Frozen-DETR (4th scale) | OpenMixer (S-OMB/T-OMB) |

### Key Insight for Approach 4

MoonViT's native resolution handling directly addresses the resolution bottleneck identified in the exp2 series (exp2c/2d at 384-448px → exp2e's fix at 800×1333). A native-resolution encoder wouldn't need the fixed-resolution trade-off — small pedestrians at native scale retain full patch coverage. However, this comes at compute cost: more patches = more tokens = longer attention.

The 2D RoPE design is also noteworthy — it decouples positional encoding from a fixed grid, meaning the same weights generalize across resolutions. Our current CLIP ViT-L/14 uses fixed absolute PEs interpolated to 336px, which limits its spatial precision at higher resolutions.

### MoonViT-3D for Video Action Detection

The temporal packing strategy (4 frames → spatiotemporal volume → shared attention) is a lightweight alternative to explicit temporal modules like OpenMixer's T-OMB or our exp2b's temporal self-attention. For ROAD-Waymo's 8-frame clips, this would pack into 2 spatiotemporal volumes — minimal overhead. Whether the implicit temporal modeling is sufficient for fine-grained action detection (vs. explicit temporal attention) is an open question.

## Related

- [[papers/wang-2026-locate-anything|LocateAnything]] — uses MoonViT as vision encoder for parallel box decoding
- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — uses CLIP ViT-L/14 as frozen vision encoder (our exp2 series)
- [[papers/bao-2025-openmixer|OpenMixer]] — uses CLIP-ViP as frozen video encoder
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — the resolution/detection problem MoonViT's native resolution helps address
- [[concepts/vlm-architectures|VLM Architectures]] — where MoonViT fits in the generative VLM pattern
