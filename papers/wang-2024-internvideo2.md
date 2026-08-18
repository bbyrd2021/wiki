---
type: paper
title: "InternVideo2: Scaling Foundation Models for Multimodal Video Understanding"
aliases: ["InternVideo2", "IV2"]
created: 2026-08-18
updated: 2026-08-18
sources:
  - "wiki/raw/InternVideo2.pdf"
tags: [paper, video-foundation-model, contrastive, masked-modeling, text-aligned, backbone-candidate, approach-8]
status: complete
authors: "Wang et al."
year: 2024
venue: "ECCV"
arxiv: "2403.15377"
datasets_used: []
---

# InternVideo2 — Scaling Foundation Models for Multimodal Video Understanding

Wang, Li, Li, Yu, He, et al. (OpenGVLab / Shanghai AI Lab, Nanjing, SIAT). **arXiv:2403.15377, v4 Aug 2024.**

A family of video foundation models (ViFM) up to **6B params** built by a **three-stage progressive scheme** that unifies the era's three video-representation recipes in one encoder. SOTA on 60+ video/audio tasks at publication; **Dr. Moradi's pointer (2026-08-18) as the text-aligned encoder for the contrastive phrase-classification idea, and a fallback backbone candidate.**

## The three stages

| Stage | Objective | Mechanism |
|---|---|---|
| 1 — Spatiotemporal perception | masked token reconstruction (distillation) | mask 80%/frame; align unmasked tokens by MSE to two frozen teachers — **InternVL-6B** (semantic, last 6 layers) and **[[papers/tong-2022-videomae\|VideoMAEv2-g]]** (motion, last 4 layers); projection layers dropped after |
| 2 — Multimodal alignment | crossmodal contrastive + matching + MLM | video–audio–speech–text; BERT-Large text encoder (19 layers + 5 cross-attn as multimodal decoder), BEATs audio encoder (90M); masked-then-unmasked two-step post-pretraining |
| 3 — Open-ended dialogue | next-token prediction | QFormer + LLM (BLIP-style), LoRA on the LLM; HD post-training with sub-video tiling |

Video encoder: ViT (up to 6B) with attention pooling, 8-frame sparse sampling, 14×14 spatial downsample, 3D position embeddings. **InternVideo2_clip** = stage-2 model post-trained to keep only video+text encoders with contrastive loss — the CLIP-style variant relevant to us.

## Data machinery (half the contribution)

402M entries: K-Mash (2M unlabeled videos), InternVid2 (50M video-audio-speech captions), 300M image-text. Clips cut by **AutoShot** semantic boundary detection (not pixel-diff scene cuts); captions from **VidCap** — separate video/audio/speech captioners fused by an LLM. Paper's thesis: temporal segmentation quality and caption fusion drive alignment quality.

## Results (headline)

- **End-to-end fine-tune:** K400 **92.1** / K600 91.9 / K700 85.9 top-1 at 16×224 (prior SOTA needed 576px or ensembles); SSv2 77.5; MiT 51.2; ANet 95.9; HACS 97.0.
- **Attentive probe:** K400 88.8 — above ViT-22B and CoCa-g; matches/exceeds V-JEPA-H and VideoPrism-g on temporal SSv2.
- **Linear probe:** K400 84.2 / SSv2 56.7 — stage 2 adds +2.2/+8.9 over stage 1, i.e., **the multimodal alignment stage measurably improves the frozen features**.
- **Zero-shot (clip variant):** K400 72.7; strong across UCF/HMDB/MiT/Charades.
- Temporal action localization (feature-based, frozen layer-7 features + ActionFormer): highest mAP on THUMOS14/ANet/HACS/FineAction.

## Why it matters here

1. **The contrastive-phrase encoder candidate.** Moradi's suggestion (2026-08-18 email) — classify RoI features by proximity to text embeddings of composed triplet phrases — needs a video/text pair with pre-aligned spaces. InternVideo2_clip is purpose-built for that: its stage-2 contrastive space is exactly a video-feature↔text-embedding shared space, and the linear-probe deltas show the alignment survives freezing. Alternative to SigLIP (current exp6 rationale encoder, image-text only, no video pretraining).
2. **Fallback backbone.** If exp9-class approaches stall, this is the "InternVideo2 I think is newer" option vs [[papers/tong-2022-videomae|VideoMAE]] — but note the 6B encoder is a different compute regime from our 3D-RetinaNet (~50M) or even Qwen's ViT (~670M); the 1B variant or frozen layer-7 features (their TAL recipe) are the workstation-realistic entry points.
3. **Design lesson for the temporal question:** stage 1 explicitly distills *both* a semantic teacher and a motion teacher — the paper's evidence that neither image-semantic nor motion-only pretraining suffices alone mirrors our action-head diagnosis on [[datasets/road-plusplus|ROAD-Waymo]].

## Related

- [[directions/vlm-reasoning-layer|Approach 8]] — the contrastive-head idea (Moradi 2026-08-18) and fallback-backbone context
- [[papers/tong-2022-videomae|VideoMAE]] — stage-1 motion teacher lineage (V2-giant); the other Moradi backbone pointer
- [[papers/chen-2026-vl-jepa|VL-JEPA]] — the JEPA-family alternative to token-level alignment; compares against VideoMAEv2/InternVideo2-era encoders
- [[papers/kimi-2025-moonvit|MoonViT]] — native-resolution encoder alternative
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — why any of these enter as feature/embedding sources, not detectors
