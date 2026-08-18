---
type: paper
title: "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
aliases: ["VideoMAE", "video masked autoencoder", "tube masking"]
created: 2026-08-18
updated: 2026-08-18
sources:
  - "wiki/raw/VideoMAE.pdf"
tags: [paper, video-foundation-model, masked-modeling, self-supervised, data-efficient, backbone-candidate]
status: complete
authors: "Tong et al."
year: 2022
venue: "NeurIPS"
arxiv: "2203.12602"
datasets_used: []
---

# VideoMAE — Masked Autoencoders are Data-Efficient Learners for SSVP

Tong, Song, Wang, Wang (Nanjing / Tencent AI Lab / Shanghai AI Lab). **NeurIPS 2022, arXiv:2203.12602.**

The video extension of ImageMAE, and the first masked-video pretraining framework on **plain ViT backbones**. **Note: this is V1** — Dr. Moradi's 2026-08-18 pointer named "VideoMAE V2" (Wang et al. 2023, arXiv:2303.16727, the billion-param dual-masking scale-up whose giant model teaches [[papers/wang-2024-internvideo2|InternVideo2]] stage 1). V1 is where the core mechanism lives; V2 is the scaled instance.

## Core mechanism — tube masking at extreme ratios

Video's two temporal properties break naive MAE: **redundancy** (semantics vary slowly → 75% masking is too easy) and **correlation** (a masked patch's content usually survives in adjacent frames → reconstruction leaks into low-level copying). The fixes:

- **Extreme masking ratio, 90–95%** (vs ImageMAE's 75%) — restores task difficulty and cuts encoder cost (only ~10% of tokens enter the encoder).
- **Tube masking** — one masking map shared across all frames, so a masked cube is masked in *every* frame; no temporal neighbor to copy from, forcing high-level spatiotemporal reasoning.
- Vanilla ViT with **joint space-time attention**, cube embedding 2×16×16, strided temporal sampling (τ=4 Kinetics / 2 SSv2), MSE on masked cubes, asymmetric encoder–decoder (4-block decoder optimal).

## Results and findings

- Vanilla ViT, **no extra data**: K400 87.4 / SSv2 75.4 / UCF101 91.3 / HMDB51 62.6.
- **Data-efficient:** trains successfully on 3.5k videos (HMDB) — masked reconstruction as supervision works at dataset scales where contrastive methods starve.
- **Data quality > quantity for SSVP; domain shift matters** — pretraining on the target dataset beats transferring from a larger foreign one (K400→SSv2 transfer loses to direct SSv2 pretraining).
- Ablations: 90% tube > 75% tube > random > frame masking; MSE > L1/smooth-L1; from-scratch video ViT fails without it (32.6 SSv2) while VideoMAE reaches 69.6 (ViT-B).
- Transfers to **AVA spatiotemporal action detection** — the closest task shape to ROAD-Waymo in its downstream suite.

## Why it matters here

1. **The fallback-backbone family** (Moradi 2026-08-18: "if we failed on all approaches … try things like VideoMAE V2 or similar"). The pitch for our setting is the data-efficiency finding: ROAD-Waymo's 798 videos is exactly the "small video dataset" regime V1 proved masked pretraining handles — a domain-matched SSVP pretrain on ROAD-Waymo clips themselves is the recipe the paper's own domain-shift finding recommends, vs zero-shot features from someone else's pretraining domain.
2. **Motion-aware by construction.** Tube masking forces motion reasoning — the property our action head lacks (action 15.28 all-anchor for the [[methods/3d-retinanet|I3D baseline]]; the compositional chain degrades from there). VideoMAEv2-g is used as the *motion* teacher in InternVideo2 stage 1 for exactly this reason.
3. **No text alignment** — unlike [[papers/wang-2024-internvideo2|InternVideo2]], VideoMAE features live in a pure visual space; for Moradi's contrastive-phrase head it would need an alignment stage trained from scratch. The two pointers are therefore complementary: VideoMAE(V2) = motion-strong visual features; InternVideo2 = text-aligned features.

## Related

- [[papers/wang-2024-internvideo2|InternVideo2]] — uses VideoMAEv2-g as its stage-1 motion teacher; the other Moradi backbone pointer
- [[directions/vlm-reasoning-layer|Approach 8]] — fallback-backbone context (status ledger, 2026-08-18)
- [[papers/bardes-2023-jepa|V-JEPA]] — the direct predict-in-latent-space counterpart (cites/benchmarks against masked-pixel video methods)
- [[papers/maes-2026-lewm|LeWM]] — same axis, no citation link: LeWM's own taxonomy is reconstruction-based vs reconstruction-free, and its Tab. 7 ablation finds adding a reconstruction loss does not improve downstream control — the empirical form of the pixel-reconstruction-vs-latent-prediction dispute this paper sits on the other side of
- [[concepts/encoder-collapse|Encoder Collapse]] — the frozen-feature failure mode any backbone swap must clear
