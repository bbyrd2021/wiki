---
type: concept
title: "Text encoder choice for the contrastive phrase head: alignment beats quality"
aliases: []
created: 2026-09-01
updated: 2026-09-01
sources: []
tags: []
status: complete
---

# Text encoder choice for the contrastive phrase head: alignment beats quality

<!-- stub: fill in body -->

Literature grounding for the phrase head's text-encoder choice, assembled for the
proposal defense (2026-09-01).

## The core principle: alignment beats standalone quality

For a frozen visual encoder with a cosine classifier, the operative variable is not
"which text encoder is best" but "which text encoder shares an embedding space with
the visual features." CLIP (Radford et al., ICML 2021) established the paired-tower
paradigm: text and image encoders are only meaningful together. LiT (Zhai et al.,
CVPR 2022) showed the asymmetric recipe: lock the image tower, tune a text tower
INTO its space. A stronger-but-unpaired text encoder (BERT, Sentence-T5) produces
embeddings the visual space cannot read without new alignment training.

## What open-vocabulary detection actually uses

- ViLD (Gu et al., ICLR 2022), Detic (Zhou et al., ECCV 2022), OWL-ViT (Minderer
  et al., ECCV 2022), YOLO-World (Cheng et al., CVPR 2024): CLIP text tower as
  classifier weights, exactly our formulation A.
- GLIP (Li et al., CVPR 2022) and Grounding DINO (Liu et al., 2023) use BERT, but
  only because they TRAIN deep vision-language fusion jointly; not applicable to a
  frozen-feature, linear-probe regime.
- Current OVD surveys (2025) describe the standard head as cosine similarity against
  text embeddings from the paired tower.

## Phrase wording literature

- CLIP's own prompt ensembling (80 templates) shows wording matters and averaging
  helps.
- CuPL (Pratt et al., ICCV 2023) and DCLIP (Menon and Vondrick, ICLR 2023): LLM-
  generated class descriptions are a published, citable methodology; our v1 phrases
  have precedent.
- WaffleCLIP (Roth et al., ICCV 2023): the skeptic's control; random descriptors
  sometimes match LLM ones, so wording claims need ablation, which our
  deterministic-vocabulary v2 run provides ([[findings/exp12-crop-full-record]]).

## For our stack specifically

InternVideo2 CLIP_S ships a text tower trained jointly with its visual encoder
(Wang et al., 2024); that pairing is the literature-supported choice for the frozen
regime and is what the phrase head uses. Upgrade paths, in evidence order:
1. Swap encoder PAIRS, never towers: the 1B InternVideo2 with its own text tower
   (September ladder), or a SigLIP2 pair (Tschannen et al., 2025: sigmoid loss +
   multi-task pretraining, strong dense-prediction transfer).
2. LiT-style adapter: if a richer text encoder is ever wanted, train a projection
   from it into the frozen visual space rather than swapping raw embeddings.
3. Per-class prompt ensembling over the deterministic templates (cheap, CLIP-paper
   endorsed).

## Defense one-liner

"The text encoder is not a free choice; it must be the tower trained jointly with
the visual encoder. We use InternVideo2's own text tower, which is the same design
choice as ViLD, Detic, OWL-ViT and YOLO-World, and we ablate vocabulary wording
with a deterministic generator."

## Related

- [[findings/exp12-phrase-head-attribution]] — the mechanism evidence
- [[findings/exp12-crop-full-record]] — where the vocabulary ablation lands
- [[directions/thesis-proposal-fall-2026]] — the 1B encoder rung
