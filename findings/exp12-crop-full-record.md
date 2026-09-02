---
type: finding
title: "Crop-full sweep: new record on every head; phrase head is a tail specialist"
aliases: [exp13, exp13-crop-full, exp14, exp14-fusion]
created: 2026-09-01
updated: 2026-09-02
sources:
  - "ROAD_Reason/experiments/exp12_phrase_head/crop_full"
tags: []
status: complete
---

# Crop-full sweep: new record on every head; phrase head is a tail specialist

<!-- stub: fill in body -->

The full-coverage crop run (NCShare H200s + local A6000 finish, 2026-08-30/09-01)
produced a new best row on every learned head, and isolated what the contrastive
phrase mechanism actually does.

## The table (f-mAP %, full-candidate protocol, 36,716 val frames, IoU 0.5)

| configuration | agentness | agent | action | loc | duplex | triplet |
|---|---|---|---|---|---|---|
| baseline 3D-RetinaNet (own top-300) | 35.36 | 16.67 | 13.23 | 12.77 | 11.32 | 7.54 |
| old record: I3D feats + comp MLP | 65.36 | 35.25 | 15.92 | 19.60 | 15.52 | 9.73 |
| crop flat head (no stack) | 65.36 | 35.25 | 18.99 | 22.31 | 15.71 | 8.46 |
| crop phrase head (no stack) | 65.36 | 35.25 | 18.09 | 19.38 | 15.16 | 8.94 |
| **crop flat + comp MLP (NEW RECORD)** | 65.36 | 35.25 | **18.99** | **22.31** | **17.94** | **11.13** |
| crop phrase + comp MLP | 65.36 | 35.25 | 18.09 | 19.38 | 17.75 | 10.96 |

Probe-trained-head sanity rows (flat 16.62/18.55/13.29/6.63, phrase 14.74/16.07/12.29/7.08)
confirmed the probe within 0.3 everywhere before the full rows landed.

## What it proves

1. **Substrate**: crop-resolution InternVideo2 beats I3D features on every head at full
   training (the probe-era location deficit was an undertraining artifact; see the
   corrected 09-01 addendum in [[findings/exp12-phrase-head-attribution]]).
2. **Engine**: the stacked composition MLP transfers to the new substrate and adds
   +2.2 duplex / +2.7 triplet over its own bare head. Same OOF protocol as
   [[methods/stacked-composition-mlp]], input widened to 49+1024.
3. **Mechanism**: at the RAW HEAD level on identical features, the phrase head wins
   37/49 rarest triplets (+1.90 mean AP) and loses the 37 common ones (-1.41): a tail
   specialist by construction. The stack equalizes aggregates (ties within 0.2) and
   absorbs the specialization: stack-level rare-49 is a coin flip (23/49, -0.24).

## Vocabulary provenance ablation (2026-09-01)

The hand-authored phrases were replaced by a deterministic generator
(gen_phrases.py: 4 template rules over a documented glossary in ROAD-paper
nomenclature; 135/183 phrases differ) and the crop phrase head retrained on
identical features: 17.80 / 20.20 / 14.92 / 8.93 vs the original 18.09 / 19.38 /
15.16 / 8.94. Every head within 0.3 except location (+0.8). The mechanism does not
depend on hand wording; the vocabulary is now auditable. Traffic lights improve
slightly under behavioral wording (Red 59.9, Green 40.8) but Amber stays collapsed
(6.25): near-synonym collinearity survives rewording, which strengthens the
class-wise fusion motivation (Amber wants free weights).

## Principled tail definition (2026-09-02)

The original "49 rarest" split was ad hoc and ranked by val GT counts. The
principled replacement, prompted by Brandon's std-deviation question: rank
by **train-split instance counts** (v1.1 annotations, `all_triplet_labels`
name-remapped to the 86-class benchmark vocab; every class has >= 61 train
boxes, max 225,813) and cut in **log space**, where the power-law counts
are roughly symmetric: log10 counts have mean 3.81, std 0.73.

**Tail = classes below the geometric mean of train counts (z < 0, i.e.
< 6,434 train boxes) -> 47 of 86 classes.** This nearly reproduces the
ad-hoc split (46 of the old 49 overlap), so all previously booked
tail-specialist claims carry over unchanged.

Mean tail AP by stage (train-ranked membership, fixed class set per column):

| stage | triplet (86) | z=0 tail (47 cls) | z=-0.5 (28 cls) | z=-1 (16 cls) |
|---|---|---|---|---|
| 0 baseline | 7.54 | 4.14 | 2.88 | 0.55 |
| 1 copied scores | 6.81 | 3.43 | 2.19 | 0.50 |
| 2 RoIAlign head | 6.65 | 2.60 | 1.47 | 0.54 |
| 3 + comp MLP | 9.73 | 4.53 | 3.31 | 1.04 |
| 4 phrase head (raw) | 8.94 | 5.71 | **4.63** | 1.65 |
| 5 record (flat+MLP) | 11.13 | **5.72** | 4.25 | **2.04** |
| flat head, same crops | — | 3.52 | 2.73 | 0.52 |

Three claims this grounds:

1. **Same features, only the classifier weights differ:** phrase 5.71 vs
   flat 3.52 on the z=0 tail (+2.19). The language-weight advantage is not
   a substrate effect.
2. **At z=-0.5 the contrastive head alone beats the full record
   configuration** (4.63 vs 4.25) with a single trained projection versus
   trained head + MLP. The deeper the tail, the more language matters.
3. **Dose-response:** the phrase-over-flat gap grows with rarity
   (+2.19 at z=0, +1.90 at -0.5, +1.13 at -1 in a shrinking-AP regime) —
   consistent with rare classes borrowing strength through wording, not
   with a one-cutoff coincidence.

Caveat: the z=-1 column (16 classes, APs near floor) is high-variance;
don't build claims on it. Slide-safe framing: z=0 as the definition,
z=-0.5 as the exhibit.

## The evidence-motivated next rung

The stack currently averages away the phrase head's tail specialization, and the
mechanism is structural (Brandon 2026-09-01): the MLP consumes only PRIMITIVE sigmoids
plus the feature and REPLACES the composition columns, so the phrase head's superior
triplet scores (where every rare triplet has its own phrase) are discarded by
construction, while its weaker common-class primitives are kept. The architecture is
accidentally rigged against the mechanism.

Refined exp14 design: feed the phrase head's ungated duplex+triplet sigmoids into the
composition MLP as additional inputs (49 primitives + 135 phrase compositions + 1024
feature = 1208-d), letting the MLP learn where to trust the language prior. The 37/49
raw-head tail dominance plus the discard-by-construction analysis is the direct
evidence.

## Exp14 result (2026-09-02): fusion takes the tail crown at every cutoff

**Ledger name: Exp14.** Ran same-day as designed: `crop_full/train_comp_mlp_fusion.py`
(four throwaway fold heads, flat + phrase per fold, same video-level 2-fold OOF, same
focal/alphas/seed/10-epoch recipe as the record MLP; only variable = the extra 135
phrase composition inputs), evaluated via `eval_comb.py --phrase-ckpt` (fusion
assembly asserts in_dim 1208 and checkpoint lineage). Artifacts:
`comp_mlp_fusion_crop_full.pt`, `results_crop_full_fusion_mlp.json`,
`chain_fusion.log`.

| config | action | loc | duplex | triplet (86) | tail z=0 (47) | z=-0.5 (28) | z=-1 (16) |
|---|---|---|---|---|---|---|---|
| record flat+MLP | 18.99 | 22.31 | **17.94** | **11.13** | 5.72 | 4.25 | 2.04 |
| phrase raw | 18.09 | 19.38 | 15.16 | 8.94 | 5.71 | 4.63 | 1.65 |
| **exp14 fusion** | 18.99 | 22.31 | 17.76 | 10.98 | **5.99** | **4.73** | **2.59** |

- Action/loc bit-identical to the record by construction (MLP touches only
  composition columns) — sanity confirmed.
- **Best tail at every cutoff, beating both parents**: +0.27 over the record at z=0,
  +0.48 at z=-0.5 (also beating the phrase head's own 4.63), +0.55 (+27% relative)
  on the 16 rarest. The MLP composes cosine evidence with features better than
  either source alone; the discard-by-construction defect is fixed.
- **Dose-response strengthened**: fusion's edge over the record grows monotonically
  with rarity — third independent data point for language-helps-where-data-starves.
- **Cost**: -0.18 duplex, -0.15 triplet on the 86-class averages. The record row
  keeps the overall crown; the fusion is the long-tail configuration. Given the
  thesis point is long-tail accuracy, exp14 is the headline candidate with the
  record as the averages-optimized ablation.

## Provenance

**Ledger name: Exp13** (Brandon 2026-09-01). Files live in exp12_phrase_head/crop_full/ because the sweep reuses exp12's phrase embeddings, trainers and eval path; exp13 is the ledger identity of the full-coverage crop sweep and its six rows.

Caches: crop_feats_train.pkl (115,602 frames, 8 H200 shards), crop_feats_val.pkl
(37,931 frames, 3 H200 shards + local A6000 finish after /work purge). Heads and
stacks: crop_full/ scripts, OOF by video, focal + exp6 alphas throughout. Results:
crop_full/results_crop_*.json.

## Related

- [[findings/exp12-phrase-head-attribution]] — the attribution grid this completes
- [[methods/stacked-composition-mlp]] — the engine
- [[findings/hybrid-evolution-narrative]] — where this row now sits as the endpoint
- [[directions/thesis-proposal-fall-2026]] — mechanism/engine/substrate framing
