---
type: finding
title: "Crop-full sweep: new record on every head; phrase head is a tail specialist"
aliases: []
created: 2026-09-01
updated: 2026-09-01
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

## The evidence-motivated next rung

The stack currently averages away the phrase head's tail specialization. Class-wise
fusion (phrase scores for tail classes, flat scores for head classes, or both primitive
sets into one MLP) is the obvious cheap ablation: the 37/49 + opposite-sign deltas are
the direct evidence that the two heads know different things.

## Provenance

Caches: crop_feats_train.pkl (115,602 frames, 8 H200 shards), crop_feats_val.pkl
(37,931 frames, 3 H200 shards + local A6000 finish after /work purge). Heads and
stacks: crop_full/ scripts, OOF by video, focal + exp6 alphas throughout. Results:
crop_full/results_crop_*.json.

## Related

- [[findings/exp12-phrase-head-attribution]] — the attribution grid this completes
- [[methods/stacked-composition-mlp]] — the engine
- [[findings/hybrid-evolution-narrative]] — where this row now sits as the endpoint
- [[directions/thesis-proposal-fall-2026]] — mechanism/engine/substrate framing
