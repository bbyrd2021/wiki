---
type: finding
title: "Exp12 — The Phrase Head Works; Features Are the Lever"
aliases: []
created: 2026-08-27
updated: 2026-08-27
sources:
  - "ROAD_Reason/experiments/exp12_phrase_head/DESIGN.md"
tags: [finding, exp12, contrastive, phrase-head, internvideo2, road-plusplus, f-map]
status: complete
---

# Exp12 — The Phrase Head Works; Features Are the Lever

The proposed thesis contribution's first results (default run, 2026-08-26→27).
All cells on the exp11 YOLO full-candidate rows, conf-gated, one variable per
cell. Gate: the [[methods/stacked-composition-mlp|stacked-MLP]] record
(duplex 15.52 / triplet 9.73).

| cell | action | loc | duplex | triplet |
|---|---|---|---|---|
| C1: InternVideo2_CLIP_S + flat head | 12.73 | 15.36 | 7.76 | 3.81 |
| C2: InternVideo2_CLIP_S + phrase head | 13.11 | 15.63 | 9.15 | 4.86 |
| C3: I3D + phrase head | **16.07** | 19.32 | 11.23 | 6.97 |
| C4: phrase primitives + comp MLP | **16.07** | 19.32 | 15.17 | 9.53 |
| exp11 flat head (reference) | 15.92 | 19.60 | 11.13 | 6.65 |
| exp11 record (gate) | 15.92 | 19.60 | **15.52** | **9.73** |

## Five attributed findings

1. **The mechanism works (C2 vs C1, same features).** Phrase-embedding
   classification beats a free linear classifier on every head: action +0.38,
   loc +0.27, duplex +1.39 (+18%), triplet +1.05 (+27%). Signature tail
   evidence: C2 wins 68/86 triplet classes and **41/49 rare ones** (<2K
   instances); HazLit +5.1, Red +7.4, TurLft/TurRht doubled. Moradi's
   borrowing-strength hypothesis is confirmed.
2. **The encoder is the bottleneck (C1 vs exp11 flat head, same head).**
   Frozen CLIP_S (224², static ×8 replication — no motion) loses to the
   domain-trained I3D everywhere (action −3.2, loc −4.2, duplex −3.4). Named
   risk from the design, now measured. NB: the InternVideo2 path has never
   seen true temporal clips or the 1B flagship — both untested levers.
3. **Alignment quality gates the mechanism (C3 vs C2 deltas).** With learned
   (not pretrained) projection from I3D features, the phrase gains shrink
   3–10× (duplex +0.10, triplet +0.32) — though C3 still sets the thesis-best
   action (16.07). The mechanism's power lives substantially in the
   pretrained video-text geometry.
4. **Head choice is irrelevant to composition once the MLP sees features
   (C4).** Feeding phrase-head primitives to the comp MLP ties the record
   (15.17/9.53 vs 15.52/9.73) — both heads are functions of the same features
   the MLP already receives raw. Head-design search closed; every remaining
   point of headroom is in the features.
5. **Phrasing matters (the Amber anomaly).** "showing an amber light" sits
   one token from red/green phrases; Amber collapsed 21.2 → 5.0 under the
   phrase head. Near-synonym phrases hurt; distinct phrases help — concrete
   input to the phrase review with Moradi.

## September implication

One lever, two upgrades, both unmeasured: true 8-frame clips (cheap re-cache)
and the InternVideo2-1B flagship (integration risk). The mechanism's full
effect (+1.4/+1.1) waits on genuinely aligned features. Timebox per
[[directions/thesis-proposal-fall-2026]]: hard mid-September go/no-go.

## Related

- [[findings/exp11-yolo-hybrid]] — the platform and the gate
- [[methods/stacked-composition-mlp]] — the record holder
- [[papers/wang-2024-internvideo2]] — the encoder family
- [[findings/exp9-attribution-grid]] — the grid discipline this inherits
