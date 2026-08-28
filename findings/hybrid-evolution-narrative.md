---
type: finding
title: "From One Network to the Sweep — Evolution of the Hybrid"
aliases: []
created: 2026-08-27
updated: 2026-08-27
sources:
  - "ROAD_Reason/experiments/exp11_yolo"
tags: [finding, narrative, exp11, hybrid, yolo, i3d, roialign, stacked-mlp, road-plusplus]
status: complete
---

# From One Network to the Sweep — Evolution of the Hybrid

The architectural narrative Dr. Moradi asked for: four stages from the
official baseline to the six-head record, each stage keeping what the last
one proved and replacing exactly what it measured as broken. Protocol
constants across every stage from Stage 1 on: identical val frames (36,717),
full-candidate rows (≤300 boxes/frame), f-mAP @ IoU 0.5, the baseline
repository's own evaluator.

---

## Stage 0 — The baseline: one network for everything (3D-RetinaNet)

**Architecture.** A single end-to-end network: I3D backbone (3D convolutions
over the clip) → feature pyramid → RetinaNet-style dense heads that score
every anchor for everything simultaneously.

**Input.** One 8-frame clip, `[8, 3, 600, 840]`.

**Internal.** Anchor grid over the pyramid levels; every anchor carries a
box regression and a 184-dim sigmoid vector: `[agentness 1 | agent 10 |
action 22 | loc 16 | duplex 49 | triplet 86]`. Compositions are free output
columns trained end-to-end alongside everything else.

**Output.** Class-agnostic NMS over its own anchors → detections, each with
the full 184 scores. On its own top-300 candidates over our val frames:
**agentness 35.36 · agent 16.67 · action 13.23 · loc 12.77 · duplex 11.32 ·
triplet 7.54.**

**What we measured as broken.** One network trades localization against
classification: its agent detection (16.67) is less than half what a
specialist detector achieves on the same frames. Its classification
knowledge, however, turned out to be good — a fact only visible after
decoupling. Rationale for the next step (Moradi directive #1): replace the
box source, keep the knowledge.

---

## Stage 1 — First hybrid: YOLO boxes + I3D scores, matched by overlap

**The change.** Localization is reassigned to a specialist. No public
ROAD-Waymo YOLO weights existed, so we trained YOLOv8x under the published
ECCV recipe; its epoch-1 checkpoint scored **35.24 agent f-mAP** (above the
31.6 published reference — pipeline validated), then froze.

**Inputs.**
- YOLO branch: single keyframe `[3, 1280, 1280]` → ≤300 boxes, each
  `(x1, y1, x2, y2)` normalized + confidence + agent class. Nothing else
  leaves YOLO.
- I3D branch: unchanged Stage-0 network producing its own top-300 candidates
  with their 184-dim sigmoids.

**The bridge (zero learned parameters).** Each YOLO box is matched to the
best-overlapping I3D candidate (IoU ≥ 0.5); the candidate's action / loc /
duplex / triplet sigmoids are copied onto the YOLO box. Unmatched boxes
score zero on those heads.

**Output row.** `[YOLO box | agentness = YOLO conf | agent = YOLO class
score | copied 4 heads]` →
**65.36 · 35.25 · action 12.86 · loc 13.36 · duplex 10.04 · triplet 6.81.**

**What we measured as broken.** The match rate: only **40.6%** of YOLO's
candidates find an I3D partner, so most rows carry zeros on the transferred
heads — a structural ceiling. The fix is obvious once stated: detections are
sparse, but *features exist everywhere*.

---

## Stage 2 — RoIAlign: pool the features, train a head (the AVA recipe)

**The change.** Stop asking I3D *where it detected things*; read its
feature map at YOLO's boxes instead. The I3D network is now run only up to
the pyramid — its detection heads are bypassed entirely.

**Inputs.**
- From I3D: pyramid level P3, `[256 ch, 8 time, 75, 105]` (stride 8, every
  cell motion-aware from the 3D convolutions); the keyframe's temporal
  slice `[256, 75, 105]`.
- From YOLO: the same boxes, scaled by 1/8 to the grid.
- Bridge: RoIAlign (1×1 output, sub-pixel bilinear) → **one 256-d vector
  per box**, for every box, matched or not. Cached to disk once.

**Trained part (the only one).** `Linear(256 → 184)`, focal loss with
per-class weights, trained on GT boxes + background rows + 293K of YOLO's
own junk train-split boxes as negatives.

**Two lessons learned here, both kept forever.**
1. **The confidence gate is structural**: raw head scores on all 300
   candidates collapse the ranking (action 5.3); multiplying by YOLO's box
   confidence restores it (15.9) — and retraining with realistic negatives
   did not remove the need. Final scores = head sigmoid × YOLO conf.
2. **Loss-side constraints are inert here**: the corrected t-norm cannot
   reach compositional columns through a linear head (columns train
   independently) — measured via bit-identical duplex weights across λ.

**Output.** **65.36 · 35.25 · action 15.92 · loc 19.60 · duplex 11.13 ·
triplet 6.65.** Beats the baseline on four heads; duplex ties; triplet still
belongs to the baseline.

**What we measured as broken.** The free compositional columns: rare
compositions (61–3,240 positives) can't be learned from scratch on frozen
features. Fixed composition (min/product of primitives over the valid
tuples, zero training) made things *worse* — it discards co-occurrence
knowledge. Conclusion: compositions need their own learned parameters, fed
by the primitives.

---

## Stage 3 — The stacked composition MLP (Brandon's design): the sweep

**The change.** Duplex and triplet stop being free columns. A small MLP
*composes* them from the head's own primitive predictions plus the pooled
evidence.

**Inputs.** Per box: the Stage-2 head's **ungated** primitive sigmoids
`[49]` (agentness + agent + action + loc), concatenated with the same 256-d
pooled feature → `[305]`.

**Trained part.** `Linear(305 → 512) → ReLU → Linear(512 → 135)` + sigmoid
→ the 49 duplex + 86 triplet scores — trained under video-level 2-fold
out-of-fold stacking (two throwaway fold-heads generate training inputs, so
the MLP never sees in-sample predictions; demanded by adversarial review).
Output vocabulary = the valid compositions only → **invalid predictions are
structurally impossible**, zero constraint penalty needed.

**Output.** Compositional columns replaced (primitives untouched), gated:
**65.36 · 35.25 · 15.92 · 19.60 · duplex 15.52 · triplet 9.73** — the
baseline beaten on **all six heads**.

**Why it works (measured ladder).** Fixed composition < free columns <
learned composition < learned composition + features: compositions carry
co-occurrence knowledge (Car-Stop = 38% of duplexes) that rules discard,
and evidence carries compositional signal beyond the primitives.

---

## The evolution in one table

| stage | agentness | agent | action | loc | duplex | triplet |
|---|---|---|---|---|---|---|
| 0 · baseline, one network | 35.36 | 16.67 | 13.23 | 12.77 | 11.32 | 7.54 |
| 1 · + YOLO boxes, IoU-copied scores | 65.36 | 35.25 | 12.86 | 13.36 | 10.04 | 6.81 |
| 2 · + RoIAlign features + trained head | 65.36 | 35.25 | 15.92 | 19.60 | 11.13 | 6.65 |
| 3 · + stacked composition MLP | 65.36 | 35.25 | 15.92 | 19.60 | **15.52** | **9.73** |

Each stage's gain is attributable to exactly one change: Stage 1 bought
localization (+18.6 agent), Stage 2 bought classification everywhere
features could reach (+3.1 action, +6.2 loc), Stage 3 bought composition
(+4.4 duplex, +2.9 triplet). Nothing was retrained twice; every component
that survived a stage is frozen in the next.

## Related

- [[findings/exp11-yolo-hybrid]] — the full result ledger
- [[methods/stacked-composition-mlp]] — Stage 3 in detail
- [[findings/road-waymo-schedule-overtraining]] — why YOLO's best is epoch 1
- [[findings/exp12-phrase-head-attribution]] — the proposed contribution built on this platform
- [[concepts/neuro-symbolic-constraints]] — the constraint arc this narrative closes
