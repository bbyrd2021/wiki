---
type: finding
title: "From One Network to the Sweep — Evolution of the Hybrid"
aliases: []
created: 2026-08-27
updated: 2026-09-02
sources:
  - "ROAD_Reason/experiments/exp11_yolo"
tags: [finding, narrative, exp11, hybrid, yolo, i3d, roialign, stacked-mlp, road-plusplus]
status: complete
---

# From One Network to the Sweep — Evolution of the Hybrid

The architectural narrative Dr. Moradi asked for: four stages from the
official baseline to the six-head record, each stage keeping what the last
one proved and replacing exactly what it measured as broken. Protocol
constants from Stage 1 on: identical val frames (36,717), full-candidate
rows (≤300 boxes/frame), f-mAP @ IoU 0.5, the baseline repository's own
evaluator.

**Naming note (important for the figures).** The baseline "3D-RetinaNet" is
three components stacked: the **I3D backbone** (3D convolutions), the
**feature pyramid (FPN)** on top of it, and the **RetinaNet machinery**
(anchor grid + dense classification/regression heads + NMS) on top of that.
"I3D" names only the backbone. The stages below track each component's fate
individually — RetinaNet's exit happens at Stage 2, not Stage 1.

---

## Stage 0 — The baseline: one network for everything (3D-RetinaNet)

**Components in play:** I3D backbone → FPN → RetinaNet anchors + heads +
NMS. All trained end-to-end, one loss.

**Input.** One 8-frame clip, `[8, 3, 600, 840]`.

**Internal.** The FPN levels carry an anchor grid; the RetinaNet heads give
every anchor a box regression and a 184-dim sigmoid vector `[agentness 1 |
agent 10 | action 22 | loc 16 | duplex 49 | triplet 86]`. Compositions are
free output columns trained end-to-end with everything else.

**Output.** RetinaNet's class-agnostic NMS over its own anchors →
detections with all 184 scores: **agentness 35.36 · agent 16.67 · action
13.23 · loc 12.77 · duplex 11.32 · triplet 7.54.**

**What we measured as broken.** One network trades localization against
classification: agent detection is less than half a specialist's on the
same frames. The classification *knowledge* turned out to be good —
visible only after decoupling. Rationale for the next step (Moradi's #1):
replace the box source, keep the knowledge.

---

## Stage 1 — First hybrid: YOLO boxes + 3D-RetinaNet scores, matched by overlap

**Components in play:** YOLOv8x (new, frozen after training) + the ENTIRE
Stage-0 3D-RetinaNet unchanged — I3D backbone, FPN, **and the RetinaNet
heads + NMS, which are still fully in service here**: they produce the
candidate detections whose scores get copied. This is RetinaNet's last
stage as a working component.

**Inputs.**
- YOLO branch: single keyframe `[3, 1280, 1280]` → its own head + NMS →
  ≤300 boxes, each `(x1, y1, x2, y2)` normalized + confidence + agent
  class. Nothing else leaves YOLO. (Trained under the published ECCV
  recipe; best = epoch 1 at **35.24 agent**, above the 31.6 published
  reference; frozen thereafter.)
- 3D-RetinaNet branch: clip `[8, 3, 600, 840]` → backbone → FPN →
  RetinaNet heads → NMS → its own top-300 candidates, each with 184
  sigmoids. Two NMS passes exist in this stage — YOLO's and RetinaNet's —
  one per branch.

**The bridge (zero learned parameters).** Each YOLO box is matched to the
best-overlapping RetinaNet candidate (IoU ≥ 0.5); that candidate's action /
loc / duplex / triplet sigmoids are copied onto the YOLO box. Unmatched
boxes score zero on those heads.

**Output row.** `[YOLO box | agentness = YOLO conf | agent = YOLO score |
copied 4 heads]` → **65.36 · 35.25 · 12.86 · 13.36 · 10.04 · 6.81.**

**What we measured as broken.** The match rate: only **40.6%** of YOLO's
candidates find a RetinaNet partner — because RetinaNet's detections are
sparse and its localization weak (the very reason it was demoted). The
diagnosis indicts the RetinaNet machinery specifically: its *detections*
are the bottleneck, while the backbone's *features* exist at every location.

---

## Stage 2 — RoIAlign: retire RetinaNet, pool the features, train a head

**Components in play:** YOLO unchanged. From the baseline model, only the
**I3D backbone + FPN** remain in the dataflow. **The RetinaNet machinery —
anchors, dense heads, NMS — is retired here**: we hook the FPN output
mid-network and ignore everything downstream (the heads still execute in
the forward pass but their outputs are never read). From this stage on,
"the I3D branch" literally means backbone + FPN only.

**Inputs.**
- From backbone+FPN: pyramid level P3, `[256 ch, 8 time, 75, 105]`
  (stride 8, every cell motion-aware from the 3D convolutions); the
  keyframe's temporal slice `[256, 75, 105]`.
- From YOLO: the same boxes, scaled by 1/8 to the P3 grid.
- Bridge: RoIAlign (1×1 output, sub-pixel bilinear) → **one 256-d vector
  per box, matched or not** — the Stage-1 ceiling dissolves. Cached once.

**Trained part (the only one).** `Linear(256 → 184)`, focal loss with
per-class weights, on GT boxes + background rows + 293K of YOLO's own junk
train-split boxes as negatives.

**Two lessons, kept forever.** (1) The **confidence gate is structural**:
ungated scores on 300 candidates collapse the ranking (action 5.3 → gated
15.9); final score = head sigmoid × YOLO conf. Note what this replaces —
in Stage 0/1, RetinaNet's own background-trained anchors did this job; the
gate is its functional successor. (2) **Loss-side constraints are inert
through a linear head** (columns train independently; bit-identical duplex
weights across λ).

**Output.** **65.36 · 35.25 · 15.92 · 19.60 · 11.13 · 6.65.** Four heads
beaten, duplex tied, triplet still the baseline's.

**What we measured as broken.** The free compositional columns: rare
compositions (61–3,240 positives) can't be learned from scratch on frozen
features — and fixed composition (min/product over valid tuples, zero
training) is *worse*, because rules discard co-occurrence knowledge.
Compositions need their own learned parameters, fed by the primitives.

---

## Stage 3 — The stacked composition MLP (Brandon's design): the sweep

**Components in play:** identical to Stage 2 plus one new trained module.
Nothing else changes; the primitives' path is untouched.

**Inputs.** Per box: the Stage-2 head's **ungated** primitive sigmoids
`[49]` (agentness + agent + action + loc) concatenated with the same 256-d
pooled feature → `[305]`.

**Trained part.** `Linear(305 → 512) → ReLU → Linear(512 → 135)` +
sigmoid → the 49 duplex + 86 triplet scores — trained under video-level
2-fold out-of-fold stacking (throwaway fold-heads generate the training
inputs; the MLP never sees in-sample predictions). Output vocabulary = the
valid compositions only → **invalid predictions structurally impossible**,
no constraint penalty needed.

**Output.** Compositional columns replaced, primitives untouched, gated:
**65.36 · 35.25 · 15.92 · 19.60 · 15.52 · 9.73** — all six heads beaten.

**Why it works (measured ladder).** Fixed composition < free columns <
learned composition < learned composition + features: compositions carry
co-occurrence knowledge (Car-Stop = 38% of duplexes) that rules discard,
and evidence carries compositional signal beyond the primitives.

---

## Component fate across the stages

| component | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| I3D backbone (3D convs) | trained | frozen, serving | frozen, serving | frozen, serving |
| FPN | trained | frozen, serving | frozen, serving (P3 tapped) | frozen, serving |
| RetinaNet anchors + heads | trained, the classifier | frozen, **score source** | **retired** (execute, unread) | retired |
| RetinaNet NMS | the box source | serving its branch | retired | retired |
| YOLOv8x | — | **box + agent source** | same | same |
| Linear head (256→184) | — | — | **trained** (primitives of record) | frozen, serving |
| composition MLP | — | — | — | **trained** |
| confidence gate | (implicit in RetinaNet's training) | — | **introduced** | serving |

## The evolution in one table

| stage | agentness | agent | action | loc | duplex | triplet |
|---|---|---|---|---|---|---|
| 0 · baseline, one network | 35.36 | 16.67 | 13.23 | 12.77 | 11.32 | 7.54 |
| 1 · + YOLO boxes, IoU-copied scores | 65.36 | 35.25 | 12.86 | 13.36 | 10.04 | 6.81 |
| 2 · + RoIAlign features + trained head | 65.36 | 35.25 | 15.92 | 19.60 | 11.13 | 6.65 |
| 3 · + stacked composition MLP | 65.36 | 35.25 | 15.92 | 19.60 | **15.52** | **9.73** |

Stage 1 bought localization (+18.6 agent), Stage 2 bought classification
everywhere features reach (+3.1 action, +6.2 loc), Stage 3 bought
composition (+4.4 duplex, +2.9 triplet). Nothing retrained twice; every
surviving component frozen in the next stage.

## The figure set (paper tier, 2026-09-02)

Seven figures in `~/Downloads/evolution_diagrams/` (`paper_stage*.{drawio,png}`),
generated from `gen_paper_stages.py` / `gen_paper_yolo.py` except the
3D-RetinaNet baseline, which is Brandon's hand-edited canonical file
(restyled in place, backup kept):

- **Two twin baselines** — `paper_stage0_retinanet` and `paper_stage0_yolov8x`
  share one composition (backbone funnel, feature cascade, pyramid, heads,
  dashed detail inset, same street-photo input) so they read as "same
  replicate-first treatment, two detectors." The YOLO figure was verified
  claim-by-claim against the installed ultralytics 8.3.226 yaml and the
  exp11 `best.pt` checkpoint itself (Detect head: Conv3x3 ×2 → Conv1x1 per
  branch, reg_max 16, nc 10, max_det 300).
- **Stage 0 pt2** — `paper_stage0_pt2`, the baseline redrawn on the stage
  scaffold as a bridge slide: bottom row only (clip → trained 3D-RetinaNet →
  detections → output), top row deliberately empty so stage 1's YOLO arrival
  reads as the diff. Its caption states the accurate lifetime (Brandon's
  catch, 2026-09-02): stages 1 to 3 keep the network frozen and change how
  it is read; stage 4 replaces it entirely — by the record configuration,
  YOLO's boxes are the only stage-1 survivor.
- **Four evolution stages** — `paper_stage1_transfer` through
  `paper_stage4_phrase`, station-consistent so consecutive slides read as
  diffs.
- **Badge grammar** — 40px inline-SVG corner badges: fire = trained,
  3D ice cube = frozen, defined in each legend. The flame migrates stage to
  stage (YOLO → linear head → composition MLP → projection) while ice
  accumulates behind it; both stage-0 baselines are all fire, no ice.
- **Style grammar** — semantic borders (each fill's darker shade), tinted
  encoder trapezoids in identity colors (YOLO blue, 3D-RetinaNet magenta,
  InternVideo2 teal), per-stage accent bar keyed to that stage's new
  component, input boxes carry code-verified dims (`[3, 600, 840]` frame,
  `[3, 8, 600, 840]` clip, `[n, 3, 8, 224, 224]` crops).
- **Exports** — 4x scale, transparent background (slide-ready; edge-label
  chips assume light slides). Type ladder 24/15/14 with a 13pt floor:
  no text on any figure renders more than one size below box text
  (Brandon 2026-09-02, after small labels blurred on projected slides).

**Caption precision (Brandon 2026-09-02).** Stage 1's caption originally
said "transferred heads are copied" — wrong on both counts: the *heads*
never move (their *scores* are copied), and it never named the source
model. Corrected to: "Each YOLO box takes the action, location, duplex and
triplet scores of the 3D-RetinaNet detection it overlaps; no match means
zeros on those four heads." The direction matters and is easy to invert
when summarizing: YOLO (the stronger detector) supplies boxes + conf +
agent scores; the four composition-head scores flow **from 3D-RetinaNet
onto YOLO's boxes**, never the reverse — YOLO has no such heads.

## Related

- [[findings/exp11-yolo-hybrid]] — the full result ledger
- [[methods/stacked-composition-mlp]] — Stage 3 in detail
- [[findings/road-waymo-schedule-overtraining]] — why YOLO's best is epoch 1
- [[findings/exp12-phrase-head-attribution]] — the proposed contribution built on this platform
- [[concepts/neuro-symbolic-constraints]] — the constraint arc this narrative closes
