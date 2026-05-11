---
type: finding
title: "Traffic Light Classifier Domain Shift — LISA → Workshop Rig"
aliases: ["traffic-light-misclassification", "red-as-yellow", "off-class-add"]
created: 2026-05-07
updated: 2026-05-07
sources:
  - "AutoDrivePerception2026/src/yolov10_ros/src/light_classifier_node.py"
  - "eff_light_detection/README.md"
  - "eff_light_detection/tools/bstld_to_patches.py"
  - "eff_light_detection/tools/merge_patch_datasets.py"
tags: [traffic-light, classification, domain-shift, lisa, bstld, ros]
status: complete
---

# Traffic Light Classifier Domain Shift

While running `perception_full.launch` against `sensor_data.bag` on 2026-05-07,
real **red** traffic lights in the bag were being classified as **yellow_light**.
Verified by side-by-side rqt overlay vs `/perception/traffic_lights`.

## Symptom (verbatim from session notes)

- Real red lights → predicted `yellow_light`
- Dark / off housings (no signal) → predicted `yellow_light` (separate failure mode)
- Repeatable across multiple frames; not a single-frame artifact.

## Root cause — not an indexing bug

Verified the index→label mapping in `light_classifier_node.py:84-91` against the
training repo (`bbyrd2021/efficient_light_detection`):

- `tools/generate_patch_dataset.py:222` writes folders alphabetically:
  `green_left, green_light, red_left, red_light, yellow_left, yellow_light`
- `train.py` consumes via `datasets.ImageFolder` which sorts directories alphabetically
- The deployed checkpoint `efficientnet_b0_20260211_195116_light_cls.pth` has
  `classifier.1.weight: [6, 1280]` — exactly 6 classes, no `green_straight`
- ROS mapping aligns 1:1 with that alphabetical order.

So the index→label assignment is correct. The misclassification is a **genuine
red↔yellow confusion**, not an off-by-one indexing bug.

## Why LISA-only training doesn't generalize to our rig

Validation accuracy of 96.5% reported on LISA's held-out split measures
performance on more San Diego dashcam frames captured with the same stereo
camera under similar lighting. It does not generalize to:

- Different camera (USB camera at the middle position vs. LISA's dashcam stereo)
- Different lighting / time-of-day distribution at the rig
- Different distance/scale distribution of light crops
- Different YOLO box framing (LISA's patch generator may crop differently
  than YOLOv10 does at deployment)

A dim red light in this bag, downscaled to 224×224 for EfficientNet-B0 input,
can land closer to LISA's *yellow* cluster in feature space than its *red*
cluster, even when it's clearly red to a human. Standard domain-shift
behavior, not a model defect.

## Off-signal failure was masked by an HSV gate

The rig copy of `light_classifier_node.py` (lines ~252-264 per the 2026-05-07
session notes) had a brightness gate that short-circuited dark crops to
`state="off"` *before* the classifier ran, because the model has **no `off`
class** and confidently mislabels dark housings as yellow. For the 2026-05-07
session this gate was commented out, exposing the dark-housing→yellow failure
directly. Note: the local `/data/repos/AutoDrivePerception2026/` checkout is a
divergent copy without this gate — the gate lives on the rig only.

This was a band-aid: the right fix is to give the model an `off` class and
remove the gate.

## Resolution — retrain with BSTLD as a domain bridge + add `off`

### Decisions

- **Class taxonomy**: 7 classes (drop `green_straight`, add `off`):
  `green_left, green_light, off, red_left, red_light, yellow_left, yellow_light`.
  Alphabetical, so ImageFolder indexing follows the same convention as before.
  - `green_straight` had only 205 samples in LISA, none in BSTLD — not viable.
  - `off` is a brand-new class, populated from BSTLD only.
- **Domain bridge**: BSTLD (Bosch Small Traffic Lights) train split. German
  urban driving — different geography, different camera, different lighting
  distribution from LISA's San Diego dashcam.
  - 5,093 images / 10,756 boxes, 1280×720 RGB
  - Non-commercial license (academic OK)
  - Has explicit `off` class — the only feasible source for this label
- **Did not use**: BSTLD test split (4-class only — no arrows), BSTLD RIIB
  raw 12-bit images, additional unlabeled set, Roboflow Universe small clones.

### BSTLD label mapping → our 7 classes

| BSTLD label | Mapped to | Train count |
|---|---|---|
| Green | green_light | 5,207 |
| Red | red_light | 3,057 |
| RedLeft | red_left | 1,092 |
| **off** | **off** | **726** |
| Yellow | yellow_light | 444 |
| GreenLeft | green_left | 178 |
| YellowLeft | yellow_left | 0 (none in train) |
| GreenStraight, GreenRight, RedStraight, RedRight, *StraightLeft, *StraightRight | (skipped — no matching class) | 52 total |

### Patch extraction

- LISA: existing `tools/generate_patch_dataset.py`, dayTrain+nightTrain →
  daySequence1+nightSequence1 split, 10px min bbox, 1.2× expansion, 128×128.
  Resulting train/val: **49,720 / 23,596** patches across 6 classes.
- BSTLD: new `tools/bstld_to_patches.py`. **6px min bbox** (BSTLD median
  min-dim is 8.5px; LISA's 10px cutoff drops 56% of BSTLD data including most
  yellow/off samples). 90/10 image-level split. Resulting **7,667 / 863**
  patches.
- Merge: `tools/merge_patch_datasets.py` hard-links both source trees into
  `/data/datasets/lisa_bstld_patches_128/`, drops `green_straight`, recomputes
  inverse-frequency class weights.

### Merged dataset (per-class train counts)

| Class | Train | Val | Class weight |
|---|---|---|---|
| green_light | 25,149 (43.8%) | 10,986 | 0.326 |
| red_light | 20,002 (34.9%) | 11,686 | 0.410 |
| red_left | 8,613 (15.0%) | 232 | 0.952 |
| yellow_light | 1,505 (2.6%) | 867 | 5.45 |
| green_left | 1,386 (2.4%) | 590 | 5.92 |
| **off** | **435 (0.8%)** | **45** | **18.85** |
| yellow_left | 297 (0.5%) | 53 | 27.60 |
| **TOTAL** | **57,387** | **24,459** | |

### Training recipe

- Model: EfficientNet-B0, ImageNet-pretrained, fresh classifier head (no
  warm-start from the 6-class checkpoint — class-count differs).
- 50 epochs, batch 32, Adam lr 0.001, cosine annealing, early-stop patience 10.
- 128×128 input, `--use-class-weights`.
- Augmentation: RandomRotation(10), ColorJitter brightness/contrast/saturation
  ±30-40%, hue ±2%, GaussianBlur, RandomAdjustSharpness. No horizontal flip
  (would scramble left-arrow semantics).
- Logging: added per-N-batch `print(..., flush=True)` lines so `tail -f` shows
  intra-epoch progress (tqdm-only output is invisible in tee'd logs).

### Results

Trained on RTX A6000, early-stopped at epoch 25 (peak at epoch 15), 21 minutes
wall clock. Checkpoint:
`eff_light_detection/experiments/efficientnet_b0_20260507_134855/best.pth`.

**Overall val accuracy: 96.05%** (24,459 patches)

Per-class metrics on the merged val set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| red_light    | 0.990 | 0.986 | **0.988** | 11,686 |
| green_light  | 0.971 | 0.974 | **0.973** | 10,986 |
| yellow_light | 0.794 | 0.975 | **0.875** | 867 |
| yellow_left  | 0.589 | 1.000 | 0.741 | 53 |
| red_left     | 0.619 | 0.582 | 0.600 | 232 |
| **off**      | **0.389** | **0.933** | **0.549** | **45** |
| green_left   | 0.618 | 0.334 | 0.433 | 590 |
| **Macro avg**| 0.710 | 0.826 | 0.737 | |

Critical confusions vs the rig failure modes:

| True → Predicted | Count | Rate | Note |
|---|---|---|---|
| **red_light → yellow_light** | 73 / 11,686 | **0.62%** | overall rate is low — but see BSTLD-only breakdown below |
| yellow_light → red_light | 3 / 867 | 0.35% | reverse direction, negligible |
| **off → yellow_light** | **0 / 45** | **0%** | dark-housing failure — closed |
| green_light → off | 51 / 10,986 | 0.46% | new false-off rate (model occasionally calls active green "off") |
| red_light → off | 10 / 11,686 | 0.085% | new false-off rate on red, very low |
| yellow_light → yellow_left | 17 / 867 | 1.96% | both yellow states, arrow-vs-circular swap (not a safety regression) |

### Old vs new — apples-to-apples comparison

The merged val set's overall accuracy is dominated by LISA (23,596 of 24,459
patches), so headline numbers mostly reflect within-domain LISA performance.
The honest test is BSTLD-only val (818 patches, OOD for the old model — closer
analog to the rig's USB camera).

Old checkpoint:
`experiments/efficientnet_b0_20260211_195116/best.pth` (LISA-only, 6 classes,
no `off`). Evaluated on the same 24,414 non-`off` val patches.

| | Old (LISA-only, 6 cls) | New (LISA+BSTLD, 7 cls) |
|---|---|---|
| Overall (6 cls) | **96.22%** | 96.06% |
| LISA-only val | **96.50%** | 96.27% |
| **BSTLD-only val** | 88.14% | **90.10%** |

#### BSTLD red_light (257 patches) — the original failure proxy

| | Old | New |
|---|---|---|
| → red_light (correct) | **222 (86.4%)** | 208 (80.9%) |
| → yellow_light | 17 (6.6%) | **24 (9.3%)** ⚠️ |
| → off | n/a | 9 (3.5%) |
| → red_left | 11 (4.3%) | 16 (6.2%) |

**Trade-off:** on BSTLD red_lights specifically, the new model's red→yellow
rate is *slightly worse* (9.3% vs 6.6%), and overall red recall on BSTLD
dropped from 86.4% to 80.9%. Capacity went into the new `off` class and into
arrow classes — some uncertain reds get re-routed to "off" or "yellow"
instead of being committed to "red". Adding more red/yellow data (e.g. a
Roboflow tertiary source) is a candidate next step to push red recall back up
without losing the `off` win.

#### Off-class behavior (45 BSTLD `off` val patches)

| Predicted | Old (no `off` class) | New |
|---|---|---|
| off | n/a | **42 (93%)** |
| green_light | **35 (78%)** | 0 |
| red_left | 6 (13%) | 1 |
| red_light | 3 (7%) | 0 |
| yellow_left | 1 | 0 |
| green_left | 0 | 2 |

**Unambiguous win.** Without an `off` class the old model dumps 78% of dark
housings into `green_light` (note: not `yellow_light` as the rig session
notes suggested — likely the rig camera/exposure produces dark frames that
look different from BSTLD's, and routes to a different wrong class). The new
model gets it right 93% of the time end-to-end and the HSV gate becomes
redundant.

#### Per-class recall on BSTLD val

| Class | Old | New | Δ |
|---|---|---|---|
| green_left | 90% | 100% | +10 |
| green_light | 98.5% | 98.8% | ≈ |
| **red_left** | **59%** | **80%** | **+21** |
| red_light | 86% | 81% | **−5** ⚠️ |
| yellow_light | 62% | 83% | +21 |
| off (separate) | 0% | 93% | +93 |

#### Honest read of v1

- **Big wins:** `off` class works (gate retired), `red_left`, `yellow_light`,
  `green_left` BSTLD recall up ~20 points each.
- **Trade-off:** BSTLD red_light recall dropped 5 points; red→yellow
  specifically is slightly worse (not better) on BSTLD.
- **Within-domain (LISA val):** essentially a wash, slight drop for new
  model — the price for cross-domain generalization.
- **Bag verdict is still empirical.** None of these val numbers are directly
  the rig's `sensor_data.bag`. They're a directional signal at best.

### v2 — adding S2TLD as a third domain

To address v1's BSTLD red recall regression, we added S2TLD (SJTU Small
Traffic Light Dataset, MIT license) as a third source. Chinese urban
dashcam, 5,786 images / 14,130 instances, classes: red/yellow/green/off
(no arrows; `wait_on` skipped as it's a composite multi-light state).

Merged train/val:

| Source | Train | Val |
|---|---|---|
| LISA | 49,720 | 23,596 |
| BSTLD | 7,667 | 863 |
| S2TLD | 12,418 | 1,397 |
| **Total** | **69,805** | **25,856** |

Added S2TLD train counts: red_light 6,914, green_light 4,809, off 459,
yellow_light 236. Big red boost (+35%), `off` doubled.

v2 trained 30 min on RTX A6000, early-stopped at epoch 30 (peak at 20),
**best val acc 96.44%**. Checkpoint:
`experiments/efficientnet_b0_20260507_152742/best.pth`.

### Three-way comparison: old (6-class) vs v1 (LISA+BSTLD) vs v2 (+S2TLD)

| | Old | v1 | v2 |
|---|---|---|---|
| **LISA val acc** | 96.50% | 96.27% | **96.64%** |
| **LISA red recall** | 98.61% | 98.99% | **99.54%** |
| **LISA red→yellow** | ~1% | 0.43% | **0.02%** |
| **BSTLD val acc** | 88.14% | **90.27%** | 87.02% |
| **BSTLD red recall** | **86.38%** | 80.93% | 74.71% |
| **BSTLD red→off** | n/a | 3.50% | 5.45% |
| **BSTLD off recall** | 0% | **93.33%** | 91.11% |
| **S2TLD val acc** | 71.01% | 71.65% | **99.00%** |
| **S2TLD red recall** | 64.43% | 58.99% | **99.62%** |
| **S2TLD red→yellow** | 6.84% | 12.53% | **0.25%** |
| **S2TLD off recall** | 0% | 85.11% | **100%** |
| **Overall macro F1** | n/a | 0.737 | **0.802** |

### Honest read of v2

- **v2 is broadly better.** Macro F1 0.74 → 0.80, LISA red→yellow basically
  eliminated (0.02%), S2TLD essentially solved (99% acc, 99.6% red recall).
- **But the original goal — push BSTLD red recall back up — failed.**
  v1: 80.9% → v2: 74.7%. **Adding S2TLD made BSTLD red recall worse, not
  better.** S2TLD's red distribution is far enough from BSTLD's that the
  decision boundary moved further from BSTLD. More data doesn't always
  help a specific subdomain.
- **`off` recall is solid across all three test domains** (91-100%).
  The dark-housing fix is robust.

### Decision: ship both checkpoints, A/B at the rig

We don't have labeled `sensor_data.bag` data, so we can't pick blind.
The unified deploy doc at
`/data/repos/eff_light_detection/RIG_AB_HANDOFF.md` covers both checkpoints,
expected behavior on each domain analog, and an A/B procedure for the rig
agent to pick empirically based on visible failure-mode reduction.

The original v1-only handoff at
`experiments/efficientnet_b0_20260507_134855/RIG_HANDOFF.md` is preserved
as the v1 reference but is superseded by `RIG_AB_HANDOFF.md`.

**Off precision is 38.9%** because val support is only 45 patches — the model
predicts off ~108 times across the val set, of which 42 are true off. Most
false positives come from green_light (51) but at 0.46% per-row this is still
better than the disabled HSV gate's behavior (which caught all dim crops
indiscriminately). Worth watching on the rig.

### Where the deliverables live

- `experiments/efficientnet_b0_20260507_134855/best.pth` — 7-class checkpoint
- `experiments/efficientnet_b0_20260507_134855/eval/` — confusion matrix
  (PNG + CSV), per-class metrics CSV, classification report, misclassified
  sample crops grouped by true class
- `class_index.json` — updated 7-class mapping at the repo root
- `experiments/efficientnet_b0_20260507_134855/RIG_HANDOFF.md` — deploy
  brief for the rig's Claude (paths, code edits, verification steps)

## ROS pipeline change required after retrain

After auditing the live `bbyrd2021/yolov10_ros` GitHub repo, the deploy is
much smaller than originally inferred from the session notes:

1. **`config/light_classifier_params.yaml`** — add a `class_names:` list with
   the 7 alphabetical labels (`green_left, green_light, off, red_left,
   red_light, yellow_left, yellow_light`). The node already reads this via
   `~class_names` ROS param at `light_classifier_node.py:119-123` and
   rebuilds the classifier head dynamically from `len(self.class_map)`
   (line 137) — so **no Python edit is needed** to switch taxonomy.
2. **`launch/pipeline.launch`** — update the `light_weights` arg (line 18)
   to point at the new `.pth`.
3. **`src/light_classifier_node.py:77`** — `_INPUT_SIZE = 224` is a bug
   independent of the retrain. Training is at **128**. EfficientNet doesn't
   crash on size mismatch (no rigid input layer; GAP handles any feature
   map), but features learned at 128 don't transfer ideally to 224. **This
   may have been silently degrading rig classification regardless of which
   checkpoint was loaded.** Set to 128. Apply this fix even if rolling back
   to the old 6-class checkpoint.
4. **No HSV gate to delete.** The session notes mentioned a gate at
   `:252-264` but the committed source has no such code there — it's just
   the publish-and-spin block. The gate was a local-only edit on the rig
   that was never pushed (or the line numbers crossed).

**Bonus (non-blocking):** `crop_roi` in `image_utils.py` applies *symmetric
pixel* padding (`roi_padding: 5`) before cropping. Training used
*multiplicative* 1.2× expansion. For 25-px lights this is ~10% expansion at
inference vs 20% at training; for 8-px BSTLD-style lights it's 62% vs 20%.
The inference framing for small lights is much looser than training. Either
bump `roi_padding` or refactor to multiplicative expansion to close the
gap — small effect on typical lights but worth tightening.

The matching `class_index.json` for the new model is at the repo root in
`/data/repos/eff_light_detection/class_index.json`.

## Related

- [[projects/eff-light-detection]] — classifier project page
- [[tools/traffic-light-classification]] — pipeline tool page
- [[tools/ros-perception-pipeline]] — ROS deployment context
- BSTLD: https://zenodo.org/records/12706046 (non-commercial license)
