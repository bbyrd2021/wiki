---
type: concept
title: "Encoder Collapse"
aliases: ["Majority-Class Collapse", "Frozen Encoder Collapse"]
created: 2026-05-08
updated: 2026-05-08
sources:
  - "SLURP/E1_results_summary.md"
  - "SLURP/encoder_ablation_notes.md"
tags: [concept, training-failure, imbalance, frozen-encoder]
status: complete
---

# Encoder Collapse

A failure mode where a model with a frozen pretrained encoder + classifier heads + standard loss converges to predicting the majority class for all inputs, within the first epoch, with no recovery. Distinct from underfitting (loss does decrease) and from collapse-to-zero (gradient norms are healthy). Documented on SLURP in [[findings/slurp-collapse-e1|E1]].

## Diagnostic signature

Three signals jointly indicate encoder collapse rather than other failure modes:

1. **Prediction concentration** — >95% of predictions on a single class (the most-frequent training class).
2. **Per-class F1 = 0 for >90% of classes** — minority classes receive zero correct predictions.
3. **Healthy gradients with declining magnitude** — gradient norm not zero, not exploding, but decreasing each epoch as the heads settle deeper into the collapsed solution.

If gradients are zero: encoder is frozen *and* heads are dead — different problem (initialisation or vanishing gradient).
If predictions are spread across classes but accuracy is low: underfitting — different problem (capacity or schedule).

## Three compounding causes

Encoder collapse rarely has one cause. Three factors compound:

### 1. Encoder pretraining objective misaligned with task

A frozen encoder pretrained for a different objective (e.g., contrastive frame-level on Wav2Vec2) produces representations that do not geometrically separate the downstream classes. Classifier heads then have no separable space to operate in.

| Encoder | Pretraining | Frozen suitability for SLU |
|---|---|---|
| Wav2Vec2 | Contrastive (frame-level phonetic) | Poor |
| HuBERT | Masked unit prediction (BERT-like) | Better |
| WavLM | Masked unit + denoising | Best (per SUPERB) |

### 2. Class imbalance + unweighted loss

When standard CE meets a long-tail label distribution, the gradient direction that most reduces loss is "predict the majority class for everything." Inverse-frequency weighting (capped) breaks this attractor.

### 3. No LR warmup / scheduling

A fixed learning rate gives heads no chance to explore the loss surface before the majority-class optimum captures the trajectory. Linear warmup + cosine annealing delays the capture long enough for non-trivial features to emerge.

**Removing any one cause is usually insufficient.** SLURP E5 (partial unfreeze + class-weighted CE) still collapsed because Wav2Vec2's representations remain unsuited even when partly fine-tuned. Only the combined fix — better encoder + class weighting + warmup — escaped (see [[methods/wavlm-hier|WavLM-Hier]]).

## Domain transfer

The pattern generalises beyond audio SLU. Conditions that produce it:

- A frozen or lightly-tuned pretrained backbone whose pretraining was task-agnostic.
- A classification head over a long-tail label distribution.
- Standard CE without class weighting or focal loss.
- Fixed LR without warmup.

Concrete risk areas in this lab:

- **[[datasets/road-plusplus|ROAD++]] action classes**: 21 actions with extreme imbalance (Mov 1.84M boxes ↔ Push 5K boxes). Frozen-backbone models risk collapsing to `Mov` for everything.
- **[[datasets/jaad|JAAD]] crossing labels**: 70/30 not-cross/cross split is mild but enough to produce majority-class collapse if backbone is poorly aligned.
- **VLM zero-shot detection**: a frozen CLIP backbone may collapse to background-class-equivalent predictions if heads aren't given warmup.

## Diagnostic checklist (apply early)

When training a new head on a frozen or partially-frozen encoder:

1. After epoch 1, log per-class prediction counts. If any class has > 50% of predictions, collapse risk is high.
2. After epoch 1, log per-class F1. If > 50% of classes have F1 = 0, collapse is forming.
3. Track gradient norm by epoch. Slow decline with no metric improvement = locked into collapsed solution.
4. If 1–3 trigger, swap encoder before trying loss tweaks. SLURP E5 showed loss tweaks alone do not escape.

## Related

- [[findings/slurp-collapse-e1|SLURP E1 — frozen Wav2Vec2 collapse]]
- [[methods/wavlm-hier|WavLM-Hier — collapse fix]]
- [[concepts/domain-shift|Domain Shift]]
- [[projects/slurp|SLURP project]]
