---
type: finding
title: "SLURP E1 — Frozen Wav2Vec2 Total Collapse"
created: 2026-05-08
updated: 2026-05-08
sources:
  - "SLURP/E1_results_summary.md"
  - "SLURP/logs/E1_frozen_ce/"
  - "SLURP/run_e1.py"
tags: [finding, slurp, collapse, wav2vec2, baseline]
status: complete
---

# SLURP E1 — Frozen Wav2Vec2 Total Collapse

The failure baseline that motivates [[methods/wavlm-hier|WavLM-Hier]]. A frozen Wav2Vec2-base + classifier heads + standard cross-entropy collapses to the majority class on both SLURP tasks within epoch 1, never recovers, and reproduces the Phase 3 paper's failure mode to three decimal places of F1w.

**Run**: `E1_frozen_ce` (2026-02-18, 5 epochs, RTX A6000, seed=42)

## Configuration

| Param | Value |
|---|---|
| Encoder | `facebook/wav2vec2-base` (94.4M params) |
| Freeze | Fully frozen — encoder weights not updated |
| Trainable | 411,200 params (heads only) |
| Loss | Standard CE, no class weighting |
| LR | 1e-4 fixed (no warmup, no schedule) |
| Grad clip | None |

## Test-set results

| Metric | Value |
|---|---|
| Scenario Accuracy | 0.135 |
| Scenario F1 (weighted) | 0.032 |
| Scenario F1 (macro) | 0.013 |
| Action Accuracy | 0.284 |
| Action F1 (weighted) | 0.125 |
| Action F1 (macro) | 0.010 |
| Scenario classes with zero F1 | **17 / 18** |
| Action classes with zero F1 | **45 / 46** |

Every scenario prediction is `calendar` (2,974 / 2,974 = 100%). Every action prediction is `query` (2,974 / 2,974 = 100%). These are the two highest-frequency training classes (calendar 14.7%, query 29.1%).

## Reproduction match to Phase 3 paper

| Metric | Phase 3 paper | E1 (this run) |
|---|---|---|
| Scenario F1w | 0.032 | **0.032** |
| Action Accuracy | 0.284 | **0.284** |
| Action F1w | 0.125 | **0.125** |
| Dominant scenario | calendar | calendar |
| Dominant action | query | query |

The match to three decimals confirms E1 reproduces the original failure rather than introducing a new one.

## Root cause

Three compounding factors:

1. **Frozen encoder pretrained for the wrong task** — Wav2Vec2 contrastive pretraining produces frame-level phonetic representations, not utterance-level semantic ones. The 18 scenarios and 46 actions are not linearly separable in its frozen embedding space.
2. **Unweighted CE under class imbalance** — `query` accounts for 29.1% of training actions; the loss landscape rewards predicting it for everything.
3. **No LR warmup or scheduling** — fixed 1e-4 with no warmup gives the heads no chance to escape the early majority-class optimum before it captures the trajectory.

Gradient norms (mean 3.6, range 2.2–10.4) confirm the heads are receiving signal. The collapse is not a gradient flow problem — it's that the only direction reducing loss in the frozen representation space is toward the majority class.

## Why this matters

E1 establishes that **encoder choice is a primary failure mode for SLU on imbalanced data**, not just a secondary tuning knob. The same training recipe with a properly chosen encoder (WavLM-Large in [[methods/wavlm-hier|WavLM-Hier]]) reaches scenario F1w 0.877 — a 27× improvement from the same data, same loss family, same compute budget. The fix is the encoder, not class weighting alone.

## Generalization

The collapse pattern — frozen pretrained encoder + imbalanced classifier heads + standard CE → majority-class collapse — is documented as a domain-general risk in [[concepts/encoder-collapse|Encoder Collapse]]. The diagnostic signals (per-class F1 = 0 for >90% of classes; prediction concentration > 95% on one class) transfer to AV perception (e.g., long-tail action classes in [[datasets/road-plusplus|ROAD++]]).

## Related

- [[methods/wavlm-hier|WavLM-Hier]] — the fix
- [[findings/slurp-wavlm-hier-results|WavLM-Hier results]] — what recovery looks like
- [[concepts/encoder-collapse|Encoder Collapse]] — generalised pattern
- [[projects/slurp|SLURP project]]
