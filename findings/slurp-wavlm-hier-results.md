---
type: finding
title: "SLURP WavLM-Hier Results — 0.877/0.833 F1w, no collapse"
created: 2026-05-08
updated: 2026-05-08
sources:
  - "SLURP/logs/wavlm_large_hier_v3/final_results.json"
  - "SLURP/logs/wavlm_large_hier_v3/epoch_log.csv"
  - "SLURP/logs/wavlm_large_hier_v3/test_per_class_scenario.csv"
  - "SLURP/logs/wavlm_large_flat_ablation/final_results.json"
  - "SLURP/logs/whisper_hier/final_results.json"
  - "SLURP/logs/wavlm_large_hier/final_results.json"
tags: [finding, slurp, wavlm, hierarchical]
status: complete
---

# SLURP WavLM-Hier Results

Best run is **`wavlm_large_hier_v3`** (2026-03-26, 15 epochs, 47,475 steps, A6000). [[methods/wavlm-hier|Method details]].

## Headline test-set numbers

| Metric | Value |
|---|---|
| Scenario Accuracy | 0.878 |
| Scenario F1 (weighted) | **0.877** |
| Scenario F1 (macro) | 0.866 |
| Action Accuracy | 0.835 |
| Action F1 (weighted) | **0.833** |
| Action F1 (macro) | 0.783 |
| **Intent Accuracy (joint)** | **0.820** |
| Scenario collapse | none |
| Action collapse | none |

Best epoch by dev metric: 15 (final). Dev/test gap is small — dev scenario F1w 0.882 vs test 0.877.

## Dev curves (selected epochs)

| Epoch | teacher_p | Sc Acc | Act Acc | Sc F1w | Act F1w | Train Loss |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 0.683 | 0.462 | 0.683 | 0.457 | 5.532 |
| 5 | 0.71 | 0.855 | 0.784 | 0.853 | 0.784 | 0.974 |
| 10 | 0.36 | 0.877 | 0.839 | 0.876 | 0.838 | 0.556 |
| 15 | 0.00 | 0.883 | 0.850 | 0.882 | 0.850 | 0.460 |

The curriculum schedule visibly stabilises early action training — by epoch 5 (teacher_p=0.71) the action head is already at 0.78 F1w, faster than the scenario head reached 0.85.

## Hierarchy ablation

Same encoder, same training schedule, hierarchy components removed (no scenario→action conditioning, no curriculum, no ontology mask).

| Model | Sc F1w | Act F1w | Intent Acc | Δ vs flat |
|---|---|---|---|---|
| WavLM-Large flat | 0.874 | 0.823 | 0.796 | — |
| **WavLM-Large + Hier (v3)** | **0.877** | **0.833** | **0.820** | +0.003 / +0.010 / **+0.024** |

The hierarchy buys little on the per-task metrics but +2.4pp on joint Intent Accuracy (scenario AND action both correct). For papers that report Intent as the headline number, this is the load-bearing ablation.

## Encoder ablation

| Encoder | Params | Sc F1w | Act F1w | Intent Acc |
|---|---|---|---|---|
| WavLM-base-plus | 95M | 0.816 | 0.734 | 0.711 |
| WavLM-Large | 317M | **0.877** | **0.833** | **0.820** |

Going from base-plus to Large is a much bigger lever (+0.06 Sc F1w, +0.099 Act F1w, +0.109 Intent Acc) than the hierarchy stack. Encoder choice is the dominant factor.

## Per-class scenario F1 (test, top and bottom)

| Class | Support | F1 |
|---|---|---|
| email | 271 | 0.954 |
| iot | 220 | 0.950 |
| play | 387 | 0.935 |
| datetime | 103 | 0.923 |
| weather | 156 | 0.915 |
| transport | 124 | 0.906 |
| ... | ... | ... |
| takeaway | 57 | 0.877 |
| recommendation | 94 | 0.784 |
| cooking | 72 | 0.810 |
| **general** | **189** | **0.577** |

`general` is the lone outlier — F1 0.577 on 189 test utterances. This is consistent with `general` being a semantic catch-all in SLURP (questions/commands not fitting other scenarios), so it has the highest acoustic-lexical heterogeneity within a single class. Worth flagging as the obvious target for any per-class improvement work.

## Comparator landscape

| System | Sc F1w | Act F1w | Intent Acc | Notes |
|---|---|---|---|---|
| [[findings/slurp-collapse-e1|E1: frozen Wav2Vec2 + CE]] | 0.032 | 0.125 | — | Collapsed — failure baseline |
| Wang et al. 2021 (arXiv:2111.02735) | — | — | 0.8938 | Public Wav2Vec2/HuBERT benchmark; intent acc only |
| Phase 2 (SpeechBrain + HuBERT) | — | 0.73 (SLURP-F1) | 0.88 | Advisor's prior; SLURP-F1 includes slot-filling |
| Text Oracle (RoBERTa + gold transcript) | 0.790 | 0.553 | 0.541 | NLU ceiling on text — see [[comparisons/slurp-audio-vs-text-oracle|comparison]] |
| WavLM-base-plus + Hier | 0.816 | 0.734 | 0.711 | This work, smaller encoder |
| WavLM-Large flat | 0.874 | 0.823 | 0.796 | This work, hierarchy ablation |
| **WavLM-Large + Hier (v3)** | **0.877** | **0.833** | **0.820** | **This work, best** |

Wang et al. 2021's 89.38% intent accuracy on SLURP still beats this work's 82.0%. Their result is accuracy on slot-aware Intent (SLURP joint metric) trained with full slot-filling supervision; this work does not yet have a slot-filling head, which is the most likely cause of the gap.

## Caveats

- **Single seed** (seed=42). Reviewers will expect ≥3 seeds with std deviations.
- **No slot filling**. Phase 2's headline SLURP-F1 0.73 measured slot+intent jointly; this work measures only scenario+action+intent classification.
- **Hierarchy gains are marginal** on per-task metrics. The +0.024 Intent Acc is the only convincing ablation number.
- **Single training run per variant**. v2 has no final test eval (interrupted/superseded).
- **Run id mismatch**: `wavlm_large_hier_v3/final_results.json` records `run_id: whisper_hier` and `encoder: microsoft/wavlm-large` — the script's hardcoded run_id wasn't updated when switching encoders. Verified the encoder field for ground truth.

## Related

- [[methods/wavlm-hier|WavLM-Hier method spec]]
- [[findings/slurp-collapse-e1|E1 Collapse]] — what this method recovers from
- [[comparisons/slurp-audio-vs-text-oracle|Audio vs Text Oracle]]
- [[projects/slurp|SLURP project]]
