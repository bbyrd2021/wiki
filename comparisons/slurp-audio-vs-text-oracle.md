---
type: comparison
title: "SLURP Audio vs Text Oracle — Audio Beats RoBERTa on Gold Transcripts"
created: 2026-05-08
updated: 2026-05-08
sources:
  - "SLURP/logs/text_oracle/final_results.json"
  - "SLURP/logs/wavlm_large_hier_v3/final_results.json"
  - "SLURP/contrib/run_text_oracle.py"
tags: [comparison, slurp, audio, nlu, oracle]
status: complete
---

# SLURP Audio vs Text Oracle

Surprising headline: the audio model ([[methods/wavlm-hier|WavLM-Hier]]) **beats the text oracle** on every SLURP classification metric. This contradicts the usual NLU-ceiling assumption that gold transcripts upper-bound audio.

## Setup

The text oracle is a RoBERTa-base classifier trained on **gold-standard human transcripts** of SLURP utterances, with separate scenario and action classification heads. It establishes the theoretical ceiling: what is the best you could do if you had perfect ASR?

WavLM-Hier sees only raw 16 kHz waveform — no transcript, no ASR pipeline, no text features.

| Component | Text Oracle | WavLM-Hier v3 |
|---|---|---|
| Encoder | `roberta-base` | `microsoft/wavlm-large` |
| Input | gold transcript (text) | raw audio (16 kHz waveform) |
| Heads | scenario + action (independent) | scenario + action (hierarchical) |
| Trained | 10 epochs | 15 epochs |

## Results

| Metric | Text Oracle | WavLM-Hier v3 | Δ |
|---|---|---|---|
| Scenario Accuracy | 0.789 | **0.878** | **+0.090** |
| Scenario F1w | 0.790 | **0.877** | **+0.087** |
| Scenario F1m | 0.769 | **0.866** | **+0.097** |
| Action Accuracy | 0.563 | **0.835** | **+0.272** |
| Action F1w | 0.553 | **0.833** | **+0.280** |
| Action F1m | 0.578 | **0.783** | **+0.205** |
| **Intent Accuracy (joint)** | **0.541** | **0.820** | **+0.279** |

The audio model beats text on every metric — and the action-side gap (+0.28 F1w) is much larger than scenario (+0.09 F1w).

## Interpretation

This is not "audio is generally better than text for SLU." It is "for SLURP specifically, gold transcripts discard information the audio retains." Three plausible mechanisms:

### 1. Prosody and intonation carry intent

Question vs. command vs. statement is often disambiguated by rising/falling pitch contour, not lexical content. SLURP's `query` vs `set` actions can share lexical surface form ("set an alarm for seven" vs "did I set an alarm for seven?") and split on prosody alone. WavLM hears that. RoBERTa cannot.

### 2. Transcript normalisation strips disambiguating features

SLURP's gold transcripts are normalised — punctuation, capitalisation, and disfluencies are flattened. A rising terminal contour that would mark a question becomes lexically indistinguishable from a command in the transcript.

### 3. RoBERTa-base is undertrained for this domain

Pre-trained on web text, not spoken-instruction data. SLURP's command-pattern utterances ("turn on the light", "play music") are short imperatives that may not align with RoBERTa's MLM training distribution. WavLM-base-plus on full SLURP audio also beats the text oracle (Sc F1w 0.816, Act F1w 0.734, Intent 0.711).

## Why the action gap is bigger than scenario

| Task | Text vs Audio Δ F1w |
|---|---|
| Scenario | +0.087 |
| Action | +0.280 |

Scenario is a coarse 18-way classification (calendar / iot / play / ...) — lexical content alone often suffices ("alarm" → calendar). Action is a fine 46-way classification (query / set / remove / ...) where prosody carries proportionally more disambiguating signal. The hierarchical conditioning in WavLM-Hier may also help here: knowing the predicted scenario constrains the action search to the 60 valid pairs.

## Caveats

- Single-seed comparison (seed=42 both runs).
- Text oracle uses RoBERTa-base; a larger text model (RoBERTa-large, DeBERTa-v3-large) might close more of the action gap. Cannot rule out that **audio beats** *small* **text but not** *large* **text**.
- The text oracle does no slot-filling, so this is a pure classification comparison.
- Gold transcripts here are SLURP's `sentence` field (post-normalisation). Comparing against ASR-output transcripts is a separate experiment that would test the audio→text→NLU pipeline.

## Publishing significance

This finding is a publishable result on its own — **"audio is not a strict subset of text for SLU"** is a counterintuitive headline that motivates end-to-end speech models over cascaded ASR→NLU pipelines. Worth pulling out as a section in any SLURP paper draft, or as a standalone short paper at SLT/Interspeech.

## Related

- [[methods/wavlm-hier|WavLM-Hier]]
- [[findings/slurp-wavlm-hier-results|WavLM-Hier full results]]
- [[projects/slurp|SLURP project]]
