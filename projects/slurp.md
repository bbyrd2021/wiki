---
type: project
title: "SLURP — Spoken Language Understanding"
aliases: ["SLURP"]
created: 2026-04-07
updated: 2026-05-08
sources:
  - "SLURP/CLAUDE_INSTRUCTIONS.md"
  - "SLURP/E1_results_summary.md"
  - "SLURP/encoder_ablation_notes.md"
  - "SLURP/contrib/README.md"
  - "SLURP/contrib/run_whisper_hier.py"
  - "SLURP/logs/wavlm_large_hier_v3/"
  - "SLURP/logs/wavlm_large_flat_ablation/"
  - "SLURP/logs/text_oracle/"
  - "SLURP/logs/E1_frozen_ce/"
tags: [project, spoken-language, wavlm, slu, hierarchical]
status: complete
---

# SLURP

Spoken Language Understanding (SLU) research on the SLURP corpus (18 scenarios, 46 actions, ~16K utterances). Group project (Brandon Byrd, Ire Adetu, Miles Johnson, Nakiya Holliday, Nandi Hawkins) at NC A&T, COMP 651. Supervised by Dr. Ehsan.

**Repo:** `/data/repos/SLURP/` | **Hardware:** RTX A6000 (`cuda:0` / `cuda:1`)

The project's narrative arc is **collapse → diagnosis → fix**:
1. Reproduce the Phase 3 paper failure (frozen Wav2Vec2 collapses to majority class) under controlled, fully-logged conditions
2. Diagnose root cause (encoder pretraining objective + class imbalance + LR schedule)
3. Build [[methods/wavlm-hier|WavLM-Hier]] — WavLM-Large + 5-component novel architecture — that recovers to scenario F1w 0.877 / action F1w 0.833 / intent acc 0.820

## Headline numbers

| System | Sc F1w | Act F1w | Intent Acc | Status |
|---|---|---|---|---|
| [[findings/slurp-collapse-e1|E1: Frozen Wav2Vec2 + CE]] | 0.032 | 0.125 | — | Collapsed (failure baseline) |
| Phase 2 (SpeechBrain + HuBERT) | — | 0.73 (SLURP-F1) | 0.88 | Advisor's prior baseline |
| Wang et al. 2021 (arXiv:2111.02735) | — | — | 0.8938 | Public benchmark (must position against) |
| Text Oracle (RoBERTa + gold transcript) | 0.790 | 0.553 | 0.541 | NLU ceiling — see [[comparisons/slurp-audio-vs-text-oracle|comparison]] |
| WavLM-base-plus + Hier | 0.816 | 0.734 | 0.711 | Encoder ablation |
| WavLM-Large flat (no hier) | 0.874 | 0.823 | 0.796 | Hierarchy ablation |
| **WavLM-Large + Hier (v3)** | **0.877** | **0.833** | **0.820** | **Best** |

Surprising finding: **the audio model beats the text oracle on every metric** — see [[comparisons/slurp-audio-vs-text-oracle|Audio vs Text Oracle]]. Action F1w gap is +0.28; consistent with prosody carrying intent information that gold transcripts discard.

## Method

[[methods/wavlm-hier|WavLM-Hier]] (script: `contrib/run_whisper_hier.py`, name retained from original Whisper plan). Five novel components:

1. WavLM-Large encoder — replaces collapsed Wav2Vec2
2. Attention pooling — learned temporal query
3. Hierarchical conditioning — action head sees soft scenario probs
4. Curriculum teacher forcing — GT scenario (epoch 1) → predicted (epoch 15), linear
5. Ontology masking — 60 valid scenario→action pairs masked at inference

Hierarchy ablation (flat vs hier on WavLM-Large): +0.024 Intent Acc, marginal on per-task F1.
Encoder ablation (base-plus vs Large + hier): +0.109 Intent Acc — encoder choice dominates.

## Repo structure

```
SLURP/
├── run_e1.py                      # E1: frozen Wav2Vec2 + CE (failure baseline)
├── run_e5.py                      # E5: partial unfreeze + weighted CE (also collapsed)
├── run_e8.py                      # E8: full fine-tune + weighted CE
├── run_hubert.py                  # E8 variant w/ HuBERT
├── run_wavlm.py                   # E8 variant w/ WavLM-base-plus
├── contrib/                       # Novel hierarchical method
│   ├── build_ontology.py          # Extract valid scenario→action pairs (one-time)
│   ├── run_whisper_hier.py        # Main novel method (encoder swappable)
│   ├── run_text_oracle.py         # RoBERTa on gold transcripts (NLU ceiling)
│   ├── run_ablation_flat.py       # Flat (no hier) ablation
│   ├── eval_test_hier.py          # Test-set evaluator for hier checkpoints
│   └── ontology/                  # Built artefacts (gitignored)
├── logs/                          # All run outputs
│   ├── E1_frozen_ce/              # E1 collapse evidence
│   ├── E5_partial_weighted_ce/    # E5 (still collapsed)
│   ├── E8_*/                      # Full fine-tune runs
│   ├── text_oracle/               # NLU ceiling
│   ├── whisper_hier/              # WavLM-base-plus + hier
│   ├── wavlm_large_hier_v{1,2,3}/ # WavLM-Large + hier (v3 = best)
│   └── wavlm_large_flat_ablation/ # WavLM-Large, hierarchy off
├── legacy/                        # COMP 651 phase reports + Nandi's notebook
├── CLAUDE_INSTRUCTIONS.md         # Colab-form runbook for Phase 1 replication
├── E1_results_summary.md          # E1 failure analysis (frozen W2V2)
├── encoder_ablation_notes.md      # Why encoder choice matters
└── contrib/README.md              # Novel method overview
```

## Publishing readiness — ~60% to submit

**What is in place**:
- Complete narrative arc (collapse → diagnosis → fix) with reproducible logs, configs, predictions
- Strong ablations: encoder size (base-plus vs Large), hierarchy on/off, text NLU ceiling
- Headline-quality surprise: audio beats text oracle (+0.28 action F1w) — publishable on its own
- Per-class F1 breakdown identifying `general` as the single weak scenario class

**Gaps blocking submission**:

1. **Wang et al. 2021 still beats this work on intent accuracy** (89.38% vs 82.0%). Likely cause: their result includes slot-filling supervision; this work has no slot head. Either add slot filling, or reframe headline metric to F1w / macro F1 (more meaningful on imbalanced data).
2. **No slot-filling head**. Phase 2's SLURP-F1 0.73 measures slot+intent jointly; current work measures only scenario+action+intent classification. Adding a CRF or pointer head over the encoder is the single highest-leverage gap to close.
3. **Single seed (42)**. Reviewers expect ≥3 seeds with std deviations. ~14h × 3 reruns = ~42h additional A6000 time.
4. **Hierarchy gain is marginal** on per-task metrics (+0.003 / +0.010 F1w). Strongest framing is the +0.024 Intent Acc gain. May want stronger ablations (curriculum schedule, ontology mask isolation).
5. **No paper draft exists**. Only legacy COMP 651 phase reports in `legacy/`.
6. **Focal loss / loss function ablation not run** despite being in advisor's plan.
7. **wavlm_large_hier_v2 has no final test eval** — partial run, superseded by v3.

**Realistic path to submission** (~3–4 weeks):
- Week 1: Add slot-filling head, run on WavLM-Large + Hier base (target SLURP-F1 ≥ 0.74 to match Phase 2)
- Week 2: 3-seed reruns of v3 + flat ablation; add focal-loss ablation
- Week 3: Draft paper around the audio-beats-text-oracle headline + collapse-to-fix narrative
- Week 4: Internal review + submission polish

**Realistic target venues**:
- ICASSP 2026 (deadline typically October — already past for 2026 cycle)
- Interspeech 2026 (deadline typically March — past)
- IEEE SLT 2026 (deadline typically July) — best fit
- Workshop track or short paper at any of the above

## Methodology parallels to AV research

The encoder-collapse pattern documented here applies cross-domain. See [[concepts/encoder-collapse|Encoder Collapse]] for the generalised diagnostic. Risk areas:

- **[[datasets/road-plusplus|ROAD++]] action classes** — extreme long-tail (Mov 1.84M ↔ Push 5K). Frozen backbones risk majority-class collapse.
- **[[datasets/jaad|JAAD]] crossing labels** — 70/30 split mild but enough.
- VLM zero-shot detection with frozen CLIP backbones.

The diagnostic checklist (per-class prediction counts at epoch 1, per-class F1 = 0 for >50% of classes, gradient norm with no metric improvement) transfers directly.

## Related

### Findings
- [[findings/slurp-collapse-e1|SLURP E1 Collapse]] — frozen Wav2Vec2 failure baseline
- [[findings/slurp-wavlm-hier-results|WavLM-Hier Full Results]] — best run + ablations + per-class

### Methods
- [[methods/wavlm-hier|WavLM-Hier]] — architecture spec

### Comparisons
- [[comparisons/slurp-audio-vs-text-oracle|Audio vs Text Oracle]] — surprising audio-beats-text result

### Concepts
- [[concepts/encoder-collapse|Encoder Collapse]] — generalised failure pattern
- [[concepts/domain-shift|Domain Shift]] — imbalance as a form of distribution shift
