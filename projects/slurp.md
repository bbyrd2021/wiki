---
type: project
title: "SLURP — Spoken Language Understanding"
aliases: ["SLURP"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "SLURP/CLAUDE_INSTRUCTIONS.md"
  - "SLURP/E1_results_summary.md"
tags: [project, spoken-language, wav2vec2, methodology]
status: complete
---

# SLURP

Spoken Language Understanding (SLU) research. Investigates and fixes Wav2Vec2 frozen-encoder collapse on imbalanced SLU data.

**Repo:** `/data/repos/SLURP/` | **Environment:** Google Colab T4/A100

## Research Phases

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Baseline replication (frozen Wav2Vec2 + classifier head) | Complete |
| Phase 2 | Diagnose collapse on imbalanced classes | Complete |
| Phase 3 | Novel contributions: partial unfreezing, class-weighted loss, focal loss | In progress |

## Phase 1 Results

| Metric | Score |
|--------|-------|
| Intent Accuracy | 0.88 |
| SLURP-F1 | 0.73 |

## Methodology Parallels to AV Research

The encoder-collapse problem (frozen pretrained encoder underperforming on imbalanced downstream data) applies directly to:
- **EfficientPIE**: EfficientNet backbone on imbalanced crossing/non-crossing labels
- **ROAD_Reason**: VLM fine-tuning on imbalanced triplet label distributions

**Fix techniques applicable across domains:**
- Partial layer unfreezing (unfreeze last N transformer blocks)
- Class-weighted cross-entropy loss (re-weight by inverse class frequency)
- Focal loss (down-weight easy examples, focus on hard/rare ones)

## Related

- [[concepts/domain-shift|Domain Shift]] (imbalance as a form of distribution shift)
