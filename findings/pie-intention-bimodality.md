---
type: finding
title: "PIE Intention Probability — Bimodal Distribution"
aliases: ["PIE bimodal", "intention_prob distribution", "PIE bimodality"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/PIE_analysis/pie_stats.json"
  - "PedestrianIntent++/SYNTHESIS.md"
tags: [finding, pie, intention-probability, bimodal, distribution]
status: complete
---

# PIE Intention Probability — Bimodal Distribution

Verified from `PIE_analysis/pie_stats.json`, computed from all 1,842 PIE pedestrian tracks.

## Distribution

| Bucket | Count | % |
|--------|-------|---|
| 0.0 – 0.1 | 132 | 7.2% |
| 0.1 – 0.3 | 177 | 9.6% |
| 0.3 – 0.5 | 113 | 6.1% |
| 0.5 – 0.7 | 167 | 9.1% |
| 0.7 – 0.9 | 470 | 25.5% |
| **0.9 – 1.0** | **783** | **42.5%** |

**Mean = 0.712, Median = 0.850** — strongly bimodal.

## Interpretation

The distribution is bimodal because the task is designed to be mostly unambiguous:
- Most pedestrians are clearly going to cross (annotators near-unanimously say Yes) → high prob
- A smaller fraction clearly aren't going to cross → low prob
- The middle buckets (0.3–0.7) represent genuinely ambiguous cases — the "interesting" ones from a safety perspective

## Implication: Binarization Discards Information

Thresholding at 0.5 treats `intention_prob=0.51` identically to `0.99`. For AV safety:
- 0.51 → uncertain, slow down and observe
- 0.99 → commit to braking now

This is the core motivation for [[directions/uncertainty-aware-intent|Direction 1: Uncertainty-Aware Intent]].

## Intention vs. Outcome

| intention_prob bucket | Crossed | Not Crossed | Irrelevant |
|-----------------------|---------|-------------|------------|
| low (0–0.33) | 0% | 2% | **98%** |
| medium (0.33–0.67) | 7% | 43% | 50% |
| high (0.67–1.0) | 39% | 58% | 2% |

High-probability pedestrians (39% crossed) still frequently didn't cross — they **intended** to but were stopped by external factors (red light, car didn't yield). This confirms that **intention ≠ action**.

## Related

- [[concepts/intention-probability|Intention Probability]] | [[datasets/pie|PIE Dataset]]
- [[directions/uncertainty-aware-intent|Uncertainty-Aware Intent Direction]]
