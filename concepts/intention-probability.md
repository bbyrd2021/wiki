---
type: concept
title: "Intention Probability (PIE)"
aliases: ["intention_prob", "intent score", "intention probability"]
created: 2026-04-07
updated: 2026-04-07
sources:
  - "PedestrianIntent++/SYNTHESIS.md"
  - "PedestrianIntent++/PIE_analysis/pie_stats.json"
tags: [concept, intention-probability, pie, uncertainty]
status: complete
---

# Intention Probability

PIE-exclusive continuous measure of crossing intent. A scalar per pedestrian track representing crowd-sourced human judgment of crossing likelihood.

## How It Is Derived

1. Each pedestrian track was shown to **up to 30 human subjects**
2. Humans watched the clip from `exp_start_point` → `critical_point` — ending **before** any crossing occurs
3. Each human answered: "Will this person cross in front of the vehicle? Yes/No"
4. `intention_prob` = fraction who answered "Yes"

**Key property:** `intention_prob` is a true *intent* measure — it captures human uncertainty before the outcome. A pedestrian with `intention_prob=0.9` who ultimately does not cross (e.g., signal turns red) is correctly labeled — the annotator saw intent before the outcome.

## Distribution (1,842 PIE pedestrians)

| Bucket | Count | % |
|--------|-------|---|
| 0.0 – 0.1 | 132 | 7.2% |
| 0.1 – 0.3 | 177 | 9.6% |
| 0.3 – 0.5 | 113 | 6.1% |
| 0.5 – 0.7 | 167 | 9.1% |
| 0.7 – 0.9 | 470 | 25.5% |
| **0.9 – 1.0** | **783** | **42.5%** |

**Mean = 0.712, Median = 0.850** — strongly bimodal. Most pedestrians are unambiguously judged.

See [[findings/pie-intention-bimodality|PIE Intention Bimodality]].

## As Regression Target

Most papers binarize `intention_prob` at a threshold (e.g., ≥0.5) and train a classifier. This discards calibration information.

**Research opportunity:** Train as a regression target to preserve uncertainty. A prediction of `intention_prob=0.75` is qualitatively different from `0.95` — the former warrants a cautious response, the latter confident braking. See [[directions/uncertainty-aware-intent|Uncertainty-Aware Intent]].

## Temporal Prediction Horizon

`critical_point` → `crossing_point` gap: **median 30 frames = 1 second**. This is the window within which the AV must respond. The model must predict intent at critical_point from pre-crossing cues alone.

## Comparison to JAAD

JAAD has no `intention_prob`. Binary crossing outcome is assigned post-hoc by the annotator (not crowd-sourced), with no uncertainty information.

## Related

- [[datasets/pie|PIE Dataset]] | [[concepts/crossing-intent|Crossing Intent]]
- [[directions/uncertainty-aware-intent|Uncertainty-Aware Intent Direction]]
- [[findings/pie-intention-bimodality|PIE Intention Bimodality]]
