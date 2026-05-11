---
type: dataset
title: "BDD-X — Berkeley DeepDrive eXplanation"
aliases: ["BDD-X", "Berkeley DeepDrive Explanations"]
created: 2026-04-10
updated: 2026-04-10
sources:
  - "ROAD_Reason/datasets/BDD-X/BDD-X-Annotations_v1.csv"
  - "ROAD_Reason/datasets/BDD-X/README.md"
tags: [dataset, captioning, explanation, bdd100k, approach3, road-reason]
status: complete
clips: 7000
frames: 8400000
primary_units: 26538
annotation_format: csv
---

# BDD-X — Berkeley DeepDrive eXplanation

**Kim et al., UC Berkeley** | ECCV 2018 | Exp3 in Approach 3 Qwen multi-task series

## What It Is

A natural language explanation dataset built on top of BDD100K dashcam videos. Human annotators (acting as driving instructors) describe what the ego vehicle is doing and *why* — producing action+justification pairs for every behavior change in the video.

## Verified Statistics (from actual CSV)

| Property | Value |
|----------|-------|
| Unique videos | 7,000 |
| CSV rows (annotator submissions) | 12,997 (≈ 1.86 annotators per video) |
| Total annotated action segments | 26,538 |
| Average actions per video | ~3.8 (across all annotators) |
| Mean justification length | 7.9 words |
| Action segment duration (mean) | 8.8 seconds |
| Total hours | 77+ |
| Total frames | ~8.4M |
| Train split | 5,588 videos |
| Val split | 698 videos |
| Test split | 698 videos |

Note: 12,997 CSV rows for 7,000 unique videos indicates ~1.86 annotators per video on average — multiple annotators were used for coverage.

## Annotation Format

**CSV — wide format.** One row per annotator submission per video. Up to 15 action annotations per row.

**Columns (61 total):**
- `Input.Video` — S3 URL to BDD100K video clip
- For each action slot N (1–15):
  - `Answer.Nstart` — start timestamp (seconds)
  - `Answer.Nend` — end timestamp (seconds)
  - `Answer.Naction` — what the vehicle is doing (e.g. *"The car slows down"*)
  - `Answer.Njustification` — why (e.g. *"because the light has turned red"*)

**Sample pairs:**
```
ACTION:  The car accelerates
JUSTIFY: because the light has turned green.

ACTION:  The car slows down
JUSTIFY: because it is approaching an intersection and the light is red.

ACTION:  The car stops
JUSTIFY: because a pedestrian is crossing the street.
```

## What It Does NOT Provide

- No frame-level annotations (action-level only)
- No trajectory / waypoint data
- No ego speed / steering / CAN bus data
- No explicit traffic light or leading vehicle annotations
- No weather or road type labels
- Video files: S3 URLs in CSV — these may be expired. Video data requires BDD100K download separately.

## Top Action Verbs (from actual CSV)

`accelerates` (1,988) · `slows` (2,632) · `stopped` (2,520) · `moving` · `driving` · `turns` · `stops` · `moves`

## Splits File Format

`train.txt`, `val.txt`, `test.txt` — one video filename per line (no header).

## Local Paths

| File | Location |
|------|----------|
| Annotations CSV | `/data/datasets/BDD-X/BDD-X-Annotations_v1.csv` |
| Split files | `/data/datasets/BDD-X/{train,val,test}.txt` |
| Repo clone | `/data/repos/ROAD_Reason/datasets/BDD-X/` |

## Relevance to Approach 3

BDD-X is **Experiment 3** in the Approach 3 Qwen2.5-VL multi-task series (after ROAD-R detection experiments). The task: fine-tune Qwen2.5-VL on BDD-X to produce action + justification captions for driving clips. Evaluated with BLEU-4 / CIDEr / METEOR.

| Caption field | Purpose |
|---|---|
| `Answer.Naction` | What the vehicle is doing |
| `Answer.Njustification` | Why (causal explanation) |

Smaller than CoVLA (26K action segments vs 6M frames) but manually written — higher quality reasoning signal. Part of the Approach 3 multi-task baseline alongside CoVLA and ROAD-R.

## License

UC Berkeley custom license. Non-commercial research and educational use permitted. Commercial use requires contact with UC Berkeley Office of Technology Licensing.

## Citation

```bibtex
@article{kim2018textual,
  title={Textual Explanations for Self-Driving Vehicles},
  author={Kim, Jinkyu and Rohrbach, Anna and Darrell, Trevor and Canny, John and Akata, Zeynep},
  journal={ECCV},
  year={2018}
}
```

## Related

- [[papers/covla-2025|CoVLA]] — Exp4 in Approach 3 series (larger scale, frame-level)
- [[directions/qwen-multitask-finetuning|Approach 3: Qwen Multi-Task]] — where BDD-X fits
- [[datasets/bdd100k|BDD100K]] — base video dataset BDD-X is built on
