---
type: comparison
title: "BDD-X vs CoVLA — Stage 1 Pre-training Sources"
aliases: ["BDD-X vs CoVLA"]
created: 2026-04-10
updated: 2026-04-10
sources:
  - "ROAD_Reason/datasets/dataset_comparison.csv"
  - "ROAD_Reason/datasets/explore_annotations.py"
tags: [comparison, dataset, bdd-x, covla, stage1-pretraining, road-reason]
status: complete
---

# BDD-X vs CoVLA — Stage 1 Pre-training Sources

Both datasets serve as Stage 1 pre-training sources for the causal head in Approach 3. They are complementary rather than redundant: BDD-X provides **manual quality** at small scale; CoVLA provides **automated coverage** at large scale. Together they ground both nodes of the DSDAG causal structure in natural language before any ROAD-Waymo fine-tuning.

---

## Side-by-Side Comparison

| Property                      | BDD-X                                                             | CoVLA                                                                     |
| ----------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Full name**                 | Berkeley DeepDrive eXplanation                                    | Comprehensive Vision-Language-Action                                      |
| **Authors**                   | Kim et al. (UC Berkeley)                                          | Arai et al. (Turing Inc.)                                                 |
| **Year**                      | 2018                                                              | 2025                                                                      |
| **Venue**                     | ECCV 2018                                                         | arXiv:2408.10845                                                          |
| **Collection region**         | USA (diverse)                                                     | Tokyo, Japan (urban, highway, residential, mountain)                      |
| **Total clips / scenes**      | 7,000 unique videos                                               | 10,000 scenes (30s each)                                                  |
| **Total frames**              | ~8.4M                                                             | 6,000,000                                                                 |
| **Total hours**               | 77+                                                               | 83.3                                                                      |
| **Camera resolution**         | ~720p (BDD100K standard)                                          | 1928×1208 (H.265, 20 FPS)                                                 |
| **Annotation granularity**    | Action-level (temporal segment)                                   | Frame-level                                                               |
| **Total annotated instances** | 26,538 action segments                                            | 6,000,000 frame captions                                                  |
| **Annotation format**         | CSV (wide — 1 row per annotator per video, up to 15 action slots) | JSONL (concatenated JSON objects, one file per scene per modality)        |
| **Caption generation**        | Manual (human annotators — driving instructors)                   | Auto (rule-based + VideoLLaMA2-7B; hallucination-mitigated)               |
| **Action / behavior field**   | `Answer.Naction` — free-text action description                   | `plain_caption` — behavior description + present objects                  |
| **Reason / risk field**       | `Answer.Njustification` — why the vehicle acted                   | `risk` (extracted from `rich_caption`) — why the driver should be careful |
| **Mean reason length**        | 7.9 words                                                         | Varies (sentence-level)                                                   |
| **Trajectory data**           | No                                                                | Yes — 60×3 pts in vehicle frame, ~3s at 20 FPS                            |
| **Ego speed**                 | No (BDD100K has it separately)                                    | Yes — `vEgo`, `vEgoRaw` (CAN bus)                                         |
| **Steering angle**            | No                                                                | Yes — `steeringAngleDeg` (CAN bus)                                        |
| **Brake / gas**               | No                                                                | Yes — `brake`, `brakePressed`, `gas`, `gasPressed`                        |
| **Traffic light annotations** | No                                                                | Yes — `class` (green/red/amber), `bbox` per frame                         |
| **Leading vehicle**           | No                                                                | Yes — `has_lead`, `lead_prob`, `lead_x/y`, `lead_speed_kmh`               |
| **Weather label**             | No (implicit in video)                                            | Yes — `weather` field with confidence rate                                |
| **Road type label**           | No (implicit in video)                                            | Yes — `road` field (wide/narrow) with confidence rate                     |
| **Train split**               | 5,588 videos                                                      | 7,000 scenes (70%)                                                        |
| **Val split**                 | 698 videos                                                        | 1,500 scenes (15%)                                                        |
| **Test split**                | 698 videos                                                        | 1,500 scenes (15%)                                                        |
| **License**                   | UC Berkeley — non-commercial research                             | Turing Inc. — non-commercial, no redistribution                           |
| **Video availability**        | S3 URLs in CSV (may be expired); BDD100K needed separately        | Included in HuggingFace download                                          |
| **Access**                    | GitHub (annotations) + Google Drive                               | HuggingFace (gated — requires accepted terms)                             |

---

## DSDAG Relevance

| Field | BDD-X source | CoVLA source | DSDAG node |
|-------|-------------|--------------|------------|
| **Action / behavior** | `Answer.Naction` | `plain_caption` | **Y** — what action is occurring |
| **Reason / risk** | `Answer.Njustification` | `risk` | **W** — causal origin behind the action |

Both datasets provide supervision for the same two DSDAG nodes. The difference is quality vs. scale:

- **BDD-X justifications** are written by human driving instructors in complete sentences with explicit causal connectives (*"because the light has turned red"*). Higher-quality W supervision.
- **CoVLA risk field** is VLM-generated with rule-based grounding. More varied phrasing, frame-level granularity, 230× more instances.

---

## Key Contrasts for Stage 1 Training

**Scale:** CoVLA has ~226× more captioned instances (6M vs 26K). For training W to generalize across diverse causal contexts, CoVLA dominates by volume.

**Quality:** BDD-X justifications are human-authored and grammatically complete. CoVLA risk captions are auto-generated — hallucination is mitigated but not eliminated by rule-based grounding.

**Granularity:** BDD-X operates at action-segment level (average 8.8s per segment). CoVLA operates at every frame. For a model that must learn frame-level causal state, CoVLA provides denser supervision.

**Sensor richness:** CoVLA includes the full CAN bus and trajectory — allowing W to be conditioned on speed, steering, and future path during Stage 1. BDD-X provides language only.

**Geography:** BDD-X = US driving rules and road geometry. CoVLA = Tokyo driving (left-hand traffic, Japanese signage, narrow urban streets). Stage 1 sees both, which may help W generalize or may introduce conflicting priors — worth monitoring in ablation.

---

## Stage 1 Training Strategy

Use both jointly in Stage 1. Mini-batch should sample from each:

```
Stage 1 batch:
  - BDD-X: (video_clip, action_text, justification_text)
           → supervise Y with action_text, W with justification_text
  - CoVLA: (frame_image, plain_caption, risk_text)
           → supervise Y with plain_caption, W with risk_text
```

The combined Stage 1 objective forces the model to learn:
1. Y encodes *what is happening* — from diverse US + Tokyo driving contexts
2. W encodes *why* — from 26K high-quality human justifications + 6M auto-generated risk captions

After Stage 1, Stage 2 fine-tunes on ROAD-Waymo using label reconstruction proxy (no captions required). W is zero-shot transferred from Stage 1.

---

## Related

- [[datasets/bdd-x|BDD-X dataset page]] — verified statistics and annotation format
- [[datasets/covla|CoVLA dataset page]] — verified schema from CoVLA-Mini
- [[methods/multimodal-causal-driving|Multimodal Causal Driving Model]] — Stage 1 context
- [[directions/constrained-vlm-reasoning|Approach 3]] — training pipeline overview
- [[papers/covla-2025|CoVLA paper]] — CoVLA-Agent architecture and results
