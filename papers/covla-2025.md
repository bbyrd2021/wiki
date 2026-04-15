---
type: paper
title: "CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving"
aliases: ["CoVLA", "CoVLA-Dataset", "CoVLA-Agent"]
created: 2026-04-10
updated: 2026-04-10
sources:
  - "ROAD_Reason/papers/CoVLA_paper.pdf"
tags: [paper, dataset, vla, captioning, trajectory, stage1-pretraining, road-reason]
status: complete
authors: "Arai et al."
year: 2025
venue: "arXiv"
arxiv: "2408.10845"
datasets_used: []
---

# CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving

**Turing Inc.** | arXiv:2408.10845 | Used in Approach 3 Stage 1 pre-training

## What It Is

A large-scale real-world driving dataset pairing video clips with **frame-level language captions** (behavior + reasoning) and **future trajectory annotations**. Built to address the scale and annotation richness gap in VLA (Vision-Language-Action) research for autonomous driving.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Clips | 10,000 (30-second, 20 FPS) |
| Frames | 6,000,000 (6M) |
| Duration | 83.3 hours |
| Location | Tokyo, Japan (urban, highway, intersections, narrow streets, mountain roads) |
| Weather | Sunny, cloudy, rainy, heavy rain, nighttime |
| Camera | 1928×1208 front-facing, H.265 |
| Sensors | CAN bus (accelerator, brake, steering, gear, speed), GNSS, IMU |
| Caption count | 6,000,000 (auto-generated, multi-source) |
| Trajectory | GPS/IMU → Kalman filter → 60 coords (3s horizon, 2Hz) → 10 sampled points |

## Annotation Format (Verified from CoVLA-Mini)

CoVLA Mini: 50 scenes × 600 frames = 30,000 frames. Each scene has four JSONL files (one concatenated-JSON object per frame, no newline delimiters):

| File | Fields | Notes |
|------|--------|-------|
| `captions/*.jsonl` | `plain_caption`, `rich_caption`, `risk`, `risk_correct`, `risk_yes_rate`, `weather`, `weather_rate`, `road`, `road_rate`, `is_tunnel`, `is_highway`, `has_pedestrian`, `has_carrier_car` | Per-frame captions + scene metadata |
| `states/*.jsonl` | `ego_state` (vEgo, vEgoRaw, aEgo, steeringAngleDeg, steeringTorque, brake, brakePressed, gas, gasPressed, gearShifter, leftBlinker, rightBlinker, orientations, positions_ecef, velocities, accelerations, angular_velocities, timestamp), `trajectory` (60×3 pts), `extrinsic_matrix` (4×4), `intrinsic_matrix` (3×3), `image_path`, `frame_id` | Full CAN + IMU + camera calibration |
| `front_car/*.jsonl` | `frame_id`, `has_lead`, `lead_prob`, `lead_x`, `lead_y`, `lead_speed_kmh`, `lead_a` | Leading vehicle detection (radar+camera) |
| `traffic_lights/*.jsonl` | `index`, `class` (green/red/amber), `bbox` [x1,y1,x2,y2] | Per-TL per-frame (list of dicts) |

Additionally: `images/` (PNG frames), `video_samples/` (MP4s), `index.csv` (flat frame×modality index), `metadata.json` (image_size, frequency).

**Weather distribution (mini, 50 scenes):** sunny 25 · cloudy 21 · rainy 4  
**Road type distribution (mini):** wide road 33 · narrow road 17  
**Trajectory format:** 60×3 float array in vehicle frame (x=forward, y=lateral, z=vertical), ~3s at 20FPS

## Two Caption Types

This is the key distinction for Approach 3 use. Every frame gets both:

**`plain_caption`** — what the ego vehicle is doing and what is present in the scene:
> *"The ego vehicle is moving at a moderate speed and turning left. There is a traffic light near the ego vehicle displaying a green signal."*

**`risk`** (extracted from `rich_caption`) — why the driver should be careful; risk and causal context:
> *"to pay attention to the traffic light and other vehicles on the road. The driver should also be cautious of the wet road conditions due to the rain"*

The `rich_caption` field is the full extended description including weather, road type, and the risk sentence concatenated together.

**Generation pipeline:**
1. Rule-based captions first — vehicle motion (from CAN/GNSS), detected objects (traffic lights via OpenLenda-s, leading vehicles via radar+camera fusion), road type, weather
2. Rule-based output used as factual constraints/context to prompt VideoLLaMA2-7B (60-frame window, first+last frames sampled) — suppresses hallucinations while adding richness
3. Final captions = 100,000 VLM-generated + 6,000,000 rule-augmented

## CoVLA-Agent

Baseline VLM model trained on CoVLA-Dataset for two tasks: traffic scene description + trajectory prediction.

- **Architecture:** Llama-2 7B + CLIP ViT-L (224×224) vision encoder; visual features + ego speed → MLP projection → concatenated with text embedding → Llama-2
- **Trajectory output:** 10 (x,y,z) coordinates representing predicted path over 3 seconds
- **Training format:** LLaVA instruction tuning; 302,989 training samples

**Results (ADE / FDE):**

| Condition | ADE ↓ | FDE ↓ |
|-----------|-------|-------|
| Predicted caption | 0.955 | 2.239 |
| Ground truth caption | 0.814 | 1.655 |

Gap shows accurate scene description directly improves trajectory accuracy — grounding language to visual context matters.

## Relevance to Approach 3

CoVLA is one of two Stage 1 pre-training sources for the causal head (alongside BDD-X).

**The two caption types map directly onto the DSDAG node structure:**

| Field | DSDAG node | What it supervises |
|---|---|---|
| `plain_caption` | Y (action node) | What action is occurring; grounds Y in language |
| `risk` | W (reason mode) | *Why* the driver should be careful; the causal origin behind the action |

This mapping is the key reason CoVLA is more valuable than a generic caption dataset for Stage 1. The reasoning captions provide explicit language supervision for W — the latent variable that Approach 3 is specifically designed to learn. BDD-X provides human-annotated explanations but at much smaller scale (26K frames vs 6M). CoVLA provides automated reasoning captions at scale, with hallucination mitigation from the rule-based grounding.

**Scale comparison relevant to Stage 1:**

| Dataset | Frames | Caption type | Manual? |
|---------|--------|--------------|---------|
| BDD-X | 26K | Explanation (closed-form, ego-action triggered) | Yes |
| CoVLA | 6M | Behavior + reasoning (frame-level, auto) | No (rule+VLM) |

**Limitation for Stage 1:** Captions are Tokyo-collected — different traffic rules, signage (Japanese characters), and road geometry from ROAD-Waymo (US/Waymo). W must generalize across this distribution shift. The rule-based grounding mitigates hallucination but cannot fully compensate for geographic domain gap.

## Related

- [[methods/multimodal-causal-driving|Multimodal Causal Driving Model]] — Stage 1 training details
- [[directions/constrained-vlm-reasoning|Approach 3: Constrained VLM]] — where CoVLA fits in the training pipeline
- [[projects/road-reason|ROAD_Reason]]
