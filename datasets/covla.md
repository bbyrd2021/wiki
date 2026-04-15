---
type: dataset
title: "CoVLA — Comprehensive Vision-Language-Action Dataset"
aliases: ["CoVLA", "CoVLA-Dataset", "CoVLA-Mini"]
created: 2026-04-10
updated: 2026-04-10
sources:
  - "ROAD_Reason/papers/CoVLA_paper.pdf"
tags: [dataset, captioning, explanation, trajectory, can-bus, stage1-pretraining, road-reason]
status: complete
clips: 10000
frames: 6000000
primary_units: 6000000
annotation_format: jsonl
---

# CoVLA — Comprehensive Vision-Language-Action Dataset

**Arai et al., Turing Inc.** | arXiv:2408.10845 | Stage 1 pre-training source for Approach 3 causal head

## What It Is

A large-scale real-world driving dataset pairing video clips with **frame-level language captions** and **future trajectory annotations**. Collected in Tokyo, Japan across urban, highway, residential, and mountain roads. Designed to address the scale and annotation richness gap in VLA research.

## Verified Statistics (from CoVLA-Mini)

| Property | Value |
|----------|-------|
| Total clips | 10,000 (30s each, 20 FPS) |
| Total frames | 6,000,000 |
| Total hours | 83.3h |
| Mini subset | 50 scenes × 600 frames = 30,000 frames |
| Frames per scene | 600 |
| Camera resolution | 1928×1208 (H.265) |
| Frame rate | 20 FPS |
| Location | Tokyo, Japan |
| Train split | 7,000 scenes (70%) |
| Val split | 1,500 scenes (15%) |
| Test split | 1,500 scenes (15%) |

**Weather distribution (mini, 50 scenes):** sunny 25 · cloudy 21 · rainy 4  
**Road type distribution (mini):** wide road 33 · narrow road 17

## File Structure

Each scene has four JSONL files (concatenated JSON objects — one object per frame, no newline delimiters):

```
CoVLA/
├── index.csv                     ← flat index: video_id × frame_id → all paths
├── metadata.json                 ← {"image_size": [1928, 1208], "frequency": 20}
├── captions/<scene>.jsonl        ← per-frame captions + scene metadata
├── states/<scene>.jsonl          ← per-frame ego state + trajectory + camera matrices
├── front_car/<scene>.jsonl       ← per-frame leading vehicle detection
├── traffic_lights/<scene>.jsonl  ← per-frame traffic light detections
├── images/<scene>/NNNN.png       ← extracted frames
├── video_samples/<scene>.mp4     ← video clips
└── videollama2/                  ← (full dataset only)
```

## Annotation Schema (Verified)

### captions/*.jsonl

| Field | Type | Description |
|-------|------|-------------|
| `plain_caption` | str | Behavior description: what ego is doing + scene objects present |
| `rich_caption` | str | Extended: plain + weather + road type + risk sentence concatenated |
| `risk` | str | Why the driver should be careful (→ W reason mode) |
| `risk_correct` | bool | Whether risk description is factually correct |
| `risk_yes_rate` | float | Model confidence in risk description |
| `weather` | str | Weather label (sunny / cloudy / rainy / heavy rain / night) |
| `weather_rate` | float | Confidence in weather label |
| `road` | str | Road type (wide road / narrow road) |
| `road_rate` | float | Confidence in road type label |
| `is_tunnel` | bool | In tunnel |
| `is_highway` | bool | On highway |
| `has_pedestrian` | bool | Pedestrian present |
| `has_carrier_car` | float | Carrier vehicle probability |

**Sample `plain_caption`:**
> *"The ego vehicle is moving at a moderate speed and turning left. There is a traffic light near the ego vehicle displaying a green signal."*

**Sample `risk`:**
> *"to pay attention to the traffic light and other vehicles on the road. The driver should also be cautious of the wet road conditions due to the rain"*

### states/*.jsonl

| Field | Type | Description |
|-------|------|-------------|
| `ego_state.vEgo` | float | Ego vehicle speed (m/s) |
| `ego_state.vEgoRaw` | float | Raw ego speed |
| `ego_state.aEgo` | float | Ego acceleration |
| `ego_state.steeringAngleDeg` | float | Steering angle (degrees) |
| `ego_state.steeringTorque` | float | Steering torque |
| `ego_state.brake` | float | Brake pressure |
| `ego_state.brakePressed` | bool | Brake active |
| `ego_state.gas` | float | Gas pedal |
| `ego_state.gasPressed` | bool | Gas active |
| `ego_state.gearShifter` | str | Gear (drive/reverse/park) |
| `ego_state.leftBlinker` | bool | Left turn signal |
| `ego_state.rightBlinker` | bool | Right turn signal |
| `ego_state.orientations_calib/ecef/ned` | list[3] | Orientation vectors |
| `ego_state.positions_ecef` | list[3] | ECEF GPS position |
| `ego_state.velocities_calib/ecef` | list[3] | Velocity vectors |
| `ego_state.accelerations_calib/device` | list[3] | Acceleration vectors |
| `ego_state.angular_velocities_calib/device` | list[3] | Angular velocity vectors |
| `ego_state.timestamp` | int | Unix timestamp (ms) |
| `trajectory` | list[60×3] | Future path in vehicle frame (x=forward, y=lateral, z=vertical) |
| `trajectory_count` | int | Always 60 |
| `extrinsic_matrix` | list[4×4] | Camera extrinsic |
| `intrinsic_matrix` | list[3×3] | Camera intrinsic (fx=fy=2648, cx=964, cy=604) |
| `image_path` | str | Relative path to PNG frame |
| `frame_id` | int | Frame index within scene |

**Trajectory:** 60 3D points in vehicle frame, covering ~3 seconds at 20 FPS. First point is always `[0, 0, 0]` (current position).

### front_car/*.jsonl

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame index |
| `has_lead` | bool | Leading vehicle detected |
| `lead_prob` | float | Detection confidence |
| `lead_x` | float\|null | Longitudinal distance (m) |
| `lead_y` | float\|null | Lateral offset (m) |
| `lead_speed_kmh` | float\|null | Lead vehicle speed (km/h) |
| `lead_a` | float\|null | Lead vehicle acceleration |

### traffic_lights/*.jsonl

Each object is a **list** of detections for that frame (empty list if no TLs):

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | TL index |
| `class` | str | Signal state: green / red / amber |
| `bbox` | list[4] | [x1, y1, x2, y2] in pixels |

## Caption Generation Pipeline

1. **Rule-based layer** — extracts factual elements: ego motion (CAN/GNSS), traffic lights (OpenLenda-s detector), leading vehicles (radar+camera fusion), weather, road type
2. **VLM layer** — rule-based output as factual constraints to prompt VideoLLaMA2-7B (60-frame window, first+last frames sampled) for richer natural language; suppresses hallucinations
3. **Output** — 100,000 VLM-generated + 6,000,000 rule-augmented captions

## Relevance to Approach 3

| Field | DSDAG node | Supervision signal |
|-------|------------|--------------------|
| `plain_caption` | Y (action node) | Grounds Y — what action is occurring |
| `risk` | W (reason mode) | Grounds W — causal origin behind action |

CoVLA provides scale (6M frames) for Stage 1 pre-training. Used alongside BDD-X, which provides higher-quality human-annotated reasoning at smaller scale (26K segments).

**Limitation:** Tokyo collection — different traffic rules, signage (Japanese), and road geometry from ROAD-Waymo (US). W must generalize across this domain gap.

## Local Paths

| Content | Path |
|---------|------|
| Mini dataset | `/data/datasets/CoVLA/mini/` |
| Full dataset | `/data/datasets/CoVLA/full/` (download pending) |

## License

Turing Inc. non-commercial research license. No redistribution.

## Citation

```bibtex
@article{arai2024covla,
  title={CoVLA: Comprehensive Vision-Language-Action Dataset for Autonomous Driving},
  author={Arai, Hidehisa and others},
  journal={arXiv preprint arXiv:2408.10845},
  year={2024}
}
```

## Related

- [[papers/covla-2025|CoVLA paper]] — full paper summary with CoVLA-Agent results
- [[datasets/bdd-x|BDD-X]] — complementary Stage 1 source (smaller, manual, US driving)
- [[methods/multimodal-causal-driving|Multimodal Causal Driving Model]] — Stage 1 training context
- [[directions/constrained-vlm-reasoning|Approach 3]] — where CoVLA fits
