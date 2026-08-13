---
type: finding
title: "Exp8 — Joint-LoRA Fusion Ties the Detector (language adds nothing the logits don't have)"
aliases: ["exp8 fusion results", "joint lora fusion", "stage 2 gate result", "joint fused eval"]
created: 2026-08-13
updated: 2026-08-13
sources:
  - "ROAD_Reason/experiments/exp8_joint_lora/logs/train.log"
  - "ROAD_Reason/experiments/exp8_joint_lora/cache/joint_sft_stats.json"
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/results_joint_lora.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_joint_lora_ep001.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_joint_lora_ep005.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_joint_lora_ep010.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_joint_lora_ep020.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_detector_only.json"
tags: [finding, road-plusplus, exp8, exp6, vlm, qwen, lora, fusion, f-map, approach-8, stage-2]
status: complete
---

# Exp8 — Joint-LoRA Fusion Ties the Detector

**Question ([[directions/vlm-reasoning-layer|Approach 8]] Stage 2, after the sequential BDD-X leg failed its gate):** does *joint interleaved* LoRA tuning of the VLM — one sample per corpus per step over ROAD SFT (9,596) + BDD-X (4,339) + CoVLA (2,250) — produce language outputs that finally lift the trained fusion past the detector-only control?

**Answer: no.** The joint leg is the best VLM variant so far — it beats the sequential BDD-X leg on most heads at matched epochs and roughly doubles several VLM-only scores over zero-shot — but the **fused number still ties the detector-only control** (best agent 14.69 vs control 14.62, relational heads all at or below control). The **zero-language ablation remains the best "fusion" of all** (agent 14.76 at ep020). Three VLM variants (zero-shot, BDD-X-tuned, joint-tuned) have now moved the fused agent f-mAP within a ±0.15-pp band around the control: on this architecture, **the language pathway carries ≈ no signal the detector logits don't already have.**

All numbers: [[datasets/road-plusplus|ROAD-Waymo]] val, frame-level f-mAP at IoU=0.5, top-40 post-NMS boxes — the same rows for every column (not the 17.76% all-anchor protocol).

---

## Setup

- **exp8 joint LoRA** (`exp8_joint_lora/`): Qwen2.5-VL-7B, joint interleaved SFT — 1 sample/corpus/step over ROAD structured-JSON, BDD-X explanation, CoVLA risk corpora. 28,788 steps / 3 epochs (~28 h). **Epoch-1 checkpoint kept** (joint val loss 0.432; epochs 2/3 overfit at 0.463/0.481) → `merged_checkpoint_9596`. Supersedes the sequential BDD-X→CoVLA plan (Moradi, 2026-07-27) after the [[directions/vlm-reasoning-layer|exp7 BDD-X-only leg]] failed the gate.
- **Re-cache** (exp5 harness, plain prompt): train subset 2/12 shards (9,596 frames) + full val (9,504 frames) at ~57 s/frame on 2×A6000. A parser-salvage fix (`vlm_io._salvage_boxes`, 2026-08-09) recovers the joint model's occasionally structure-corrupted JSON; one `reparse_cache.py` pass reconciled pre/post-fix frames (19,099 files, +7,198 parsed boxes).
- **Fusion leg** (exp6 harness, unchanged architecture): SigLIP rationale embeddings + structured fields ⊕ detector logits → residual zero-init head, 20 epochs on the train-subset cache; evaluated at ep 1/5/10/20 against the archived controls.

## Result 1 — VLM-only (hard labels over the same boxes)

| head | zero-shot plain | joint plain | Δ |
|------|------:|------:|------:|
| agent | 5.60 | **7.07** | +1.47 |
| action | 1.15 | **3.58** | +2.43 |
| loc | 0.82 | **1.57** | +0.75 |
| duplex | 0.80 | **1.78** | +0.98 |
| triplet | 0.02 | **0.66** | +0.64 |

Joint tuning genuinely improved the VLM's ROAD-Waymo semantics — every head lifts, triplet from floor to measurable. (Still far under detection-steered prompting's 12.88 agent, and under every fused/control column — hard labels remain the wrong currency for AP, per [[findings/exp5-detection-steered-prompting|the exp5 finding]].)

## Result 2 — Trained fusion (the gate number)

Best epoch per variant, per head:

| head | detector-only control | zero-shot fused | BDD-X fused | **joint fused** | zero-lang ablation |
|------|------:|------:|------:|------:|------:|
| agent | 14.62 | 14.34 | 14.59 | **14.69** | **14.76** |
| action | 12.16 | 11.49 | 11.93 | 11.60 | 11.75 |
| loc | 11.76 | 11.66 | 11.84 | 11.77 | 12.05 |
| duplex | 10.33 | 10.13 | 10.01 | 10.24 | 10.09 |
| triplet | 7.48 | 7.01 | 6.76 | 6.94 | 6.92 |

- **Joint ≥ BDD-X leg** at matched epochs (agent 14.69 vs 14.59, duplex 10.24 vs 10.01, triplet 6.94 vs 6.76) — the interleaved recipe is the better tuning protocol, consistent with Result 1.
- **But no fused variant separates from the control.** Agent spans 14.34–14.76 across all five columns; action/duplex/triplet sit at or below the detector everywhere. The residual head's zero-init floor works as designed — and converges to ≈ identity.
- **The zero-language ablation wins on agent and loc.** A fusion head fed *zeroed language inputs* (pure detector-logit re-projection) edges out every language-carrying variant. Whatever the head is learning, it isn't coming from the VLM.

## Interpretation

The bottleneck is no longer VLM quality. Two tuning recipes moved VLM-only scores substantially (Result 1) while moving the fused number ~0.1 pp (Result 2); the language-input channel is where the signal dies. Candidate causes, not yet separated: (a) the hard-parsed JSON fields + one-sentence SigLIP rationale embedding are too lossy a language representation; (b) whatever the VLM knows about these boxes is already linearly present in the detector's 184-dim logits; (c) frame-level fusion over top-40 boxes has no headroom — the control itself is capped at 14.62 vs the detector's own 17.76 all-anchor score. Distinguishing (a) from (b) needs a richer language representation (e.g., VLM hidden states instead of parsed JSON) fed to the same head; (c) argues the comparison should move back toward the all-anchor protocol before more VLM investment.

## Related

- [[directions/vlm-reasoning-layer|VLM Reasoning Layer over 3D-RetinaNet (Approach 8)]] — the staged plan; this is the Stage-2 gate result
- [[findings/exp5-detection-steered-prompting|Exp5 Detection-Steered Prompting]] — the prompt-side result this fusion was built to fix; its closing question is answered (negatively) here
- [[methods/3d-retinanet|3D-RetinaNet]] — the frozen trunk and the control that won't be beaten
- [[datasets/bdd-x|BDD-X]] | [[datasets/covla|CoVLA]] | [[datasets/road-plusplus|ROAD++]] — the joint SFT corpora
