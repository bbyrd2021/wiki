---
type: finding
title: "Exp5 — Detection-Steered Prompting (Qwen relays the detector, doesn't correct it)"
aliases: ["detection-steered prompting", "steered qwen", "exp5 steered", "detector-steered prompt"]
created: 2026-07-06
updated: 2026-07-06
sources:
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/qwen_infer.py"
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/vlm_io.py"
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/eval_qwen.py"
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/results_steered.json"
  - "ROAD_Reason/experiments/exp5_qwen_reasoning/results_nms_fixed.json"
  - "ROAD_Reason/experiments/exp6_detection_steered/eval.py"
  - "ROAD_Reason/experiments/exp6_detection_steered/results_detector_only.json"
tags: [finding, road-plusplus, exp5, exp6, vlm, qwen, 3d-retinanet, detection-steered, hard-label, f-map, approach-8]
status: complete
---

# Exp5 — Detection-Steered Prompting

**Question (Moradi update, 2026-07-02):** feed the frozen detector's per-box class predictions into the Qwen prompt as priors, so the VLM *verifies/refines* rather than classifying from scratch. Does that lift the relational heads (action / location / duplex) where scene context should let the VLM correct the detector?

**Answer:** Steering helps Qwen a lot relative to classifying from scratch — but steered Qwen still **loses to the detector on every head**. On this frozen top-40 setup Qwen is a *lossy relay* of the detector's predictions, not a corrector. The relational heads it was supposed to fix show the **widest** detector-vs-Qwen gaps.

See [[directions/vlm-reasoning-layer|VLM Reasoning Layer (Approach 8)]] for the overall staged plan; this is a prompt-side variant of its Stage 1, run before the trained-fusion step.

---

## Setup

- Frozen [[methods/3d-retinanet|3D-RetinaNet]] (17.76% all-anchor agent f-mAP) emits top-K boxes per frame; the top 40 by score are drawn as numbered boxes and classified by frozen **Qwen2.5-VL-7B-Instruct**.
- **Steered prompt** (`vlm_io.format_priors` + `build_prompt(..., priors=)`): the detector's top-1 agent/action/location + sigmoid confidence (read off its flat 184-dim logits) are shown per box, with explicit "verify, don't copy" instructions. Cached separately under `cache/qwen_steered/`; the plain prompt is byte-identical to the zero-shot run, so that cache stays the control.
- **Detector-only control** (`exp6/eval.py --detector-only`): the detector's own per-class logits scored directly over the *same* top-40 boxes. This is what "pure copying" of the priors would score.
- All three columns use the baseline evaluator at IoU=0.5 over the same 9,504 val frames — an apples-to-apples comparison on identical boxes.

**These are NOT the 17.76% all-anchor number.** All three columns score the top-40 post-NMS row set, which trades recall for legibility; treat them only against each other.

---

## Result (f-mAP, same boxes and evaluator)

| head | plain 0-shot | steered | detector-only |
|------|------:|------:|------:|
| agentness | 31.38 | 31.38 | 31.04 |
| agent | 5.60 | 12.88 | **14.62** |
| action | 1.15 | 7.62 | **12.16** |
| loc | 0.82 | 9.10 | **11.76** |
| duplex | 0.80 | 6.09 | **10.33** |
| triplet | 0.02 | 2.93 | **7.48** |

- **Plain → steered:** every head lifts +5 to +8. The detector's predictions are a genuinely useful prompt signal.
- **Steered → detector-only:** the detector wins on every head. The gap is widest on the relational heads Qwen was meant to improve — triplet 7.48 vs 2.93, duplex 10.33 vs 6.09. Steering pulls Qwen partway back toward the detector it was handed, but never past it.
- **`agentness` unchanged** across plain/steered confirms the columns score the same boxes and only the language differs (agentness is the box score, untouched by the prompt).

---

## Why — and the caveat

Part of the detector's advantage is a **scoring-format artifact, not semantics.** f-mAP rewards ranking many boxes per class by graded confidence:

- The detector emits **continuous per-class logits** — a rich per-class ranking.
- Qwen emits **hard labels** — every box it calls `Car` shares one box-score, a flat ranking that AP penalizes.

So the steered-vs-detector gap conflates two effects that this run can't separate: (a) Qwen's zero-shot Waymo semantics are weak, and (b) hard labels are a poor fit for AP-style evaluation. Both push the same direction.

---

## What it argues for

**Hard-label prompting is the wrong way to inject the VLM** — because the VLM's word is the *final* prediction, its output must pass through a one-class-per-box commitment, and that commitment discards the detector's continuous logits (most of what f-mAP scores). Prompting merges detector and VLM **in text, with the VLM deciding**; fusion merges them **in vector space, with a trained head deciding**. Two failure modes of prompting, and the fusion fixes each by construction:

| Prompting fails because… | Fusion fixes it by… |
|--------------------------|---------------------|
| **Representational** — a hard label collapses graded confidence into a flat per-class ranking (AP punishes this) | keeping the detector's **continuous logits** as a fusion input → ranking preserved; also gives the floor guarantee below |
| **Epistemic** — the VLM has the final say with no learned trust, so it overrides the detector even where the detector was right | turning the VLM output into an **embedding** (structured fields + rationale) that a **trained** head weighs against the logits → learns per-head, per-situation trust |

The [[directions/vlm-reasoning-layer|Approach 8]] Stage-1 fusion (scaffolded in `exp6_detection_steered/`) does exactly this: fuse Qwen's structured fields + SigLIP rationale embedding with the detector's continuous logits and train a small head. Its **floor is the detector's own 14.6 agent** (not the degraded 12.9) — the head is zero-initialized to pass detector logits through unchanged, so language can only *move* a prediction where it beats the detector.

Note the scoring-format caveat above is not a confound to apologize for — it *is* the mechanism, and the downstream embedding is what removes it (continuous representation on both sides). The caveat and this justification are the same point told twice.

**Next measurable question:** does trained fusion beat the detector-only control on the relational heads — the place hard-label prompting couldn't?

---

## Related

- [[directions/vlm-reasoning-layer|VLM Reasoning Layer over 3D-RetinaNet (Approach 8)]] — the staged plan this feeds
- [[methods/3d-retinanet|3D-RetinaNet]] — the frozen detector supplying boxes, logits, and the control
- [[findings/exp2-series-narrative|Exp2 Series Narrative]] — the image-level post-mortem that motivated keeping a spatiotemporal detector
- [[findings/exp1-vs-retinanet-baseline|Exp1 vs RetinaNet Baseline]] — earlier Qwen-vs-detector comparison (on GT boxes), same pattern on duplex/triplet
