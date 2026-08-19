---
type: finding
title: "Exp9 — The Attribution Grid: heads beat text, corrected t-norm binds, language co-training hurts"
aliases: ["exp9 results", "R1-R4 grid", "attribution grid", "joint heterogeneous results"]
created: 2026-08-19
updated: 2026-08-19
sources:
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/DESIGN.md"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r1.json"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r2_lam0.1.json"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r2_lam1.json"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r2_lam10.json"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r3.json"
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/results_r4.json"
tags: [finding, road-plusplus, exp9, t-norm, neuro-symbolic, joint-training, qwen, roi-heads, f-map, approach-8]
status: complete
---

# Exp9 — The Attribution Grid

Four runs, one architecture (Qwen2.5-VL ViT + LoRA, RoI-pooled flat-184 head on
frozen-detector boxes), each cell changing exactly one variable. Val f-mAP at
IoU=0.5 over the identical top-40 rows as the exp5/exp6/exp8 control family;
violations against the **corrected** constraint sets
([[findings/road-waymo-childs-mis-indexed|the childs bug]] — pre-2026-08-17
violation numbers are not comparable).

| | agent | action | loc | duplex | triplet | viol d% | viol t% |
|---|---|---|---|---|---|---|---|
| detector-only control | **14.62** | **12.16** | **11.76** | **10.33** | **7.48** | — | — |
| R1 — ROAD only, λ=0 | 14.51 | 8.78 | 10.18 | 7.47 | 4.73 | 17.7 | 62.9 |
| R2 — ROAD only, λ=0.1 | 13.13 | 8.32 | 9.51 | 7.61 | 4.60 | 0.61 | 45.1 |
| R2 — ROAD only, λ=1 | 12.54 | 8.10 | 9.24 | 7.87 | 4.34 | 0.06 | 30.3 |
| R2 — ROAD only, λ=10 | 12.46 | 7.87 | 8.92 | 7.94 | 4.59 | **0.0** | **7.6** |
| R3 — joint, λ=0 | 11.80 | 6.83 | 7.86 | 6.23 | 3.05 | 25.9 | 69.7 |
| R4 — joint, λ=10 | 11.33 | 7.05 | 8.04 | 6.38 | 3.45 | 0.0 | 11.2 |

(R3/R4 ep001 checkpoints score below ep003 on nearly all heads — the ROAD head
was still improving; this is not an overfit artifact. Joint runs used the
Moradi head-warmup: 1,000 ROAD-only heads-only cycles before the language legs
and LoRA joined.)

## The three attributable answers

**1. Structured heads beat text supervision, decisively (R1 vs exp8 VLM-only).**
The same per-box ROAD supervision that scored **7.07 agent** delivered as JSON
text through an LM loss scores **14.51** as RoI-heads with a task loss —
essentially matching the detector (14.62) on agent. The relational heads lag
the detector (action 8.78 vs 12.16; the compositional heads inherit it), which
localizes the remaining deficit in the *features*, not the supervision channel.

**2. The corrected t-norm binds — and is a trade, not a free lunch (R1→R2).**
First correctly-constrained loss-side t-norm run on this dataset. Violations
fall monotonically with λ (duplex 17.7→0.0%, triplet 62.9→7.6%) and duplex
f-mAP rises monotonically (7.47→7.94, +0.5); agent pays ~2 points; triplet is
flat. Replicated in the joint cells (R4 > R3 on duplex +0.15 and triplet
+0.40 with violations collapsed). Constraints buy compositional *coherence*
cheaply and compositional *accuracy* modestly; they do not substitute for
better action/location features. Contrast exp1's tainted "constraints are
inert" — with correct sets and a binding regime, they are not inert.

**3. Language co-training HURTS (R1→R3, R2→R4) — the sharpest negative yet.**
Adding the BDD-X/CoVLA LM legs (strict alternation, shared ViT LoRA, reduced
exposure via warmup) degraded every ROAD head: agent 14.51→11.80, duplex
7.47→6.23, triplet 4.73→3.05, and even constraint compliance worsened
(violations 17.7→25.9% duplex). The language gradients actively pulled the
shared visual representation away from what the detection head needs. **Reverse check (2026-08-19, all 795 held-out language val samples):** the interference is strongly asymmetric — R3/R4 keep most language capability (BDD-X val loss 0.99/0.98 vs exp8's language-only 0.87 vs base 2.39; CoVLA 0.57/0.62 vs 0.31 vs 2.05) while vision paid ~2.7 agent points. The shared representation resolved the objective competition in language's favor, at the vision task's expense. This
completes a three-way negative on language for this task: inference-time
fusion adds nothing ([[findings/exp8-joint-lora-fusion|exp8]]), text-delivered
supervision underperforms heads (point 1), and now training-time co-training
subtracts. Per Moradi's 2026-08-18 branch ("if not, we will change
direction"): the gate is not met and the direction changes.

## What it argues for

The one language mechanism *not* yet tested is Moradi's contrastive
phrase-head suggestion — text as **label-space geometry** (rare compositions
borrowing statistical strength from semantic neighbors) rather than as a
co-training signal or an inference-time input. It attacks exactly the deficit
this grid isolates (relational features + long-tail independence) and is
Tier-0 feasible with frozen [[papers/wang-2024-internvideo2|InternVideo2]]-clip
features. Fallback beyond that: the backbone axis
([[papers/tong-2022-videomae|VideoMAE]]-style domain-matched SSVP or the
[[papers/maes-2026-lewm|LeWM]]/[[papers/bardes-2023-jepa|V-JEPA]] latent-
prediction path).

## Related

- [[directions/vlm-reasoning-layer|Approach 8]] — status ledger updated with this grid
- [[findings/exp8-joint-lora-fusion|Exp8 Joint-LoRA Fusion]] — the inference-time negative this extends
- [[findings/road-waymo-childs-mis-indexed|ROAD-Waymo Childs Bug]] — why these are the first valid violation numbers
- [[findings/exp2f-flat-head|Exp2f Flat Head]] — the focal-on-all recipe the heads reuse
- [[methods/3d-retinanet|3D-RetinaNet]] — the control that still wins
