---
type: direction
title: "VLM Reasoning Layer over 3D-RetinaNet (Approach 8)"
aliases: ["VLM-as-teacher", "Approach 8", "Moradi 2026-06-02 direction", "VLM reasoning layer"]
created: 2026-06-04
updated: 2026-06-04
sources:
  - "wiki/raw/2026-06-02_moradi_vlm_reasoning_layer.txt"
tags: [direction, road-plusplus, vlm, 3d-retinanet, neuro-symbolic, t-norm, staged-experiment, primary-contribution]
status: complete
novelty: true
feasibility: "workstation"
datasets_required: [road-plusplus, bdd-x, covla]
approach_number: 8
---

# VLM Reasoning Layer over 3D-RetinaNet (Approach 8)

Dr. Moradi's [revised research direction](../raw/2026-06-02_moradi_vlm_reasoning_layer.txt) (2026-06-02). The framing shift:

> "language acts as a structured reasoning space that regularizes the visual representation"

**Augment, don't swap.** Keep the [[methods/3d-retinanet|3D-RetinaNet]] perception trunk (locked at **17.76% agent f-mAP** on [[datasets/road-plusplus|ROAD-Waymo]] val). Bolt a VLM on top as a *reasoning layer* that supplies structured semantic supervision. Long-term, distill that reasoning into the detector so the VLM can be dropped at inference.

This is a deliberately **staged** simplification of [[directions/constrained-vlm-reasoning|Approach 4]]: same end-goal (language-grounded triplet prediction with logic constraints), but starts from a working detector instead of replacing it with OpenMixer + DSDAG + VLT in one step.

## Core Insight — Why This Pivot

The [[findings/exp2-series-narrative|Exp2 series post-mortem]] (six DETR-based experiments, plateau at 5.51% agent f-mAP) revealed the diagnosis Moradi formalizes here:

> "DETR and CLIP are image-level — they throw away time. Video-mAP punishes that, so it makes sense they didn't help."

Therefore the perception backbone must remain spatiotemporal. The 3D-RetinaNet is **kept frozen**, its 17.76% becomes the floor we cannot regress below, and all new capacity goes into the language reasoning layer.

## Architecture — Late Fusion of Language with Detector Logits

```
Video clip [T=8, 600x840]
   │
   ├──► Frozen 3D-RetinaNet ───► tube boxes + 184-dim logits
   │       (model_000025.pth, 17.76% agent f-mAP floor)
   │                                │
   │                                ▼
   └──► VLM (frozen or LoRA) ───► structured JSON per agent
          prompt: frames + boxes + schema       {agent, actions, location,
          + valid-combination constraints        rationale}
          (cached once, not regenerated)         │
                                │                │
                                ▼                ▼
                          language embedding   detector logits
                                  │              │
                                  └──── fuse ────┘
                                          │
                              small trainable head
                                          │
                                  re-predicted triplet
                                          │
                              (optional) T-norm penalty
```

The fusion is **late** (language embedding × detector logit), not early (image-level patches). Per Moradi: "for now start with VLM generated the response... then use language embedding to fused with output of 3d retina net and then re predict with adding few layer to be ttrained."

## The Staged Experimental Ladder

Moradi explicitly asked for incremental experiments — *"lets start one by one."* Each stage answers one question before the next is funded.

### Stage 1 — VLM as offline teacher, no training
- Prompt a VLM with frames + baseline boxes.
- **Force structured JSON** per agent: `{agent, actions, location, rationale}`. No free-form text.
- Constraints (valid agent-action combinations from [[concepts/neuro-symbolic-constraints|ROAD-R rules]]) embedded **in the prompt**.
- Cache the outputs once — regeneration is expensive.
- Fuse cached VLM output with 3D-RetinaNet detections via a fixed rule (vote / rank-fusion).
- **Question answered:** does VLM reasoning, used as-is, lift the baseline above 17.76%?

### Stage 2 — Small trainable fusion head
- Two auxiliary branches, **joint** training:
  - *Video → Explanation* (consumes cached VLM outputs)
  - *Video → Triplet* (consumes 3D-RetinaNet outputs)
- Add a few trainable layers on top of `(language_embedding ⊕ detector_logits)` → re-predict the 184-dim flat target.
- 3D-RetinaNet and VLM both frozen; only the fusion head trains.
- **Question answered:** can a tiny learned fusion beat the Stage 1 fixed rule?

### Stage 3 — VLM fine-tuned on driving data
- LoRA-tune the VLM on [[datasets/bdd-x|BDD-X]] then [[datasets/covla|CoVLA]] for explanation style.
- Plug the driving-tuned VLM back into Stage 2's pipeline.
- **Question answered:** does a driving-aware VLM produce richer cached outputs that further lift triplet f-mAP?

### Stage 4 — Distill reasoning into the detector
- **Contrastive or MSE loss** aligning 3D-RetinaNet tube features ↔ VLM text embeddings.
- The visual network learns to carry the reasoning internally.
- **At inference, drop the VLM.** Zero VLM cost in deployment.
- **Question answered:** can the detector internalize language reasoning?

## Orthogonal Experiments — ROAD-R Constraints (Two Paths, Run Separately)

Moradi explicitly wants these teased apart so the contribution of each is attributable:

| Path | Where rules act | Implementation |
|------|----------------|----------------|
| **Prompt-side** | Inside the VLM prompt | "Negative constraints": "agent X cannot do action Y", listed in the system prompt before the JSON schema |
| **Loss-side** | On the classification heads | [[concepts/neuro-symbolic-constraints\|T-norm penalty]] (Godel form) on the fused head's sigmoid outputs — same recipe as the original ROAD-R paper |

Each runs as its own experiment with the others held fixed.

## The Fusion Design Space

The email enumerates fusion options. We start with **late embedding fusion** (Stage 1–2), but Moradi flagged other valid points to revisit:

| Where to fuse | What's fused | Status |
|---------------|--------------|--------|
| Early | Raw image features ↔ raw VLM features | Mentioned, not for Stage 1 |
| Mid | Mid-encoder features | Future ablation |
| **Late (Stage 1–2)** | **VLM language embeddings ↔ 3D-RetinaNet logits** | **Stage 1–2 start** |
| "Localize then VLM" | Boxes inserted into prompt; VLM reasons from box-aware view | Variant within Stage 1 |

## Training Stability — Aux Loss Balance (Moradi's Caution)

The joint training of `Video → Explanation` + `Video → Triplet` + (Stage 4) contrastive alignment introduces three competing signals. Moradi explicitly flagged:

> "And we need to balance the auxiliary loss or joint training gets unstable."

Strategy will be empirical (sweep weights at Stage 2 entry) — no a-priori value selected yet. The [[findings/exp2f-flat-head|Exp2f focal-on-all]] design (single 184-dim flat target with focal loss) is a candidate base loss because it already handles negative supervision cleanly; aux losses ride on top.

## Comparison to Other Approaches

| | [[directions/qwen-multitask-finetuning\|Approach 3]] | [[directions/constrained-vlm-reasoning\|Approach 4]] | **Approach 8 (this)** |
|---|---|---|---|
| Backbone | Qwen2.5-VL replaces detector | OpenMixer (frozen CLIP-ViP) replaces detector | **3D-RetinaNet preserved** |
| Where language enters | Backbone output heads | DSDAG causal head + VLT | **Cached VLM JSON fused with detector logits** |
| Trainable surface | Full backbone + LoRA + heads | OpenMixer + VLT + DSDAG | **Tiny fusion head only (Stage 1–2)** |
| Training stages | 4 phases | 3 phases | **4 stages, each gated by results** |
| Floor guarantee | None (full retrain) | None (full retrain) | **17.76% (frozen baseline)** |
| Inference cost | Full VLM | Full VLM + OpenMixer | **Stage 4: detector only (VLM dropped)** |

The defining property of Approach 8: **the baseline cannot regress.** Stage 2's worst case is the fusion head learns the identity (passes detector logits through unchanged), yielding ≥ 17.76%.

## Open Questions for the Meeting

Architectural choices Moradi left open — surface these first:

**VLM selection.** Qwen2.5-VL? Kimi-VL? GPT-4o (API cost)? Affects throughput, fine-tuning feasibility, output stability.

**Prompt input.** Frames-only, or frames + rendered box overlays? Single frame or 8-frame clip?

**Per-agent JSON keying.** How is cached VLM output aligned to tubes at training time — by baseline tube ID, by proposal IoU, by VLM-emitted descriptions?

**Stage 1 "no training" fusion rule.** Without a learned head, what combines RetinaNet + VLM into a single triplet prediction? Weighted vote? Rank fusion? Specify before launch.

**Language embedding source.** The VLM's own text tower? A separate sentence encoder (CLIP-text, SBERT)?

**Constraint sourcing.** Read ROAD-R rules from the [original ROAD-R logic file](https://github.com/EGiunchiglia/ROAD-R) or derive from `duplex_childs` / `triplet_childs` in the ROAD-Waymo JSON?

**LoRA timing.** Does VLM stay frozen through Stage 2, or does LoRA kick in there already? Moradi mentioned LoRA "to keep it fast" but didn't specify the stage.

**Aux-loss weighting.** Strategy for balancing `L_explanation`, `L_triplet`, and (Stage 4) `L_contrastive`.

## What This Means for Existing Work

| Artifact | Status |
|----------|--------|
| [[methods/3d-retinanet\|3D-RetinaNet baseline]] (17.76% agent f-mAP, `model_000025.pth`) | **Keystone.** Frozen trunk, never retrained. |
| `exp4_retinamoon` (scaffolded, eval-validated, untrained) | **Likely shelved.** It fuses MoonViT *patches* with RetinaNet; Approach 8 fuses VLM *language* with RetinaNet — different axis. The frozen-RetinaNet wrapper + eval harness are still reusable. |
| ROAD-Waymo eval harness from `exp4_retinamoon/eval.py` | **Reuse.** Reproduces baseline within ~2 pp; same protocol applies to Stage 1–3 outputs. |
| [[concepts/neuro-symbolic-constraints\|T-norm loss]] (`tnorm_loss.py`) | **Reuse** for the loss-side ROAD-R experiment. |
| [[directions/constrained-vlm-reasoning\|Approach 4]] (OpenMixer + DSDAG + VLT) | **Demoted to long-term.** Approach 8 is a strictly simpler precursor that can borrow components (e.g., the language-aligned target embedding from Stage 4 resembles VLT's reasoning embedding `r`). |
| Approved corpus order `ROAD-R → BDD-X → CoVLA → joint` | **Re-validated.** Maps onto Stages 2–3. |

## Verification Gates

| Stage | Pass criterion |
|-------|----------------|
| 1 | Cached-VLM + fixed-fusion eval > 17.76% agent f-mAP on ROAD-Waymo val |
| 2 | Trained fusion head > Stage 1 by ≥ 2 pp, plus ≥ 2 pp on duplex/triplet |
| 3 | Driving-tuned VLM > Stage 2 by ≥ 2 pp on duplex/triplet (where language matters most) |
| 4 | VLM-dropped detector ≥ Stage 3 within 1 pp; demonstrates internalized reasoning |

Per-stage logs and metrics commit to `findings/exp5-vlm-reasoning-stage<N>.md` after each gate clears.

## Related

- [[projects/road-reason|ROAD_Reason]] — overall project
- [[methods/3d-retinanet|3D-RetinaNet]] — preserved perception trunk
- [[findings/exp2-series-narrative|Exp2 Series Narrative]] — the post-mortem Moradi cited as the backbone-swap mistake
- [[concepts/vlm-localization-gap|VLM Localization Gap]] — diagnosis that motivates keeping a real detector
- [[directions/constrained-vlm-reasoning|Approach 4: Constrained VLM Reasoning]] — more ambitious sibling
- [[directions/qwen-multitask-finetuning|Approach 3: Qwen2.5-VL Multi-Task]] — earlier all-in-one VLM design
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — ROAD-R rules, both prompt-side and loss-side
- [[findings/exp2f-flat-head|Exp2f Flat-Head]] — base loss design candidate for Stage 2
- [[datasets/road-plusplus|ROAD++]] | [[datasets/bdd-x|BDD-X]] | [[datasets/covla|CoVLA]]
