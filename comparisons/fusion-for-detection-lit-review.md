---
type: comparison
title: "Literature Review: CNN-VLM Fusion for Detection with Constraint Reasoning"
aliases: ["fusion lit review", "detection fusion literature"]
created: 2026-05-06
updated: 2026-05-06
sources:
  - "ROAD_Reason/papers/vmcnet_2025.pdf"
  - "ROAD_Reason/papers/cbam_eccv2018.pdf"
  - "ROAD_Reason/papers/frozen-detr_neurips2024.pdf"
tags: [lit-review, cnn-vit-fusion, feature-modulation, detection, constraints, road-reason]
status: draft
---

# Literature Review: CNN-VLM Fusion for Detection with Constraint Reasoning

Motivated by Exp2b's localization bottleneck: EfficientNet alone localizes well, but scalar-gate fusion with frozen VLM features destroys spatial precision. The broader question is how to combine trainable CNN detection features with frozen VLM semantics so that (1) localization stays sharp, (2) semantic richness flows through to classification heads, and (3) downstream constraint reasoning (t-norms, PiShield) benefits from more confident predictions.

**Core finding from Exp2b**: when the model is confident, t-norm constraints are notably more effective. Better fusion → more confident detections → constraints have stronger signal → better triplet/duplex scores. The architecture must serve this pipeline.

---

## Papers Reviewed

### 1. VMCNet — ViT-Feature-Modulated Multi-Scale CNN (Gao et al., 2025)

**Paper**: [[papers/gao-2025-vmcnet|Gao 2025 — VMCNet]] | [arXiv:2501.16981](https://arxiv.org/abs/2501.16981)

**Architecture**: Dual-branch — trainable multi-scale CNN + frozen CLIP ViT. The CNN handles localization while the ViT provides open-vocabulary semantics. Fusion is via **modulation**: ViT features produce per-channel gamma and beta that scale and shift CNN features (FiLM-style conditioning).

**Why it matters for ROAD_Reason**: Most directly comparable to Exp2b's setup (trainable EfficientNet + frozen Qwen ViT). The critical difference:

| Aspect | Exp2b (current) | VMCNet |
|--------|----------------|--------|
| Fusion | `fpn + scalar_gate * vlm` | `gamma(vlm) * cnn + beta(vlm)` |
| Gate granularity | 1 scalar per FPN level | 256 independent channel weights |
| Spatial selectivity | None | Via multi-scale CNN features |
| Novel category transfer | Minimal (scalar bottleneck) | Full per-channel semantic flow |

**Key insight**: Per-channel modulation means different channels can encode different semantic concepts (agent type, action type, location type). This is exactly what triplet/duplex classification needs — the scalar gate squashes all 256 channels of VLM knowledge into one number.

**Results**: OV-COCO 44.3 AP50_novel (ViT-B/16), OV-LVIS 27.8 mAP. Outperforms prior SOTA on open-vocabulary benchmarks — demonstrating that the modulation mechanism successfully transfers VLM knowledge to novel categories.

**Implication for triplets/duplexes**: Novel categories in ROAD++ terms are rare triplet combinations. If per-channel modulation preserves VLM knowledge about object-action relationships, the model should better distinguish "car turning at junction" from "car moving on road" even for infrequent combinations. Combined with t-norm constraints enforcing valid label co-occurrences, this could substantially improve duplex/triplet scores.

### 2. CBAM — Convolutional Block Attention Module (Woo et al., 2018)

**Paper**: [[papers/woo-2018-cbam|Woo 2018 — CBAM]] | [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)

**Architecture**: Two sequential sub-modules applied to any feature map:

1. **Channel attention** ("what"): Global avg pool + max pool → shared MLP → sigmoid → per-channel weights `[B, C, 1, 1]`
2. **Spatial attention** ("where"): Channel-wise avg + max pool → concat → 7x7 conv → sigmoid → spatial mask `[B, 1, H, W]`

~10K params per instance. Plug-and-play — insert after any conv block.

**Important caveat**: CBAM is a **single-stream self-attention** module — it refines one feature map using information from that same feature map. It has no mechanism to bring in VLM features. It does not perform cross-modal fusion on its own.

**What CBAM actually contributes**: The design principle, not the module itself. CBAM decomposes attention into channel ("which of my 256 filters need emphasis") and spatial ("where in the image is relevant"). This decomposition is what we borrow for cross-modal gating — using CNN features to generate channel + spatial masks that control how much VLM information enters at each location.

**CBAM-inspired cross-modal gating for Exp2b**: Instead of standard CBAM (feature map gates itself), the CNN features generate gates that control VLM flow:

```
# Channel gate: CNN decides which channels need VLM help
channel_gate = sigmoid(MLP(global_pool(fpn_feat)))     # [B, 256, 1, 1]

# Spatial gate: CNN decides where VLM info is useful
spatial_gate = sigmoid(conv7x7(spatial_pool(fpn_feat))) # [B, 1, H, W]

# Combined: VLM only enters where CNN invites it
fused = fpn_feat + (channel_gate * spatial_gate) * vlm_projected
```

This preserves CNN edge/boundary precision (gate → 0 at sharp edges) while injecting VLM semantics in object interiors (gate → 1 where CNN sees object-like structure). Compare to current: `fused = fpn_feat + 0.3 * vlm_projected` — one scalar vs 256 x H x W independent gate values.

**Key insight from the paper**: The spatial attention component specifically improves detection at **high IoU thresholds** — exactly Exp2b's weakness. Grad-CAM shows CBAM-enhanced networks focus more precisely on object boundaries. Adapting this to cross-modal gating means: suppress VLM noise at boundaries, preserve CNN localization.

**Results (original CBAM)**: +1-2% mAP on COCO when added to Faster R-CNN/ResNet. Modest but targeted — the gains land on spatial precision and high-IoU detection quality.

**Relationship to VMCNet**: Complementary, not competing — they address different axes:

| Mechanism | What it controls | Source of gate |
|-----------|-----------------|---------------|
| VMCNet (FiLM) | Per-channel scale+shift of CNN features | VLM features generate gamma/beta |
| CBAM-inspired gating | Per-channel AND per-pixel flow of VLM into CNN | CNN features generate the mask |

VMCNet: VLM modulates CNN ("here's how to reinterpret your features"). CBAM-inspired: CNN gates VLM ("here's where I need your help"). A full fusion could use both directions.

**Implication for constraints**: Spatially gated fusion means features reaching classification heads are concentrated on the detected object, not averaged over background. Sharper features → more confident class probabilities → t-norm constraints receive cleaner input. This matters most for small agents (pedestrians, traffic lights) where background features dominate under the current scalar gate.

### 3. Frozen-DETR — Frozen Foundation Model as Feature Enhancer (Fu et al., NeurIPS 2024)

**Paper**: [[papers/fu-2024-frozen-detr|Fu 2024 — Frozen-DETR]] | [arXiv:2410.19635](https://arxiv.org/abs/2410.19635)

**Architecture**: Frozen foundation model (DINOv2 or CLIP) runs in parallel with a trainable CNN backbone. Two fusion points — and this is where it gets interesting compared to the gating/modulation approaches above:

1. **Patch tokens as extra FPN level**: VLM patch tokens are reshaped to a 2D feature map and **concatenated with CNN multi-scale features**, then all tokens attend to each other through the DETR encoder's self-attention layers.
2. **CLS token as image query**: The CLS token (scene-level VLM summary) is concatenated with the learnable object queries in the decoder, providing global scene context during query decoding.

**This is a fundamentally different fusion philosophy.** VMCNet and CBAM-inspired gating are about one modality controlling the other — directional flow with explicit gates. Frozen-DETR says: **put both feature sets in the same room and let self-attention figure out the interaction.** CNN tokens and VLM tokens attend to each other through standard transformer self-attention. No hand-designed gating, no explicit "who controls whom." The encoder learns the joint embedding implicitly.

**Why this is the joint embedding connection**: Remember the joint embedding intuition? Frozen-DETR's encoder self-attention is literally building a shared representation. CNN token at position (10, 5) can attend to VLM patch at position (3, 2) and pull semantic context. VLM patch (3, 2) can attend to CNN tokens at (9,5), (10,5), (11,5) and absorb their spatial precision. After 6 encoder layers, the features are no longer purely "CNN features" or "VLM features" — they're fused tokens in a learned shared space.

**What Exp2b actually implemented vs what Frozen-DETR does**:

| Aspect | Exp2b (current) | Frozen-DETR (actual) |
|--------|----------------|---------------------|
| Patch token fusion | `fpn + scalar * vlm_proj` (additive, gated by 1 scalar) | Concatenated as extra FPN level → encoder self-attention |
| CLS token | Not used | Additional decoder query providing scene context |
| Interaction depth | Single addition | 6 encoder layers of bidirectional attention |
| Alignment learned by | One scalar gate param | Full transformer self-attention weights |

Exp2b took the paper's *concept* (frozen VLM alongside trainable CNN) but used the simplest possible fusion. The paper's actual mechanism is far richer.

**Results**: DINO (R50) 49.0 → 51.9 AP (+2.9) with one foundation model, 53.8 AP (+4.8) with two. +6.6% on LVIS long-tail categories, +8.8% novel AP on open-vocabulary COCO. The long-tail and novel-category gains are particularly relevant — ROAD++ triplets are inherently long-tail.

**Key insight**: The encoder self-attention is doing what our FiLM + gating pipeline tries to do manually, but in a **learned, bidirectional, multi-layer** way. Each encoder layer refines the CNN-VLM interaction. By layer 6, the model has had six rounds of "CNN asks VLM what's here" and "VLM asks CNN where exactly."

**Trade-off vs gating approaches**:

| | Gating (VMCNet + CBAM) | Encoder fusion (Frozen-DETR) |
|--|----------------------|---------------------------|
| Interpretability | High — you can visualize gate masks | Low — attention is distributed across 6 layers |
| Compute cost | Minimal (~30K params) | Higher — VLM patches go through encoder self-attention |
| Localization control | Explicit — CNN controls where VLM enters | Implicit — attention learns it, no guarantee |
| Semantic depth | Shallow — one-shot modulation | Deep — 6 layers of iterative refinement |
| Implementation for Exp2b | Replace scalar gate in `vlm_fusion.py` | Feed VLM patches into Deformable DETR encoder (encoder already exists) |

**Implication for constraints**: The deeper fusion means more semantically coherent features reaching the classification heads. Instead of "CNN features with some VLM modulation," you get features that have been jointly refined through six rounds of cross-modal attention. The resulting class probabilities should be more calibrated — and calibrated probabilities are exactly what t-norm constraints need to distinguish "confidently wrong" from "uncertain but correctable."

**The CLS-as-query trick**: This is a free win that Exp2b doesn't use. The VLM's CLS token already encodes a scene-level summary ("intersection with vehicles and pedestrians"). Adding it as a decoder query gives the decoder global context — useful for resolving ambiguous detections. One extra query, zero architectural changes to the decoder.

---

## Summary Table

| Paper | Fusion Type | Granularity | Localization | Semantics | Constraint Fit |
|-------|-----------|-------------|-------------|-----------|---------------|
| [[papers/gao-2025-vmcnet\|VMCNet]] | FiLM modulation | Per-channel | Preserved (CNN branch) | Full VLM transfer | High — confident per-channel features feed t-norms |
| [[papers/woo-2018-cbam\|CBAM]] | Design principle (not direct fusion) | Per-channel + per-pixel | Improved when adapted as cross-modal gate | Requires adaptation — CNN generates mask that gates VLM flow | High — spatially precise gating gives t-norms cleaner signal, especially small agents |
| [[papers/fu-2024-frozen-detr\|Frozen-DETR]] | Encoder self-attention (joint embedding) | Per-token (full attention) | Implicit — learned through 6 encoder layers | Deep — 6 rounds of bidirectional refinement | Highest — calibrated joint features, but less interpretable |

---

## Implications for ROAD_Reason

The overarching pattern: **decouple localization from semantic enrichment** at the architectural level, not just at the loss level. The CNN branch owns spatial precision; the VLM branch owns semantic knowledge; the fusion mechanism controls information flow without destroying either.

This directly validates Exp2b's dual-backbone design — the architecture is right, the fusion mechanism is the bottleneck.

**Two implementation paths are emerging**:

**Path A — Explicit gating (VMCNet + CBAM-inspired):**
1. CBAM-inspired cross-modal gating: CNN generates channel + spatial masks controlling VLM flow (~10K params/level)
2. VMCNet-style FiLM: VLM modulates CNN via per-channel gamma/beta (~512 params/level)
3. Total: ~30K params. Fast to implement, interpretable, predictable.

**Path B — Encoder fusion (Frozen-DETR-style):**
1. Feed VLM patch tokens as extra FPN level into the existing Deformable DETR encoder
2. Add CLS token as additional decoder query (free scene context)
3. Total: zero new modules — the encoder already exists, just needs VLM tokens concatenated in. Deeper interaction, less control.

Path A gives explicit control over the CNN-VLM boundary. Path B lets the model learn the interaction end-to-end. They are not mutually exclusive — Path A could operate at FPN level, with Path B operating at encoder level.

**Update (2026-05-07):** [[findings/exp2c-frozen-detr|Exp2c]] implements Path B faithfully using Frozen-DETR's architecture with CLIP ViT-L/14. Training results pending.

---

## Related

- [[findings/exp2b-deformable-detr|Exp2b Deformable DETR]] — the localization bottleneck motivating this review
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — downstream t-norm reasoning that benefits from better fusion
- [[papers/perez-2018-film|FiLM]] — foundational conditioning mechanism
- [[papers/woo-2018-cbam|CBAM]] — channel + spatial attention (complementary to modulation)
- [[papers/fu-2024-frozen-detr|Frozen-DETR]] — encoder self-attention fusion (joint embedding approach)
- [[papers/stoian-2024-pishield|PiShield]] — hard constraint guarantees downstream of detection
