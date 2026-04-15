---
type: concept
title: "VLM Architectures"
aliases: ["vision-language model", "VLM", "CLIP", "contrastive learning", "DFA", "cross-attention VLM"]
created: 2026-04-07
updated: 2026-04-07
sources: []
tags: [concept, vlm, clip, architecture, road-reason]
status: complete
---

# VLM Architectures

A VLM (Vision-Language Model) connects a **vision encoder** to a **language model** or **language embedding space**. Three major patterns appear across the literature and in [[methods/multimodal-causal-driving|MCDM]].

---

## Pattern 1 — Contrastive (CLIP-style)

Train two encoders (vision + text) to map matched pairs close together in a shared embedding space:

```
Image of cat ──→ [Vision Encoder] ──→ embedding
"a cat" ────────→ [Text Encoder] ───→ embedding
                                       ↑ pull together, push mismatches apart
```

**Output:** A shared embedding space. No language generation — just similarity scoring.

**Key property:** Any image and any text can be compared via dot product or cosine similarity. This enables zero-shot classification: embed class names as text, embed image, find nearest class.

**Examples:** CLIP, CLIP-ViP, SigLIP, ALIGN

**Used in MCDM:** CLIP-ViP (frozen encoder) and DFA (open-vocab action scoring)

---

## Pattern 2 — Generative (LLaVA / InstructBLIP-style)

Connect a visual encoder to an autoregressive LLM:

```
Image ──→ [Vision Encoder] ──→ visual tokens
                                    ↓
                         [Projection / Adapter]
                                    ↓
                    [LLM: "The pedestrian is..."] ──→ text output
```

The projection layer (MLP or cross-attention) maps visual features into the LLM's token embedding space. The LLM then generates text token-by-token, conditioned on both visual tokens and any text prompt.

**Training:** Typically two-stage — freeze vision encoder, train projection; then fine-tune end-to-end with instruction tuning.

**Examples:** LLaVA, InstructBLIP, Flamingo, SmolVLM, Qwen-VL

**Used in MCDM:** SmolVLM is the baseline for ROAD_Reason zero-shot / constraint-prompted / GT-conditioned reasoning (Approach 1 baseline, not Approach 3)

---

## Pattern 3 — Cross-attention Fusion (VLT-style)

Rather than generating free text, cross-attention fuses a visual or causal embedding with language representations to produce a **fixed-size reasoning embedding**:

```
c ∈ ℝ^D  ──→ [Cross-attention over language space] ──→ r ∈ ℝ^D
```

`r` is not text — it's a dense vector carrying language-grounded semantic meaning. An optional decoder head on top of `r` can emit natural language, but the core output is the embedding itself.

**Key advantage:** The reasoning embedding `r` is differentiable and can directly gate downstream predictions (e.g., `L_final = L_raw ⊙ f(r)`). Generative VLMs (Pattern 2) are harder to integrate into end-to-end differentiable pipelines.

**Used in MCDM:** VLT module — maps causal embedding `c` → reasoning embedding `r` for logit reweighting

---

## DFA — Dual-stream Feature Alignment

DFA is a specific application of the contrastive pattern inside OpenMixer (Bao et al., WACV 2025). It enables **open-vocabulary action recognition** without a fixed classifier.

```
E_text:  "Wait2X"    → CLIP text encoder → vector  ┐
         "MovAway"   → CLIP text encoder → vector  ├─ computed once offline, cached
         "XingFmLft" → CLIP text encoder → vector  ┘

At inference:
Q_t (visual actor queries)  ·  E_text^T  →  similarity scores  →  action probabilities
```

Adding a new action class = add one text embedding. No retraining needed. This is the same mechanism used by OWL-ViT, GLIP, and other open-vocabulary detectors.

**Why this works:** CLIP's vision and language encoders share an embedding space. The dot product between a visual query and a text embedding of a class name is a meaningful similarity score — because CLIP was trained to align them.

**At inference:** E_text is a fixed constant (precomputed offline). The model sees only video — no text input required.

---

## How the Three Patterns Connect in MCDM

| Module | VLM Pattern | Role |
|--------|-------------|------|
| CLIP-ViP (frozen) | Contrastive | Extracts semantically grounded spatiotemporal features |
| DFA (inside OpenMixer) | Contrastive similarity | Open-vocab action scores via Q_t · E_text^T |
| VLT | Cross-attention fusion | Maps causal embedding `c` → reasoning embedding `r` for logit gating |
| SmolVLM (baseline only) | Generative | Zero-shot / prompted scene reasoning (Approach 1, not Approach 3) |

**Why CLIP-ViP being frozen is so valuable:** It already "speaks language" — its visual features align with text. This is what makes both DFA (similarity scoring against text class names) and the VLT's language grounding (cross-attention over language representations) work without massive in-domain captioned datasets.

---

## Related

- [[methods/multimodal-causal-driving|MCDM Architecture Spec]] — full pipeline with DFA and VLT in context
- [[methods/smolvlm-road|SmolVLM Baseline]] — generative VLM used in ROAD_Reason baseline
- [[directions/constrained-vlm-reasoning|Approach 3 — Constrained VLM Reasoning]]
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
