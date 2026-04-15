---
type: method
title: "Multimodal Causal Driving Model — Architecture Spec"
aliases: ["MCDM", "causal driving model", "Approach 4 architecture"]
created: 2026-04-07
updated: 2026-04-07
sources: ["ROAD_Reason/docs/APPROACHES.md", "ROAD_Reason/docs/approach3_architecture.md"]
tags: [method, road-plusplus, causal, vlm, openmixer, dsdag, t-norm, primary-contribution]
status: complete
datasets_evaluated: [road-plusplus]
---

# Multimodal Causal Driving Model — Architecture Spec

Detailed architecture for [[directions/constrained-vlm-reasoning|ROAD_Reason Approach 4]] — the primary thesis contribution. See [[projects/road-reason|ROAD_Reason]] for research context.

**Core flow:** Perception → Structure → Reasoning → Correction → Constraints

## Objective

A model that simultaneously:
1. Performs structured event prediction (agent, action, location triplets)
2. Learns **causal reasoning** for *why* events occur
3. Uses reasoning to **improve** structured predictions (not a separate explanation module)
4. Enforces logical constraints on outputs

## Pipeline

```
Video Clip  X ∈ ℝ^{T × H × W × 3}
   │
   ├──→ CLIP-ViP Visual Encoder Ψ_VE ──→ V ∈ ℝ^{T×hw×D},  f_v ∈ ℝ^D
   │                                      S = V ⊗ f_t  (patch-text correlation)
   │
Action class prompts (GPT-4, precomputed once)
   └──→ CLIP-ViP Text Encoder Ψ_TE ───→ F_t ∈ ℝ^{C×D}  (cached, always available)
                                          f_t = arg max_{y} f_v^T ⊗ f_t^(y)

OpenMixer (S-OMB uses S for location priors; T-OMB uses f_v as semantic condition; DFA uses F_t)
   ↓  Q_t ∈ ℝ^{K × D},  B ∈ ℝ^{K × 4}
 ┌──────────────────────┬────────────────────────────────┐
 │   Structured Head    │      Causal Reasoning Path     │
 │ 5× sigmoid heads     │                                │
 │ L_raw ∈ ℝ^{K×86}    │  Attention Pool (Q_t → s)      │
 │                      │  Projection Layer (s → s')     │
 │                      │  DSDAG (s', Z=f_v → c ∈ ℝ^D)  │
 │                      │  VLT (c → r ∈ ℝ^D)            │
 └──────────┬───────────┴──────────────┬─────────────────┘
            ↓                          ↓
            └──── L_final = L_raw ⊙ f(r) ────┘
                         ↓
              t-norm Constraint Loss
              L_total = L_det + λ₁·L_tnorm + λ₂·L_language + λ₃·L_sparse
                         ↓
         Entropy-Gated Fusion (inference only)
                         ↓
        Final Predictions + Reasoning
```

---

## Module 1: CLIP-ViP (Frozen Encoder)

CLIP-ViP has **two separate encoders** that run independently (Bao et al., Section 3):

### 1A. Visual Encoder Ψ_VE

| | |
|-|-|
| **Input** | Video clip X ∈ ℝ^{T × H × W × 3} |
| **Output** | V ∈ ℝ^{T×hw×D} (patch-level spatiotemporal features); f_v ∈ ℝ^D (global video-level embedding); S ∈ ℝ^{T×hw} (patch-text correlation map) |
| **Purpose** | Extract semantic, language-aligned visual features |

S is computed as `S = V ⊗ f_t` — the correlation between visual patches and the matched text feature. S is used by S-OMB as the **location prior** for actor sampling. It requires f_t (see below).

### 1B. Text Encoder Ψ_TE

| | |
|-|-|
| **Input** | Action class prompts — GPT-4-generated visually descriptive descriptions per class (precomputed once offline) |
| **Output** | F_t ∈ ℝ^{C×D} — one aggregated text feature vector per action class; cached as a fixed lookup table |
| **Purpose** | Encode action semantics into the shared CLIP embedding space for DFA alignment |

**How f_t is selected at runtime:** `f_t = arg max_{y∈C} f_v^T ⊗ f_t^(y)` — the text feature most similar to the global video embedding f_v. This does not require knowing which action is present — it finds the best-matching class description from the precomputed F_t table.

**At inference:** F_t is a fixed constant (precomputed offline). No runtime text input required. Only video is needed. The text encoder does not run at inference — its output was cached when the vocabulary was defined.

**Prompts are richer than bare class names.** The paper uses GPT-4 to generate multiple visually descriptive prompts per class (e.g., for `Wait2X`: *"A person standing at a junction, looking both ways, waiting for a gap in traffic"*) and aggregates their encodings. For ROAD-Waymo, this prompt generation step is done once for the 19 agent behavior action classes.

---

## Module 2: OpenMixer (Perception Backbone)

Converts video into actor-centric spatiotemporal representations with localization and semantics.

### 2A. S-OMB (Spatial)
| | |
|-|-|
| **Input** | V; Q_s (its own spatial queries); b_{m-1} (boxes from previous stage; b_0 from location sampling via CLIP-ViP attention map S) |
| **Output** | Q_s (updated); person scores ô_m; box offsets Δb_m → b_m = b_{m-1} + Δb_m |
| **Purpose** | Localize actors — *who* and *where*; iteratively refine boxes across M stages |

### 2B. T-OMB (Temporal)
| | |
|-|-|
| **Input** | V; Q_t (its own temporal queries); b_m (boxes from S-OMB); f_t (best-matching text feature) |
| **Output** | Q_t ∈ ℝ^{K × D} — same variable, updated in-place |
| **Purpose** | Capture motion + temporal dynamics — *what is happening* |

**Q_s does NOT feed into T-OMB directly.** Only boxes b_m (derived from S-OMB's Q_s processing) cross over. Q_t and Q_s are separate streams that interact only through b_m.

**Update formula (Eq. 1, Bao et al.):**
```
Q_t ← Ψ_qv(Ψ_qq(Q_t, b_m) ⊕ f_t, V, b_m)
```
1. Q-Q mixing: queries attend to each other, conditioned on boxes b_m
2. ⊕ f_t: best-matching text feature added to inject semantic condition
3. Q-V mixing: queries attend to all spatiotemporal patch features V, conditioned on b_m

**Semantic condition is f_t, not f_v.** f_t is the single text feature most similar to f_v: `f_t = arg max_{y∈F_t} f_v^T ⊙ f_t`. f_v is used to *select* f_t, but f_t is what enters the query stream.

### 2C. DFA (Dynamically Fused Alignment)

| | |
|-|-|
| **Input** | Q_t (post-T-OMB temporal queries); f_v (global video feature); F_t ∈ ℝ^{C×D} (precomputed text features) |
| **Output** | F̄_v ∈ ℝ^{K×D} (fused per-actor features); action scores ŷ_m per query |
| **Purpose** | Open-vocabulary action recognition — align visual query features with text class embeddings |

DFA has two sub-steps (Section 3.4):

**Step 1 — Dynamic Feature Fusion:** fuses the global video feature f_v into each temporal query Q_t dynamically:
```
F̄_v = λ ⊙ f_v + (1 − λ) ⊙ Q_t      where λ ∈ ℝ^{N×1} is learned per-query
```
This allows different queries (different actors) to draw different amounts of global context. F̄_v is the output used both for action scoring and (in Approach 3) as input to the causal branch attention pool.

**Step 2 — Query-Text Alignment:** scores each query against all action class text features:
```
P(ŷ | Q_t) = softmax(F̄_v ⊗ F_t^T / τ)      at training
ŷ = arg max_{y∈C}(F̄_v ⊗ F_t^T)             at inference
```

**F_t is precomputed from GPT-4 prompts, not bare class names.** The paper generates multiple visually descriptive prompts per class via GPT-4 and aggregates their CLIP text encodings. F_t is cached as a fixed lookup table — no text encoder runs at inference.

**Text flows through the whole pipeline, not just DFA:**
- S-OMB uses S = V ⊗ f_t for location priors
- T-OMB uses f_t (best-matching text feature, selected via f_v) as semantic condition added to queries
- DFA uses f_v to build F̄_v, then aligns to F_t for classification

**Where ŷ_m goes — OpenMixer's action classification loss:** `ŷ_m` feeds `L_cls`, OpenMixer's own action classification training objective (L = L_det + L_cls in the original paper). In Approach 3, this is absorbed into the `L_det` term of the joint loss. This is distinct from the Structured Head's action logits: DFA produces **single-label softmax** scores (open-vocabulary recognition task), while the Structured Head's action branch produces **multi-label sigmoid** logits for ROAD-Waymo's 19 agent behavior classes — both operating on the same `Q_t` for different purposes. Since OpenMixer is actively fine-tuned in Stages 2 & 3, `L_cls` is live during training and keeps DFA's open-vocabulary alignment calibrated throughout.

**Source:** Bao et al., WACV 2025

---

## Module 3: Structured Head (Triplet Prediction)

| | |
|-|-|
| **Input** | Q_t ∈ ℝ^{K × D} — per-object feature vectors from OpenMixer |
| **Output** | Post-sigmoid probabilities per label type; L_raw ∈ ℝ^{K × 86} (triplet head is primary) |
| **Purpose** | Benchmark-facing predictions (ROAD-style triplets); evaluated by triplet mAP |

Five independent sigmoid classification heads, applied per object query:

```
object_embedding ∈ ℝ^D
        │
   Linear(D → 10)  + sigmoid  →  agent logits      (10 agent classes)
   Linear(D → 19)  + sigmoid  →  action logits     (19 agent behavior classes — excludes Red/Amber/Green)
   Linear(D → 10)  + sigmoid  →  location logits   (10 location classes)
   Linear(D → 49)  + sigmoid  →  duplex logits     (49 valid duplexes)
   Linear(D → 86)  + sigmoid  →  triplet probs     (86 valid triplets — L_raw, "raw" = pre-reweighting)
```

**Why sigmoid, not softmax:** This is a **multi-label** problem — a single tube can simultaneously carry multiple true labels (e.g., `Ped`, `Wait2X`, and `RhtPav` are all true at once). Sigmoid treats each class independently in [0,1]. Softmax would force a single winner across all classes.

**Why sigmoid is required for t-norm:** The Łukasiewicz formula `violation = max(0, p(A) + p(B) - 1)` requires p ∈ [0,1]. Sigmoid probabilities satisfy this; softmax would break the constraint formulation.

**19 vs 22 action classes:** The dataset has 22 `action_labels` total, but `Red`, `Amber`, `Green` are traffic light signal states specific to the `TL` agent class — not agent behavior actions. The action head uses the remaining 19.

**Head design is identical to Approaches 1 & 2 (3D-RetinaNet baseline).** Any improvement in triplet mAP is attributable to the OpenMixer backbone and causal head, not head redesign — keeping comparisons clean. Backbone is different so weights are not transferable; trained from scratch in Stage 2.

### Why Q_t alone — not Q_s, not f_v explicitly

This is a principled design choice from the OpenMixer paper (Bao et al., Section 3.4 and Table 4 ablation):

> *"The additional condition on spatial queries (SQ) hurts the performance on both the base and novel classes, because this essentially makes the recognition and localization entangled in training."*
> *"We do not include the spatial queries Q_s as the input of our DFA module. This makes the T-OMB and the S-OMB to be decoupled in training such that the person localization is class-agnostic."*

**Q_t is not just "what is happening."** By the time it exits T-OMB it already contains:
- **Spatial context** — T-OMB is conditioned on boxes `b` from S-OMB: `Q̂_t = Ψ_qv(Ψ_qq(Q_t, b) ⊕ f_v, V, b)`. Where the actor is has been baked in via the box predictions, not the Q_s vector itself.
- **Global semantic context** — `f_v` is broadcast-added (⊕) into Q_t before Q-V mixing. Scene-level semantics are already inside Q_t.
- **Temporal motion** — Q_t attends to all spatiotemporal patch features V globally, not a local RoI crop.

Adding Q_s or f_v again would be redundant and harmful. The paper's ablation (Table 4) confirms: TQ alone outperforms TQ+SQ on both base and novel classes.

**Why location labels work from Q_t alone:** ROAD-Waymo's location labels (`RhtPav`, `Junction`, `OutgoLane`) are not pixel-level position — they are semantic scene roles. Q_t, semantically conditioned by f_v (which encodes road geometry, intersection type, etc.), can classify these. It's not spatial in the image-coordinate sense; it's spatial in the scene-understanding sense — exactly what CLIP-aligned features handle well.

**Comparison to 3D-RetinaNet's head input:**

| | 3D-RetinaNet | OpenMixer Q_t |
|---|---|---|
| Spatial context | RoI-cropped 3D patch around anchor | Baked into Q_t via box conditioning in T-OMB |
| Scene/semantic context | None — purely local conv features | f_v fused into Q_t via T-OMB |
| Temporal context | 3D conv over local tube | Global attention across all T frames |
| Vocabulary | Fixed closed-set MLP | Open — Q_t aligned to F_t via DFA |

RetinaNet's head sees a spatially richer local patch (FPN multi-scale, 3D motion conv) but has no VLM semantic grounding. OpenMixer's Q_t is semantically richer but spatially coarser — the right trade-off for the classification task. For fine-grained localization of small distant actors, RetinaNet's FPN still has an edge (see [[comparisons/openmixer-vs-retinanet|backbone gap analysis]]).

---

## Module 4: Environment Z (from CLIP-ViP Global Embedding)

| | |
|-|-|
| **Input** | `f_v ∈ ℝ^D` — the global embedding already produced by CLIP-ViP |
| **Output** | Z = f_v (no additional module needed) |
| **Purpose** | Capture global environmental context for the DSDAG Z node |

CLIP-ViP produces two outputs: per-token spatiotemporal features `V` (consumed by OpenMixer for actor-level queries) and a **global clip-level embedding `f_v`**. OpenMixer's Q_t is actor-centric — it does not capture scene context. `f_v` is already semantically grounded in language and encodes exactly what Z requires: road type, weather, time of day, intersection geometry, traffic signal presence.

**Z = f_v. No separate scene embedding module is needed.**

```
CLIP-ViP → V ∈ ℝ^{T×N×D}  →  OpenMixer  →  Q_t (actor-level)
         → f_v ∈ ℝ^D       →  DSDAG Z      (scene-level, free)
```

> **Defense answer — "Where does Z come from?"** Z is the global clip embedding `f_v` from CLIP-ViP, produced for free alongside the spatiotemporal features. OpenMixer consumes `V` to produce actor-centric queries — `f_v` is the complementary scene-level signal. It captures road geometry, weather, intersection type, and traffic signal state — all the environmental factors that mediate *why* an agent behaves as it does.

---

## Module 5: Causal Branch

**Data flow:**
```
V[0]    → mean pool → W_Xs ───────────────────┐
V[T-1]  → mean pool → W_Xe ───────────────────┤→ DSDAG(Xs, Xe, Y, Z) → c ∈ ℝ^D
F̄_v    → attn pool → W_Y  ───────────────────┤
Z = f_v ──────────────────────────────────────┘
```

MCAM's MFE (VidSwin + 3DResNet) is replaced entirely by CLIP-ViP. No separate feature extractor is needed — V already provides per-frame patch tokens. Three learned projection layers (W_Xs, W_Xe, W_Y) map into DSDAG's input space. All three are trained from scratch in Stage 1; MCAM pretrained weights are **not** transferred (calibrated for VidSwin + 3DResNet features, incompatible with CLIP-ViP).

### 5A. Start State (Xs)

| | |
|-|-|
| **Input** | V[0] ∈ ℝ^{hw×D} — first-frame patch tokens from CLIP-ViP |
| **Output** | Xs ∈ ℝ^D |
| **How** | Mean-pool V[0] over spatial positions → linear projection W_Xs |
| **MCAM equivalent** | `W_Xs(F_init_global + F_init_local)` (Eq. 20) |

### 5B. End State (Xe)

| | |
|-|-|
| **Input** | V[T-1] ∈ ℝ^{hw×D} — last-frame patch tokens from CLIP-ViP |
| **Output** | Xe ∈ ℝ^D |
| **How** | Mean-pool V[T-1] over spatial positions → linear projection W_Xe |
| **MCAM equivalent** | `W_Xe(F_end_global + F_end_local)` (Eq. 20) |

### 5C. Action Node (Y)

| | |
|-|-|
| **Input** | F̄_v ∈ ℝ^{K×D} — per-actor DFA-fused features (whole-video, action-aware) |
| **Output** | Y ∈ ℝ^D |
| **How** | Learned attention pool over F̄_v (weights focus on causally relevant actors) → linear projection W_Y |
| **MCAM equivalent** | `W_Y(F_whole_global, F_whole_local)` (Eq. 20) |

**Old Y (MCAM) vs New Y (Approach 4):**

| | MCAM | Approach 4 |
|---|---|---|
| Source | VidSwin (global) + 3DResNet (local) on whole video | Attention-pooled F̄_v from DFA |
| Vocabulary | Closed — Kinetics-pretrained | Open — CLIP-aligned to text action descriptions |
| Language alignment | None | Yes — F̄_v is in CLIP embedding space, directly comparable to text |
| Action awareness | Implicit — emergent from video features | Explicit — DFA has already aligned queries to action class text embeddings |
| Actor focus | Whole-frame global features | Learned attention pool weights causally relevant actors over background |
| Motion sensitivity | Strong (3DResNet 3D conv) | Moderate (T-OMB attention-based) |

The new Y is better suited for conditioning DSDAG in Approach 4: it enters the DAG already knowing what action is occurring in language terms, reducing the burden on DSDAG to infer action identity and focusing its capacity on reasoning about *why*. The trade-off is reduced motion sensitivity — if subtle motion distinctions (braking vs. slowing vs. stopping) prove limiting, the dual-stream extension (optical flow → 3D conv fused into Y) addresses this.

### 5D. DSDAG

**Structure:**
```
Start State (Xs) → Action (Y) → End State (Xe)
                       ↑
                  Environment (Z = f_v)
                       ↑
                  Reason Mode (W)
```

| | |
|-|-|
| **Input** | Xs, Xe, Y (from 5A–5C); Z = f_v |
| **Output** | Causal embedding c ∈ ℝ^D |
| **Purpose** | Model state transitions, environment influence, reason mode W |

- **W (Reason Mode)**: latent variable encoding the underlying reason behind an observed action. No direct supervision — W is implicitly learned. Called "hidden danger state" in MCAM (`F_W(U_s, Z_ξ)`); renamed here because W is not always danger-related — it encodes the full space of causal origins: genuine hazard, false positive, distraction, rule compliance, social context, etc.

**How the loss shapes W:**

W has no label. It is forced to learn by the training objective. Gradients flow: `L_det / L_tnorm → L_final → f(r) → VLT → c → DSDAG → W`. For W to be useful, it must encode information that:

1. **L_det** — helps predict correct triplet labels. Two clips sharing the same `(Xs, Y, Z, Xe)` but with different correct triplet outcomes force W to encode the distinguishing factor — the only thing that differs is the underlying reason.
2. **L_language** (Stage 1, BDD-X + CoVLA) — aligns W with human language. Two clips with identical surface actions but different captions (e.g. *"slows for pedestrian"* vs. *"slows for fallen object"*) force W to distinguish them at the representation level before any label is applied.
3. **L_sparse** — forces VLT attention onto a small number of visual regions. W must be grounded in specific visual evidence, not diffuse across the whole scene.
4. **L_tnorm** — constraint compliance; indirectly pushes W toward representations that produce logically valid predictions.

Concretely: two ROAD-Waymo clips both labeled `LarVeh-Stop-VehLane`. Xs, Xe, Z are similar. Y (stopping action) is identical. The only signal that can differentiate them is W — learned through the language captions in Stage 1 and the prediction loss in Stage 2/3. W is not designed; it emerges from the pressure to explain variance the label vocabulary cannot.

**Key interpretation:** DSDAG encodes *why* a behavior occurs, not just *what* happens. A pedestrian labeled `Wait2X` near a red traffic light (Z = "Red") has a different reason mode than one labeled `Wait2X` in a gap between vehicles (Z = "moving traffic"). W captures this distinction where the label vocabulary cannot.

> **Defense answer — "Where do Xs, Xe, and Y come from?"** Xs and Xe come from the first and last frames of the clip (MCAM Eq. 19-20), extracted from CLIP-ViP's V and projected by W_Xs / W_Xe. Y comes from attention-pooled F̄_v (DFA-fused whole-video features), projected by W_Y — already action-aware and language-aligned, an improvement over MCAM's Kinetics-pretrained MFE features. Z = f_v, produced for free by CLIP-ViP.

> **Defense answer — "How does W learn without labels?"** W is shaped by the gradient path from L_det and L_language back through the full causal branch. In Stage 1, BDD-X/CoVLA captions provide language supervision that forces W to distinguish clips with identical surface actions but different human-described reasons. In Stage 2/3, L_det forces W to encode whatever residual information is needed to predict correct triplet labels given identical (Xs, Y, Z, Xe). W emerges from that pressure — it is not designed, it is discovered.

**Source:** Cheng et al. (MCAM, ICCV 2025)

---

## Module 6: Vision-Language Transformer (VLT)

| | |
|-|-|
| **Input** | Causal embedding c ∈ ℝ^D |
| **Output** | Reasoning embedding r ∈ ℝ^D; optional natural language explanation |
| **Purpose** | Align causal reasoning with language supervision; sparsity loss prevents hallucination |

The VLT grounds the causal embedding in language — enabling both the reasoning-to-weight mapping and (optionally) human-readable explanations like: *"Pedestrian is waiting at a junction — likely to cross once the light changes."*

### VLT Language Supervision — Three-Stage Solution

ROAD-Waymo has no explanation captions. This is handled with a three-stage training strategy:

**Stage 1 — Pre-training on captioned driving data (BDD-X + CoVLA)**
- BDD-X: dashcam clips + human explanations ("Slow down because pedestrian is crossing")
- CoVLA (arXiv:2408.10845): 10,000 real-world Tokyo driving clips, 6M frames, 83.3h — frame-level **behavior captions** (what the ego vehicle is doing) and **reasoning captions** (why the driver should be careful). The two caption types map directly onto DSDAG nodes: behavior → Y (action), reasoning → W (reason mode). Rule-based grounding suppresses hallucinations; VideoLLaMA2-7B adds richness. See [[papers/covla-2025|CoVLA]].
- Goal: ground the VLT in language semantics and driving vocabulary
- Output: a VLT that can reason linguistically about driving scenarios

**Stage 2 — ROAD-Waymo fine-tuning via structured label reconstruction (no captions needed)**
- Proxy task: predict the correct triplet labels from the reasoning embedding r
- VLT learns to produce r embeddings that are causally informative — without any caption supervision
- Language output at inference is *emergent* from Stage 1 grounding, not a ROAD-Waymo training objective
- Sparsity loss on attention weights discourages hallucination

**Stage 3 — Optional pseudo-caption augmentation (corner cases)**
- For rare or ambiguous scenarios: generate pseudo-captions via Claude (Opus 4.6 for corner cases, Sonnet 4.6 for bulk) using Anthropic Message Batches API
- Pseudo-captions are conditioned on the GT triplet label + video frame description
- Not required for the core architecture; adds supervision signal at the margins

```
Stage 1:  BDD-X + CoVLA  →  VLT learns language grounding
Stage 2:  ROAD-Waymo GT triplets  →  VLT fine-tunes via label reconstruction (no captions)
Stage 3:  Claude pseudo-captions  →  optional augmentation for corner cases
```

> **Defense answer — "Where does the VLT get language supervision on ROAD-Waymo?"**
> It doesn't need captions from ROAD-Waymo. The VLT is pre-trained on BDD-X and CoVLA (captioned driving data) for language grounding, then fine-tuned on ROAD-Waymo using structured label reconstruction as a proxy supervision signal. The reasoning embedding r is trained to be informative enough to predict correct triplet labels — a causally meaningful objective that requires no captions. Natural language output at inference is a zero-shot emergent capability from Stage 1, not a ROAD-Waymo training target.

---

## Module 7: Reasoning-to-Weight Mapping

**Function:** `w = f(r)` where f is a learned MLP or cross-attention

| | |
|-|-|
| **Input** | Reasoning embedding r ∈ ℝ^D |
| **Output** | Weight vector w ∈ [0,1]^{C_triplet} |
| **Purpose** | Map reasoning → importance over triplets; reasoning becomes a gating signal |

**w must be in [0,1].** L_raw is already post-sigmoid probabilities in [0,1]. For `L_final = L_raw ⊙ w` to stay in [0,1] — which t-norm requires — `f(r)` must use a **sigmoid output activation**, not a bare linear layer. An unconstrained MLP output would push L_final outside [0,1] and break the Łukasiewicz formulation.

---

## Module 8: Reweighting

**Equation:** `L_final = L_raw ⊙ w`

| | |
|-|-|
| **Input** | L_raw ∈ [0,1]^{K×86} (post-sigmoid probabilities from structured head); w ∈ [0,1]^{86} (from Module 7) |
| **Output** | L_final ∈ [0,1]^{K×86} — causally-modulated probabilities |
| **Purpose** | Inject causal reasoning into structured prediction |

**Reweighting happens before any loss is applied.** Both `L_det` (classification component) and `L_tnorm` operate on L_final, not L_raw. This is what makes the causal branch end-to-end differentiable — gradients from both loss terms flow back through the reweighting into the reasoning path. If the loss operated on L_raw directly, the causal branch would be a dead-end during training.

---

## Module 9: T-norm Constraint Loss

| | |
|-|-|
| **Input** | L_final |
| **Output** | L_constraint |
| **Purpose** | Enforce logical consistency; penalize invalid triplet combinations |

**Note:** Constraints ensure *validity*, not *causality*. They prevent predictions like `LarVeh-Xing` (invalid duplex) but cannot distinguish `LarVeh-Stop` caused by a hazard vs. a parked car. That distinction is handled by the causal path.

See [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] for t-norm details.

---

## Module 10: Entropy-Gated Fusion (Inference Only)

| | |
|-|-|
| **Input** | L_final |
| **Output** | Final predictions |
| **Purpose** | Trust reasoning more when structured head is uncertain |

High entropy in L_raw → causal head gets higher weight. Low entropy (confident scene) → structured head dominates. Avoids causal head overriding confident structured predictions.

**Source:** Chlon et al. 2025; Hendrycks & Gimpel 2017; Arevalo et al. 2017

---

## Joint Training Loss

```
L_total = L_det + λ₁·L_tnorm + λ₂·L_language + λ₃·L_sparse
```

| Term | Source | Purpose |
|------|--------|---------|
| `L_det` | Structured head vs. GT triplets | Detection accuracy |
| `λ₁·L_tnorm` | T-norm on reweighted logits | Constraint compliance |
| `λ₂·L_language` | VLT vs. language supervision | Reasoning grounding |
| `λ₃·L_sparse` | VLT attention sparsity | Anti-hallucination |

**Key:** T-norm operates on causally-modulated logits (post-reweighting), not raw logits. In Approach 2 the constraint sees raw logits; in Approach 4 it sees causally-informed logits — constraint and causal reasoning work in the same direction.

---

## Training Stage Rationale

The three-stage order is **load-bearing** — not arbitrary:

| Stage | Frozen | Active | Why |
|-------|--------|--------|-----|
| 1 — Causal head pre-training (BDD-X + CoVLA) | OpenMixer, Structured Head | DSDAG, VLT | Causal head cannot learn to reweight the structured head if structured head produces random logits. Language grounding must come first. |
| 2 — Structured head warm-up (ROAD-Waymo) | Causal Head | Structured Head, OpenMixer | Causal head needs meaningful logits to modulate. |
| 3 — Joint fine-tuning (ROAD-Waymo + pseudo-captions) | CLIP-ViP only | Everything else | Couples both heads — meaningful logits to reweight, meaningful weights to modulate with. |

**On Claude pseudo-labels:** ROAD-Waymo has no language annotations. Using Claude with **both the frame image and the GT triplet** breaks circularity. The triplet label alone (`LarVeh-Stop-VehLane`) gives no signal about whether the stop is genuine or a false positive — the image does. This is why pseudo-captions are vision-grounded, not just text-conditioned.

---

## Dual-Stream Extension (Optional — Motion Gap Mitigation)

OpenMixer's T-OMB uses attention across frames, relying on position embeddings for temporal order. For subtle motion distinctions (braking vs. slowing vs. stopping vs. reversing) that DSDAG state transitions depend on, 3D conv may be more sensitive.

If motion-level distinctions prove a limiting factor empirically, a lightweight dual-stream extension can be added to the **causal branch only** — the structured head remains unchanged:

```
CLIP-ViP → OpenMixer → attention pool
                              │
                              ├── semantic stream (OpenMixer features)
                              │
optical flow → 3D conv ───────┤── motion stream
                              │
                        fuse (concat or cross-attention)
                              │
                           DSDAG
```

MCAM's own MFE (Multi-scale Feature Extractor) validates this pattern: VidSwin (transformer, semantic) + 3DResNet (conv, motion-sensitive), fused before DSDAG. Whether this extension is needed depends on empirical results after Stage 1.

---

## Three Representation Spaces

| Space | Contains | Used For |
|-------|----------|---------|
| **Perception** | Boxes B, tracks, queries Q_t | Localization, actor-centric features |
| **Structured Event** | Agent, action, location, triplets L_raw | Benchmark evaluation (triplet mAP) |
| **Causal Reasoning** | Intent, reason mode W, explanation | Logit correction, natural language output |

---

## Key Weak Points (Defense Preparation)

### ❗ "Where is environment Z?"
Z = f_v, the global clip-level embedding from CLIP-ViP — produced for free alongside V. Captures road geometry, weather, intersection type, traffic signal state. No separate module needed.

### ❗ "Where do Xs and Xe come from?"
Temporal slicing: early frames → Xs, late frames → Xe. No explicit Xs/Xe annotations required — derived from clip structure.

### ❗ "Why not just use constraints?"
Constraints enforce validity but cannot distinguish causal origins of identical triplets:
- `LarVeh-Stop` caused by plastic bag in road ≠ `LarVeh-Stop` caused by genuine hazard
- Same triplet label, different underlying intent
- The causal head makes this distinction at the representation level *before* constraints are applied

### ❗ "What is the novel claim?"
Reasoning-driven logit modulation: `L_final = L_raw ⊙ f(r)`. The causal reasoning embedding actively reshapes structured predictions rather than sitting in a parallel branch. This is end-to-end differentiable.

---

## Final Defense Statement

> The model first extracts structured event predictions from video. In parallel, it builds a causal representation of the scene using a state-transition graph (DSDAG) that models Start State → Action → End State, mediated by environment. This causal representation is mapped into a reasoning embedding, which gates structured predictions before logical constraint enforcement. This enables the model to distinguish scenarios with identical labels but different underlying intent — solving the "right label, wrong reason" problem.

---

## What Makes This Novel

1. **Combines:** Open-vocabulary detection (OpenMixer) + causal reasoning (DSDAG) + neuro-symbolic constraints (t-norm) — first on ROAD-Waymo
2. **Introduces:** Reasoning-driven logit modulation (`L_final = L_raw ⊙ f(r)`)
3. **Solves:** "Right label, wrong reason" problem — identical triplets with different causal intent are distinguished

## Related

- [[directions/constrained-vlm-reasoning|Approach 4 Overview]]
- [[methods/qwen25-vl-multitask|Approach 3 (Qwen2.5-VL Multi-Task)]] — upstream; M_bddx + M_covla weights seed Stage 1 of this model
- [[projects/road-reason|ROAD_Reason]] | [[datasets/road-plusplus|ROAD++]]
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[concepts/compositional-labels|Compositional Labels]]
- [[concepts/vlm-architectures|VLM Architectures]] — contrastive / generative / cross-attention patterns
- [[comparisons/openmixer-vs-retinanet|OpenMixer vs 3D-RetinaNet]] — backbone gap analysis
