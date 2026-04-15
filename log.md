---
type: log
---

# Wiki Operations Log

Append-only record of all wiki operations. Format: `## [YYYY-MM-DD] OPERATION — Description`

---

## [2026-04-14] UPDATE — Repo organization: README landing pages + .gitignore for Intelligent_AutoDrive

- Files created: `ROAD_Reason/README.md` (overwrite — full landing page), `ROAD_Reason/.gitignore`, `AutoDrivePerception2026/README.md`
- `ROAD_Reason/docs/` renamed to `working_docs/` (local only, gitignored); new public `docs/` directory created
- `.gitignore` covers: `papers/`, `working_docs/`, `.claude`, `.codex`, model weights, Python caches
- README reflects full roadmap (Approaches 1–7), active Approach 3 status, constraint background, and repo structure
- AutoDrive repo name will change from AutoDrive_VLM → AutoDrive_Perception (no VLM used in that pipeline)
- Source: Dr. Moradi email to lab members requesting repo landing pages + descriptions

---

## [2026-04-14] INGEST — Weekly update email to Dr. Moradi (2026-04-14)

- Pages updated: 0 (all content already captured in prior wiki entries)
- Email confirmed: BDD-X manual labels vs CoVLA rule-based/VideoLLaMA2-7B labels; CoVLA trajectory via Kalman filter (GPS+IMU, ~3s, 60 points); Qwen2.5-VL-7B selected over Qwen-Omni; Experiment 1 = ROAD-R with ViT encoder, video frames + labels → triplet/duplex + t-norm
- No new facts — email is a status summary of content already in wiki as of 2026-04-13

---

## [2026-04-13] UPDATE — Approach 3 training order confirmed by Dr. Moradi

- Pages updated: 2 (`directions/qwen-multitask-finetuning.md`, `methods/qwen25-vl-multitask.md`)
- Training is sequential: ROAD-R → BDD-X → CoVLA → joint (not parallel as originally planned)
- ROAD-R experiment 1 spec confirmed: ViT encoder, video frames + labels as input, triplet + duplex as output, t-norm as loss
- Source: weekly update email to Dr. Moradi, 2026-04-13

---

## [2026-04-10] INGEST — Approach 3: Qwen2.5-VL Multi-Task Fine-Tuning

- Pages created: 2 (`directions/qwen-multitask-finetuning.md`, `methods/qwen25-vl-multitask.md`)
- Pages updated: 7 (`projects/road-reason.md`, `directions/constrained-vlm-reasoning.md`, `directions/jepa-intent-head.md`, `directions/lewm-scene-prediction.md`, `methods/multimodal-causal-driving.md`, `index.md`, `log.md`)
- Raw source updated: `ROAD_Reason/docs/APPROACHES.md` — inserted Approach 3, renumbered old 3→4, 4→5, 5→6, 6→7
- Backbone selected: Qwen2.5-VL-7B (not Qwen2.5-Omni) — Apache 2.0, native bbox output, 16-20GB VRAM, mature LoRA ecosystem
- Three task modules: T1 (BDD-X: LM head), T2 (ROAD-R: detection + cross-frame tube-linking + 5× sigmoid + t-norm), T3 (CoVLA: LM head + 10×3 trajectory MLP)
- Training: Phase 1 separate LoRA adapters; Phase 2 joint (20/40/40% task sampling)
- All "Approach 3" references updated to "Approach 4" across wiki; new Approach 3 wired in

---

## [2026-04-10] INGEST — BDD-X vs CoVLA comparison page

- Pages created: 1 (`comparisons/bdd-x-vs-covla.md`) | Pages updated: 1 (`index.md`)
- Full side-by-side table: 22 properties, verified field names from actual data
- DSDAG relevance table: Answer.Naction/plain_caption → Y; Answer.Njustification/risk → W
- Key contrasts: scale (6M vs 26K), quality (human vs VLM-auto), granularity (frame vs segment), sensor richness (CoVLA CAN bus + trajectory)
- Stage 1 joint training strategy documented with mini-batch schema

---

## [2026-04-10] INGEST — CoVLA dataset page (verified from CoVLA-Mini)

- Pages created: 1 (`datasets/covla.md`) | Pages updated: 1 (`index.md`)
- Full annotation schema documented from actual JSONL files: captions (plain_caption, rich_caption, risk, weather, road, scene flags), states (ego_state CAN bus + IMU + ECEF, trajectory 60×3, camera matrices), front_car (lead detection), traffic_lights (class + bbox)
- Trajectory: 60×3 float array in vehicle frame, ~3s at 20FPS; first point always [0,0,0]
- Camera intrinsics confirmed: fx=fy=2648, cx=964, cy=604 (1928×1208 sensor)
- Dataset index updated: Datasets count 5→6

---

## [2026-04-10] UPDATE — CoVLA annotation schema verified from CoVLA-Mini

- Pages updated: 1 (`papers/covla-2025.md`)
- Explored CoVLA Mini (50 scenes × 600 frames = 30K frames) by parsing concatenated-JSON JSONL files
- Confirmed actual field names differ from paper terminology: `plain_caption` (not "behavior_caption"), `risk` (not "reasoning_caption"), `rich_caption` (extended, combines all)
- Full `states/` schema verified: ego_state (vEgo, aEgo, steeringAngleDeg, brake, gas, gearShifter, blinkers, IMU, ECEF position), trajectory (60×3 pts), extrinsic/intrinsic matrices, frame_id
- `front_car/` schema: has_lead, lead_prob, lead_x/y, lead_speed_kmh, lead_a (per frame)
- `traffic_lights/` schema: index, class (green/red/amber), bbox [x1,y1,x2,y2] (list per frame)
- Weather distribution (50 scenes): sunny 25, cloudy 21, rainy 4
- Road type distribution: wide road 33, narrow road 17
- Updated `explore_annotations.py`: replaced parquet assumption with correct JSONL concatenated-JSON parser
- Updated `dataset_comparison.csv`: CoVLA columns now use verified field names
- Added Annotation Format table to covla-2025.md with all verified field names and types

---

## [2026-04-10] INGEST — BDD-X + CoVLA datasets (download + exploration)
- Pages created: 2 (`datasets/bdd-x.md`, `papers/covla-2025.md` updated) | Pages updated: 2 (`index.md`, `methods/multimodal-causal-driving.md`)
- BDD-X: cloned repo to `/data/repos/ROAD_Reason/datasets/BDD-X/`, downloaded annotations to `/data/datasets/BDD-X/`. Verified: 7,000 unique videos, 26,538 action segments, 61-col wide CSV, mean justification 7.9 words. Splits: 5,588/698/698.
- CoVLA: README/metadata downloaded. Full data pending HuggingFace authentication (gated dataset). Mini + full downloads queued for after auth.
- Comparison CSV written: `/data/repos/ROAD_Reason/datasets/dataset_comparison.csv` (34 rows, CoVLA columns populated from paper; will be verified from actual data post-auth).
- Exploration script: `/data/repos/ROAD_Reason/datasets/explore_annotations.py` — re-run after CoVLA auth to fill verified CoVLA stats.

---

## [2026-04-10] INGEST — CoVLA (arXiv:2408.10845)
- Pages created: 1 (`papers/covla-2025.md`) | Pages updated: 2 (`methods/multimodal-causal-driving.md`, `index.md`)
- Summary: Turing Inc. VLA dataset, 10K clips / 6M frames / 83.3h of Tokyo driving. Frame-level behavior + reasoning captions (auto-generated via rule-based + VideoLLaMA2-7B). Two caption types map directly onto DSDAG nodes: behavior → Y (action), reasoning → W (reason mode). Stage 1 pre-training source for Approach 3 causal head alongside BDD-X. Architecture spec updated with accurate CoVLA description replacing placeholder.

---

## [2026-04-07] INGEST — Initial Bootstrap (Full Research Portfolio)

**Sources processed:**
- `PedestrianIntent++/MEMORY.md` — project tracking and verified statistics
- `PedestrianIntent++/SYNTHESIS.md` — 7-part master synthesis (75KB+)
- `PedestrianIntent++/JAAD/JAAD_summary.md` (960 lines)
- `PedestrianIntent++/PIE/PIE_Dataset_Summary.md` (978 lines)
- `PedestrianIntent++/ROAD_plusplus/ROAD_plusplus_summary.md` (846 lines)
- `PedestrianIntent++/ROAD_plusplus/APPROACHES.md` — 6-approach PhD roadmap
- `PedestrianIntent++/JAAD_analysis/cross_tabulations.md`
- `PedestrianIntent++/JAAD_analysis/analysis/cross_tab_results.json`
- `PedestrianIntent++/PIE_analysis/cross_tab_results.json`
- `PedestrianIntent++/PIE_analysis/pie_stats.json`
- `ROAD_Reason/docs/CLAUDE.md`
- `ROAD_Reason/docs/APPROACHES.md`
- `EfficientPIE/README.md`
- `EfficientPIE/docs/RESULTS.md`
- All repo READMEs (AutoDrivePerception2026, YOLOPX, YOLO_BDD, bevfusion, mmdetection3d, TwinLiteNet, eff_light_detection, eff_sign_detection, SLURP)

**Pages created: 74**

| Category | Count |
|----------|-------|
| Projects | 12 |
| Datasets | 4 |
| Concepts | 10 |
| Papers | 19 |
| Methods | 8 |
| Directions | 9 |
| Findings | 4 |
| Comparisons | 3 |
| Tools | 4 |
| Infrastructure (CLAUDE.md, index.md, log.md) | 3 |

**Key content encoded:**
- JAAD/PIE/ROAD++ full dataset specifications with verified statistics
- The gaze reversal finding (walking+looking → 95.7% JAAD vs. 56% PIE)
- PIE intention_prob bimodal distribution (mean=0.712, median=0.850)
- ROAD++ paper overcounting (verified: 153K frames, 41,935 tubes, 3.3M boxes)
- 6 ROAD_Reason research approaches (Approaches 3-5 are novel thesis contributions)
- SparseTemporalPIE v3 results (0.926 acc / 0.947 AUC on PIE)
- Full detection/segmentation stack (YOLOPX, YOLO_BDD, TwinLiteNet, AutoDrivePerception2026)
- 19 papers from the literature survey

**Raw source integrity:** All existing files in /data/repos/ were read-only. No raw source was modified.

---

## [2026-04-07] INGEST — Multimodal Causal Driving Model Architecture Spec

**Source:** Draft architecture spec (pasted directly into conversation)

**Pages created: 1**
- `methods/multimodal-causal-driving.md` — full module breakdown with math, three representation spaces, defense prep, key weak points

**Pages updated: 2**
- `directions/constrained-vlm-reasoning.md` — added DSDAG structure, L_final equation, "right label, wrong reason" framing, link to arch spec
- `projects/road-reason.md` — updated Approach 3 components with Xs/Xe/Z detail and logit reweighting equation

**Key new content:**
- Module I/O with math: V ∈ ℝ^{T×N×D}, Q_t ∈ ℝ^{K×D}, L_raw ∈ ℝ^{K×C_triplet}, c/r/w ∈ ℝ^D
- DSDAG: Xs (early frames) → Y → Xe (late frames), mediated by Z (scene embedding)
- Core equation: L_final = L_raw ⊙ f(r)
- Entropy-gated fusion: trusts causal head when structured head entropy is high
- Three representation spaces: Perception / Structured Event / Causal Reasoning
- Defense answers: "Where is Z?" (= f_v from CLIP-ViP, free, no separate module) / "Where do Xs/Xe come from?" / "Why not just constraints?"

---

## [2026-04-07] UPDATE — OpenMixer inference clarification + VLT supervision strategy

**Pages updated: 2**
- `methods/multimodal-causal-driving.md` — Module 2C (DFA): clarified E_text is precomputed CLIP embeddings of action class names (fixed lookup table, not runtime text); Module 6 (VLT): added three-stage language supervision solution
- `directions/constrained-vlm-reasoning.md` — training strategy expanded to three stages with rationale

**Key new content:**
- DFA at inference: `video → CLIP-ViP → V → OpenMixer → DFA(Q_t, E_text) → action scores` — E_text is a constant, always available
- VLT Stage 1: BDD-X + CoVLA pre-training (language grounding, real captions)
- VLT Stage 2: ROAD-Waymo fine-tuning via label reconstruction proxy (no captions required)
- VLT Stage 3: optional Claude pseudo-captions for corner cases (Batch API)
- Defense answer: ROAD-Waymo language output is zero-shot emergent from Stage 1, not a training target on ROAD-Waymo

---

## [2026-04-08] INGEST — Pearl 2009 Causality (Chapters 1, 3, 7)

**Source:** https://archive.illc.uva.nl/cil/uploaded_files/inlineitem/Pearl_2009_Causality.pdf

**Pages created: 1**
- `papers/pearl-2009-causality.md` — full treatment: DAGs, d-separation, Bayesian networks, causal Bayesian networks, SCMs/structural equations, do-calculus, counterfactuals, ladder of causation; connection table mapping Pearl concepts to DSDAG; custom causal graph design thought for ROAD-Waymo

**Key content encoded:**
- Three rungs: Association P(Y|X) / Intervention P(Y|do(X)) / Counterfactual P(Y_x|X=x',Y=y')
- d-Separation: chain/fork blocked by conditioning on middle; collider OPENED by conditioning (Berkson's paradox)
- SCM structural equations: xi = fi(PAi, Ui) where Ui = unmeasured factors
- W in DSDAG = Pearl's U — error/disturbance term representing unmeasured causal factors
- Backdoor criterion: control set Z blocks all confounding paths from X to Y
- Custom causal graph sketch: Road Context → [Agent State + Social Context] → Intent → Decision → Behavior
- Pearl's argument: t-norm constraints operate at Rung 1 (associational); DSDAG operates at Rung 2

---

## [2026-04-08] INGEST — approach3_architecture.md (full document)

**Source:** `ROAD_Reason/docs/approach3_architecture.md`

**Pages created: 1**
- `comparisons/openmixer-vs-retinanet.md` — full backbone gap analysis: spatial (FPN vs patch), temporal (3D conv vs T-OMB), semantic (Kinetics vs CLIP-ViP), tube formation engineering gap, feature semantics for DSDAG

**Pages updated: 2**
- `methods/multimodal-causal-driving.md` — major expansion:
  - Pipeline diagram updated with attention pool, projection layer, joint loss equation
  - Module 3 (Structured Head): five sigmoid heads with exact dimensions (10/19/10/49/86), multi-label rationale, t-norm compatibility rationale
  - Module 5 (DSDAG) restructured as 5A/5B/5C: attention pooling, projection layer, DSDAG with W detail
  - Joint training loss: `L_total = L_det + λ₁·L_tnorm + λ₂·L_language + λ₃·L_sparse`
  - Training stage rationale (why order is load-bearing)
  - Claude pseudo-label rationale (image required to break circularity)
  - Dual-stream extension note (motion gap mitigation for DSDAG)
  - Fixed stale "Where is Z?" defense answer
  - Links to comparison page and VLM architectures concept
- `index.md` — added openmixer-vs-retinanet comparison entry

**Key new content:**
- Action classes in structured head: 19 (confirmed correct — 22 total in dataset minus Red/Amber/Green which are TL signal states, not agent behavior actions)
- Structured head input rationale added (Module 3): why Q_t alone, not Q_s or f_v explicitly; paper ablation Table 4 cited; RetinaNet head comparison table; location label semantic classification rationale
- Attention pooling learns causal salience, not mean pooling
- Projection layer is required; MCAM weights NOT transferable (different feature space)
- Hidden danger state W: the representation-level mechanism for causal disambiguation
- Tube linking is the highest-severity engineering gap for video-mAP

---

## [2026-04-07] INGEST — VLM Architectures concept page

**Source:** Explanation generated in conversation

**Pages created: 1**
- `concepts/vlm-architectures.md` — three VLM patterns (contrastive, generative, cross-attention fusion); DFA deep dive; how CLIP-ViP / DFA / VLT / SmolVLM map to patterns in MCDM

**Pages updated: 2**
- `index.md` — added vlm-architectures entry under Concepts
- `methods/multimodal-causal-driving.md` — Module 2C: added link to vlm-architectures concept page

**Key new content:**
- Pattern 1 (Contrastive / CLIP): shared embedding space, dot-product similarity, zero-shot classification
- Pattern 2 (Generative / LLaVA-style): vision encoder → projection → LLM → text output
- Pattern 3 (Cross-attention fusion / VLT): fixed-size reasoning embedding r, differentiable, gates downstream predictions
- DFA mechanism: Q_t · E_text^T as open-vocab action scoring; E_text = fixed precomputed CLIP embeddings
- Connection table: which MCDM module uses which VLM pattern and why

---

## [2026-04-14] PROJECT — Experiment 1 setup: Qwen2.5-VL on ROAD-R

- Pages created: 0  |  Pages updated: 0  |  Code written: 4 new files
- SmolVLM scrapped per user decision; Experiment 1 now uses Qwen2.5-VL-7B
- Code written at `/data/repos/ROAD_Reason/experiments/exp1_road_r/`:
  - `config.py` — all hyperparameters, paths, label dims
  - `dataset.py` — ROADWaymoDataset: 27,971 train clips (clip_len=8, stride=4); triplet lookup via string parsing (all 86 resolved)
  - `model.py` — QwenROADModel: frozen ViT → ROI-average-pool (GT boxes) → TubeLinkingModule (cross-frame MHA) → 5× sigmoid heads
  - `losses.py` — ROADLoss: BCE (5 heads) + Łukasiewicz t-norm; loss test passed
  - `train.py` — training loop with AdamW + cosine LR + warmup + checkpointing
- environment.yml updated: transformers>=5.0, peft>=0.11, qwen-vl-utils added
- Pending: pip install peft + qwen-vl-utils, download Qwen/Qwen2.5-VL-7B-Instruct, first training run

## [2026-04-15] INGEST — Staged detection design decision

- Pages updated: 2 (methods/qwen25-vl-multitask.md, directions/qwen-multitask-finetuning.md)
- Documented why Experiment 1 uses GT-box ROI-pooling instead of learned detection:
  decouples detection from classification to isolate failures, provides classification ceiling,
  warm-starts heads for Experiment 1b
- Added Experiment 1b entry: detection head added on top of Exp 1 checkpoint
- Documented Option B (joint from scratch) and when it becomes appropriate
