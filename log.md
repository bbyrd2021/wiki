---
type: log
---

# Wiki Operations Log

Append-only record of all wiki operations. Format: `## [YYYY-MM-DD] OPERATION — Description`

---

## [2026-05-11] INGEST — LeWorldModel (Maes et al., arXiv 2603.19312)

- Pages created: 1 (`papers/maes-2026-lewm.md`)
- Pages updated: 3 (`directions/lewm-scene-prediction.md`, `index.md`, `log.md`)
- Buds harvested: 15 new (I-JEPA, V-JEPA, V-JEPA 2, DINO-WM, PLDM, VICReg, LeJEPA, Causal-JEPA, DreamerV3, TD-MPC2, DINOv2, DIAMOND, IRIS, LeCun "Path to AMI", Navigation World Models)
- Source: `wiki/raw/LeWM.pdf` (Mila / NYU / Samsung SAIL / Brown; LeCun + Balestriero co-authors; 24 Mar 2026)
- Summary: First JEPA trained stably end-to-end from raw pixels with only 2 loss terms — next-embedding MSE + SIGReg anti-collapse. ViT-tiny encoder + 6-layer transformer predictor (~15M params) trainable on a single GPU in hours. Plans 48× faster than DINO-WM (0.98s vs 47s) while matching/beating it on Push-T / OGBench-Cube / Two-Room / Reacher continuous control. Single tunable hyperparameter (λ) vs PLDM's 6. Provides the published reference for [[directions/lewm-scene-prediction|Approach 6]] in ROAD_Reason, making the previously-speculative direction concrete: workstation-feasible JEPA, action-conditioned latent prediction, violation-of-expectation surprise as a free anomaly detector for driving.

---

## [2026-05-11] UPDATE — Exp2c training progress + backbone size comparison
- Pages updated: 3
  - `findings/exp2c-frozen-detr.md` — added training results table (14 epochs complete, ep15 in progress), GIoU trend analysis, backbone size comparison vs 3D-RetinaNet (ResNet-50 I3D 46M vs EfficientNet-B0 5.3M = 9x gap), scaling implications
  - `methods/3d-retinanet.md` — added backbone param counts (~46M ResNet-50 I3D, ~50-55M total trainable), f-mAP validation details, size comparison section
  - `projects/road-reason.md` — updated Exp2c status from "Ready to train" to "Training ep15/30" with current metrics
- Key finding: localization (GIoU) steadily improving every epoch (0.793→0.596) with no plateau. f-mAP eval at epoch 15 will determine if architecture validates against RetinaNet's 17.76% baseline. If comparable with 9x smaller backbone, justifies dual-backbone Frozen-DETR approach.

## [2026-05-08] INGEST — SLURP WavLM-Hier full repo review
- Pages created: 5
  - `methods/wavlm-hier.md` — architecture spec for the 5-component novel method (WavLM-Large + attention pooling + hierarchical scenario→action conditioning + curriculum teacher forcing + ontology masking)
  - `findings/slurp-wavlm-hier-results.md` — best run (`wavlm_large_hier_v3`, 2026-03-26): Sc F1w 0.877 / Act F1w 0.833 / Intent Acc 0.820, plus encoder ablation (base-plus +0.109 intent), hierarchy ablation (flat +0.024 intent), per-class F1
  - `findings/slurp-collapse-e1.md` — failure baseline lifted from `E1_results_summary.md`: frozen Wav2Vec2 collapses to majority class within epoch 1, 17/18 scenario + 45/46 action classes get zero F1, reproduces Phase 3 paper to 3 decimals
  - `comparisons/slurp-audio-vs-text-oracle.md` — surprising headline: WavLM-Hier on raw audio beats RoBERTa+gold-transcript on every metric (+0.28 Act F1w, +0.279 Intent Acc); attributable to prosody carrying intent that normalised transcripts discard
  - `concepts/encoder-collapse.md` — generalises the E1 failure pattern with diagnostic checklist + cross-domain transfer (ROAD++ long-tail actions, JAAD crossing, frozen-CLIP detection)
- Pages updated: 2 (`projects/slurp.md` overhaul, `index.md` × 6 edits, `log.md`)
- Triggered by user query about "WavHeir" status — clarified that the actual model name is WavLM-Hier (script `contrib/run_whisper_hier.py` with `ENCODER_NAME = "microsoft/wavlm-large"`).
- Status of all 4 hier runs: v1 (Sc F1w 0.852), v2 (no test eval, superseded), v3 (best, 0.877), flat-ablation (0.874 — hierarchy contributes mainly to joint Intent Acc not per-task F1).
- Quirk noted: `wavlm_large_hier_v3/final_results.json` records `run_id: whisper_hier` because the script's hardcoded run_id wasn't updated when switching encoders. The `encoder` field is the ground truth.
- Publishing readiness assessed in `projects/slurp.md` Publishing Readiness section: ~60% to submit. Gaps blocking: (1) Wang et al. 2021 89.38% intent acc still beats this work — likely fix is adding slot-filling head; (2) no slot head means SLURP-F1 not measurable; (3) single seed (42); (4) no paper draft yet; (5) focal-loss ablation incomplete. Realistic target: IEEE SLT 2026 (deadline ~July). Path-to-submit estimated 3–4 weeks.

---

## [2026-05-08] INGEST — YOLOv10-BDD13 extension (deer, cone, barrier) for AutoDrive
- Pages created: 1 (`findings/yolov10-bdd13-extension.md`)
- Pages updated: 1 (`index.md`)
- Source-repo changes (YOLO_BDD): 4 new scripts — `auto_label_deer.py` (Grounding DINO zero-shot labeling), `download_roboflow.py` (Roboflow Universe SDK helper), `build_merged_dataset.py` (BDD train JSON→YOLO + symlinked merge + class-ID remap), `train_yolov10_13class.py` (50-epoch fine-tune entrypoint).
- Datasets added: Christopher's deer photos (744 imgs, auto-labeled to 1017 boxes), Roboflow `safety-cones-vfrj2` (5960 train / 341 val / 48k cone instances; autocross domain), Roboflow `jersey-barrier` (716 train / 203 val / 1965 instances). Merged dataset at `/data/datasets/bdd13-extended/` via symlinks: 77,272 train / 10,692 val.
- Training: 7h 43m on 1× A6000 (50 epochs, batch=64, AdamW auto lr=0.000588), peak 19.4GB GPU. No early-stop; mAP plateaued ~epoch 42. `Transferred 607/619 items from pretrained weights` confirms backbone+neck preserved while head re-init for 13 classes.
- Headline result: **mAP50 = 0.602 / mAP50-95 = 0.368** all-13. New classes lead the table — barrier 0.974, deer 0.876, cone 0.701. **BDD-10 mean improved 0.524 → 0.527** vs the original 10-class baseline; only motorcycle regressed meaningfully (-0.024).
- Caveats logged: barrier 0.974 is almost certainly train/val frame leakage (716 frames from a few YouTube clips, random per-frame split); cone domain is autocross not public road; deer training set is wildlife stock photography not roadside — driving-domain val mAP for deer is the real number to chase.
- Best checkpoint: `/data/repos/YOLO_BDD/runs/detect/yolov10s-bdd13/weights/best.pt` (epoch 42).
- Decision: drop into `AutoDrivePerception2026/yolov10_ros/models/`. Ultralytics reads `model.names` dynamically so no ROS code changes needed; only `target_class` strings for the downstream classifiers stay the same.

---

## [2026-05-07] INGEST — Traffic light v2 retrain (+ S2TLD), three-way A/B
- Pages updated: 2 (`findings/traffic-light-domain-shift.md`,
  `RIG_AB_HANDOFF.md` at the source-repo root).
- Source-repo changes: `tools/s2tld_to_patches.py` (new VOC-XML adapter),
  re-runs of `merge_patch_datasets.py` and `train.py`.
- Dataset added: S2TLD (SJTU Small Traffic Light, MIT license, Zenodo via
  HuggingFace). 5,786 images / 14,130 boxes across two resolution variants
  (1080x1920 + 720x1280 normal_1 + normal_2). Patches: 12,418 train + 1,397
  val. Merged total: 69,805 train / 25,856 val. Skipped `wait_on` class
  (composite multi-light state).
- Training: 30 min on A6000, early-stopped epoch 30 (peak 20),
  **best val 96.44%** (vs v1 96.05%, old 96.22%-on-6cls-subset).
- Three-way per-domain comparison reveals that v2 is best in aggregate
  (macro F1 0.74 → 0.80) and dramatically better on LISA + S2TLD, **but
  v2's BSTLD red recall regressed further (80.9% → 74.7%)**. S2TLD's red
  distribution is far enough from BSTLD's that adding it pulled the
  decision boundary further away. The "more data fixes red recall"
  hypothesis was partially wrong: it fixed it on most domains, made it
  worse on BSTLD specifically.
- LISA red→yellow rate dropped 1% → 0.02% (basically eliminated for
  consumer-dashcam-style domains).
- Decision: ship both v1 and v2 checkpoints, A/B at the rig. Rig agent
  picks empirically since `sensor_data.bag` domain isn't characterized.
- Created unified `RIG_AB_HANDOFF.md` covering both checkpoints, the
  comparison numbers, and an A/B procedure. Single launch-arg flip swaps
  models. Rolled the audit findings into this same doc.

---

## [2026-05-07] AUDIT — yolov10_ros rig repo inference logic
- Pulled `bbyrd2021/yolov10_ros` `main` from GitHub via `gh api`. Audited
  `src/light_classifier_node.py`, `src/yolov10_ros/image_utils.py`,
  `config/light_classifier_params.yaml`, `launch/pipeline.launch`.
- **Mapping is correct** for the deployed 6-class checkpoint (alphabetical
  `green_left, green_light, red_left, red_light, yellow_left, yellow_light`).
- **HSV off-signal gate is NOT in the committed code** — session notes
  pointed at `:252-264` but those lines are just publish-and-spin. The gate
  was a local rig-only edit, never committed. Original handoff doc step
  ("delete the gate") removed.
- **`_INPUT_SIZE = 224` is a bug** at `src/light_classifier_node.py:77`.
  The classifier was trained at 128. EfficientNet works at any input size
  (no rigid input layer; GAP handles any feature map), so the rig runs
  silently — but features learned at 128 don't transfer ideally to 224.
  Likely contributed to original red→yellow rig failure independent of
  taxonomy. **Should be fixed regardless of which checkpoint ships.**
- **Deploy is YAML-only**, not Python edits: the node already supports a
  `~class_names` ROS param that overrides `_DEFAULT_CLASSES` and rebuilds
  the classifier head dynamically. New 7-class taxonomy ships via a
  `class_names:` block in `light_classifier_params.yaml` + a one-line edit
  to `pipeline.launch` for the new `.pth` path. No Python diff.
- Bonus: `crop_roi` uses symmetric pixel padding (5 default) vs training's
  multiplicative 1.2× expansion. Small lights get over-padded at inference
  relative to training. Flagged for follow-up, not blocking.
- Updated `RIG_HANDOFF.md` and `findings/traffic-light-domain-shift.md`
  with the corrected deploy path.

---

## [2026-05-07] INGEST — Traffic light classifier retrain (LISA + BSTLD, 7-class with `off`)
- Pages created: 1 (`findings/traffic-light-domain-shift.md`)
- Pages updated: 3 (`projects/eff-light-detection.md`, `tools/traffic-light-classification.md`, `index.md`)
- Source-repo changes (under `eff_light_detection/`, sync'd separately to rig):
  `tools/bstld_to_patches.py` (new), `tools/merge_patch_datasets.py` (new),
  `train.py` (intra-epoch stdout progress prints), `class_index.json` (now 7-class).
- Dataset added: BSTLD train RGB (Bosch Small Traffic Lights, Zenodo 12706046),
  5,093 images / 10,756 boxes, 1280×720, non-commercial license.
- Merged training set: 57,387 train / 24,459 val patches across 7 classes
  (dropped `green_straight`; added `off` from BSTLD only).
- Trained on RTX A6000, EfficientNet-B0 from ImageNet pretrained, 21 min wall
  clock, early-stopped at epoch 25 (peak at 15). **Best val acc: 96.05%**.
- Critical confusion rates: overall red_light → yellow_light **0.62%**, off →
  yellow_light **0%**, off recall **93.3%**, red_light F1 **0.988**.
- Old vs new comparison on the same val (24,414 non-`off` patches): old
  96.22% / new 96.06% overall — within-domain LISA basically a wash. On the
  BSTLD subset (818 patches, OOD analog), new model wins 90.10% vs 88.14% —
  but with a *trade-off*: BSTLD red_light recall dropped 86.4% → 80.9% (some
  reds re-routed to `off` / `yellow`). Off-handling: old model routes 78% of
  dark housings to `green_light` (not `yellow_light` as session notes
  suggested), new model 93% correct. The dark-housing fix is unambiguous;
  the red→yellow fix is partial — adding a Roboflow red/yellow source is the
  proposed next step.
- Deliverables: `experiments/efficientnet_b0_20260507_134855/{best.pth, eval/, RIG_HANDOFF.md}` + repo-root `class_index.json`.
- Status: complete. `RIG_HANDOFF.md` ready for the rig agent (paths, code
  edits to `light_classifier_node.py`, verification steps, rollback plan).

---

## [2026-05-07] INGEST — Exp2c Frozen-DETR implementation
- Pages created: 1 (`findings/exp2c-frozen-detr.md`)
- Pages updated: 5 (`projects/road-reason.md`, `papers/fu-2024-frozen-detr.md`, `concepts/vlm-localization-gap.md`, `comparisons/fusion-for-detection-lit-review.md`, `index.md`)
- Summary: Documented Exp2c — faithful Frozen-DETR implementation replacing Exp2b's scalar gate fusion with 6-layer deformable encoder (4 scales: P3+P4+P5+CLIP) and per-layer CLS injection in decoder. CLIP ViT-L/14 replaces Qwen2.5-VL (Dr. Moradi approved). ~445M params, ~15.7M trainable, ~5 GB GPU savings. Implementation complete, smoke test passed, awaiting first training run.

---

## [2026-05-06] CREATE — CNN-VLM Fusion Lit Review (draft)
- Pages created: 1 (`comparisons/fusion-for-detection-lit-review.md`)
- Pages updated: 1 (`index.md`)
- Summary: Started literature review on CNN-VLM fusion methods for detection + constraint reasoning, motivated by Exp2b's scalar gate bottleneck. First entry: VMCNet (Gao et al., 2025) — per-channel FiLM modulation from frozen ViT to trainable CNN. Structured for incremental additions as papers are reviewed.

---

## [2026-05-06] INGEST — Fusion methods, constraint guarantees, and action detection papers (10 new)
- Pages created: 10
- Pages updated: 1 (`index.md`)
- Summary: Literature survey on improving Exp2b's VLM-CNN fusion and validating the detection-then-constraint pipeline. Three research tracks: (1) Feature fusion — Frozen-DETR (NeurIPS 2024), VMCNet (arXiv 2025), CBAM (ECCV 2018), FiLM (AAAI 2018), ViT-Adapter (ICLR 2023), ViT-CoMer (CVPR 2024); (2) Constraint guarantees — PiShield (IJCAI 2024, hard guarantee Shield Layers replacing t-norm loss), Efficient T-norms (NeSy 2023, sparse memory-efficient formulation); (3) Action detection — OpenMixer (WACV 2025, planned Approach 4 backbone), MCAM/DSDAG (ICCV 2025, causal reasoning module). PDFs downloaded to `ROAD_Reason/papers/`.

---

## [2026-05-04] EVAL — Exp2b baseline-compatible f-mAP results
- Pages updated: 2 (`findings/exp2b-deformable-detr.md`, `projects/road-reason.md`)
- CSV updated: `results/val_metrics.csv` — Exp2b ep26 best.pt results appended
- Summary: Exp2b (Deformable DETR + EfficientNet + iterative refinement) evaluated with baseline-compat f-mAP at IoU=0.5. Agent=1.71% (2.7x better than Exp2's 0.63%, but still 10x below RetinaNet 17.76%). Recall is very high (48-62% across all heads) confirming detection coverage is not the issue — box precision at IoU=0.5 is the bottleneck. Key conclusion: VLM features (even augmented with CNN spatial features) are fundamentally unsuitable as detection backbones. This validates Approach 4's architectural split: dedicated detector (OpenMixer) for localization, VLM for downstream reasoning.

## [2026-04-27] INGEST — Exp2b Deformable DETR with EfficientNet + Iterative Refinement
- Pages created: 1 (`findings/exp2b-deformable-detr.md`)
- Pages updated: 3 (`projects/road-reason.md`, `findings/exp2-detr-detection.md`, `index.md`)
- Summary: Documented Exp2b — redesign of Exp2's decoder as standard Deformable DETR (Zhu et al., ICLR 2021) with three fixes: (1) per-frame decoding with temporal self-attention replacing temporal stacking, (2) iterative box refinement with per-layer box heads in inverse-sigmoid space, (3) auxiliary decoder losses at every layer. Adds EfficientNet-B0 + FPN as spatial backbone alongside Qwen ViT, fused via learned gates. 300 queries (up from 100), 692M total params (15.6M trainable). Warm-started from Exp2 best.pt. Training started 2026-04-27, 30 epochs. Eval pipeline includes agentness > 0.01 pre-filter for ~90% speedup. Updated experiment status table and marked Exp2 as superseded.

## [2026-04-24] INGEST — VLM Localization Gap concept page
- Pages created: 1 (`concepts/vlm-localization-gap.md`)
- Summary: Documents why frozen ViT features produce poor detection localization (single-scale, no FPN, semantic-over-spatial). Covers ViTDet, ViT-CoMer (CVPR 2024), Frozen-DETR (NeurIPS 2024), Deformable DETR. Four architectural options analyzed; recommends EfficientNet CNN + VLM enrichment + Deformable DETR (Option B+D).

## [2026-04-24] EVAL — Exp2 fixed f-mAP results (best.pt, ep26)
- Pages updated: 1 (`findings/exp2-detr-detection.md`)
- CSV updated: `results/val_metrics.csv` — replaced buggy ep15 row with fixed ep26 results
- Summary: Fixed eval pipeline (8-frame clips, no double-gating, no pre-filtering) produced: agent_ness=2.08%, agent=0.63%, action=0.76%, loc=1.00%, duplex=0.14%, triplet=0.80%. All heads improved ~1.5-2× over the buggy eval but remain an order of magnitude below 3D-RetinaNet. Recall is healthy (20-31% mR) — localization quality at IoU=0.5 is the bottleneck, not detection coverage.

## [2026-04-24] FIX — Exp2 eval pipeline audit + correction
- Pages updated: 1 (`findings/exp2-detr-detection.md`), 1 (`results/val_metrics.csv`)
- Code modified: `experiments/exp2_detr_qwen/eval_baseline_compat.py` — 3 bugs fixed
- Summary: Full audit of eval code revealed all prior Exp2 f-mAP numbers were non-comparable with the baseline due to (1) single-frame inference (model trained on 8-frame clips), (2) score double-gating (agentness×class vs raw sigmoid), (3) aggressive pre-filtering. The "124% agent_ness" was not a normalization bug — it was 1.24% mAP (JSON values already in percentage form, incorrectly ×100'd to CSV). All prior Exp2 f-mAP values retracted; re-eval pending with fixed pipeline. Also retracted the prior session's ep15 results + analysis that were based on inflated numbers.

## [2026-04-24] UPDATE — Exp2 DETR duplex loss analysis (retracted — see FIX entry above)
- Summary: Duplex loss analysis section still valid; f-mAP numbers it referenced were not.

## [2026-04-22] EVAL — SparseTemporalPIE v=0 Stationary Subset
- Pages updated: 1 (`findings/sparse-temporal-pie-results.md`, new Section 7)
- Summary: Ran `eval_vzero_comparison.py` comparing EfficientPIE (replicated) and SparseTemporalPIE v3 on the PIE test set full vs. v=0 stationary subset (871/893 samples, epsilon=5.0). v3's AUC margin widens on v=0 (+0.0279 vs +0.0268 full set), confirming pose/gaze features contribute when kinematics are zero. Key finding: 97.5% of the PIE test set is already v=0 — the main SOTA numbers already represent the stationary-pedestrian regime.

## [2026-04-21] INGEST — SparseTemporalPIE Full Results Narrative
- Pages created: 1 (`findings/sparse-temporal-pie-results.md`) | Pages updated: 1 (`index.md`)
- Summary: Created comprehensive results page from EfficientPIE/docs/RESULTS.md. Covers full experimental setup (PIE/JAAD splits, IDIL protocol, hyperparameters), v3 architecture narrative, complete SOTA comparison tables for both datasets, v3 vs v4 ablation (cross-attention contribution), backbone initialization ablation (EfficientPIE vs ImageNet), IL step progression tables for PIE and JAAD, and discussion with key findings and limitations. Existing method/project pages remain as quick-reference summaries.

## [2026-04-21] INGEST — Exp2 DETR-Style Detection Design + Exp1b Baseline-Compat Results
- Pages created: 1 (`findings/exp2-detr-detection.md`) | Pages updated: 3 (`findings/exp1b-fcos-detection.md`, `projects/road-reason.md`, `index.md`)
- Summary: Documented Exp2 (DETR-style set-prediction detector — 100 learnable queries, Hungarian matching, L1+GIoU loss, clip-level spatiotemporal attention). Filed exp1b's baseline-compatible f-mAP results (agent=3.2% vs internal 60.6%) and explained the evaluation methodology gap: FCOS box quality under IoU=0.5 is the bottleneck, not classification accuracy. Added Exp2 to experiment status table in road-reason.md. Exp2 code reviewed and all bugs fixed (t-norm flat vector offset, real agentness head); ready to train.

## [2026-04-17] INGEST — Exp1b FCOS Dense Detection Design
- Pages created: 1 (`findings/exp1b-fcos-detection.md`) | Pages updated: 2 (`projects/road-reason.md`, `index.md`)
- Summary: Documented full Exp1b experiment design — FCOS-style dense detection replacing oracle-box Exp1 pipeline. Covers architecture (DetectionHeads, 7 sub-heads), FCOS token assignment, 4-term loss (agentness focal + box SmoothL1 + classification focal + Gödel t-norm), training config, warm-start strategy, first clip training log, and expected comparisons against Exp1 and 3D-RetinaNet baseline. Training started 2026-04-17, in progress.

## [2026-04-16] PROJECT — Experiment 1b design and implementation

- Pages created: 0  |  Pages updated: 0  |  Code written: 5 new files
- Designed and implemented Exp1b at `/data/repos/ROAD_Reason/experiments/exp1b_road_r/`
- Changes from Exp1: focal loss (γ=2, dynamic per-class α), LoRA on first 8 ViT blocks (r=8), Gödel t-norm (λ=1.0), two optimizer param groups, best checkpoint by action macro-mAP
- Warm-starts from Exp1 best.pt (epoch 6); LoRA added after weight loading to preserve base weights
- Files: `config.py`, `losses.py` (FocalLoss + compute_class_alphas), `model.py` (LoRA via peft), `train.py` (two param groups + mAP criterion), `eval.py` (LoRA-aware)
- Import sanity check passed: all modules resolve correctly

## [2026-04-16] INGEST — Exp1 Evaluation Results
- Pages created: 1 (`findings/exp1-vs-retinanet-baseline.md`) | Pages updated: 1 (`index.md`)
- Summary: Filed Exp1 (Qwen2.5-VL, epoch 6 best.pt) eval results against locally-replicated 3D-RetinaNet-I3D baseline. Key findings: Qwen stronger on agent/action/loc with GT boxes but weaker on duplex/triplet; t-norm negligible in both models; class imbalance root cause for rare action F1≈0.

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

## [2026-04-20] INGEST — Exp1b Eval Results

- Pages updated: 3 (findings/exp1b-fcos-detection.md, projects/road-reason.md, index.md)
- Summary: Exp1b FCOS dense detection training (15 epochs) complete. Final eval: agent=60.6%, action=32.4%, loc=50.0%, duplex=23.1%, triplet=17.5% macro-mAP on fg tokens. Beats Exp1 oracle-box on all 5 heads. Constraint violation rate=0.29%. Model still improving at epoch 15 — no convergence plateau. Next step: tube linking for ECCV-comparable f-mAP / v-mAP.

## [2026-05-11] INGEST — VL-JEPA (Chen et al., arXiv 2512.10942)

- Pages created: 1 (`papers/chen-2026-vl-jepa.md`)
- Pages updated: 3 (`index.md`, `directions/jepa-intent-head.md`, `log.md`)
- Source: `wiki/raw/VL-JEPA.pdf` (Meta FAIR / HKUST / Sorbonne / NYU; LeCun is co-author; Feb 2 2026)
- Summary: First JEPA-style VLM — predict the *answer embedding* (Llama-3 predictor over V-JEPA 2 ViT-L + EmbeddingGemma-300M Y-Encoder, InfoNCE in 1536-dim shared space) instead of generating tokens. 1.6B params outperforms CLIP/SigLIP2/PE-Core on 8 classification + 8 retrieval benchmarks (motion-centric: SSv2, EK-100, EgoExo4D), matches Qwen-VL/InstructBLIP on VQA, new 65.7% SOTA on WorldPrediction-WM. Native selective decoding gives 2.85× fewer decode calls in streaming. Direct architectural relative of Approach 5 ([[directions/jepa-intent-head|V-JEPA intent head]]) and a latent-space alternative to the token-generative Approach 3 ([[methods/qwen25-vl-multitask|Qwen2.5-VL]]) and Approach 4 ([[methods/multimodal-causal-driving|MCDM]]).
