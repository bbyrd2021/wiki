---
type: index
updated: 2026-04-07
total_pages: 82
---

# Research Wiki — Index

All pages organized by category. One line per page: link + one-sentence summary.

---

## Projects (12)

### PhD Core
- [[projects/pedestrian-intent|PedestrianIntent++]] — 3-dataset synthesis of JAAD, PIE, ROAD++ covering annotations, cross-tabs, literature survey, and 6 research directions
- [[projects/road-reason|ROAD_Reason]] — logic-constrained scene reasoning on ROAD++ via neuro-symbolic VLM with t-norm losses (primary thesis contribution)
- [[projects/efficient-pie|EfficientPIE / SparseTemporalPIE]] — multi-frame cross-attention extension of IJCAI-25 EfficientPIE; v3 achieves 0.926 accuracy / 0.947 AUC on PIE

### Detection / Segmentation
- [[projects/auto-drive-perception|AutoDrivePerception2026]] — ROS 1 pipeline: YOLOv10 detection → traffic light and sign classifiers
- [[projects/yolopx|YOLOPX]] — anchor-free panoptic driving perception (detection + segmentation + lane) on BDD100K, 47 fps
- [[projects/yolo-bdd|YOLO_BDD]] — YOLOv10 fine-tuned on BDD100K (10 classes); feeds downstream classifiers

### Foundations / Tools
- [[projects/bevfusion|BEVFusion]] — camera + LiDAR bird's-eye view fusion framework (ICRA 2023, former Waymo SOTA)
- [[projects/mmdetection3d|mmdetection3d]] — OpenMMLab 3D detection toolbox supporting 40+ methods across KITTI, nuScenes, Waymo
- [[projects/twinlitenet|TwinLiteNet]] — lightweight dual-task segmentation + lane detection optimized for embedded hardware
- [[projects/eff-light-detection|eff_light_detection]] — 7-class EfficientNet-B0 traffic light classifier (LISA dataset)
- [[projects/eff-sign-detection|eff_sign_detection]] — 15-class EfficientNet traffic sign classifier (Mapillary MTSD v2)
- [[projects/slurp|SLURP]] — spoken language understanding research; methodology parallels (encoder collapse, class imbalance) apply to AD models

---

## Datasets (6)

- [[datasets/bdd-x|BDD-X]] — 7,000 videos, 26,538 action+justification pairs (manual); Stage 1 pre-training for W (reason mode); ECCV 2018
- [[datasets/covla|CoVLA]] — 10K clips, 6M frames, Tokyo; frame-level plain_caption (→Y) + risk (→W); full CAN bus + 60-pt trajectory; Stage 1 pre-training at scale
- [[datasets/jaad|JAAD]] — 346 videos, 686 behavioral pedestrians, 5-layer XML; 95.7% gaze-cross rate at decision_point
- [[datasets/pie|PIE]] — 53 videos, 1842 ped tracks, crowd-sourced intention_prob; gaze reversal vs JAAD
- [[datasets/road-plusplus|ROAD++]] — 798 videos, 41,935 agent tubes, 86-class compositional triplets, logic constraints
- [[datasets/bdd100k|BDD100K]] — 100K driving videos, 10-class detection + segmentation + lane benchmark

---

## Concepts (10)

### Pedestrian Intent
- [[concepts/crossing-intent|Crossing Intent]] — mental state (intent) vs. observed behavior (action); how each dataset defines and measures it
- [[concepts/gaze-and-attention|Gaze and Attention]] — the look label in JAAD/PIE; gaze means commitment in JAAD, hesitation in PIE
- [[concepts/intention-probability|Intention Probability]] — PIE's crowd-sourced continuous intent score; mean=0.712, median=0.850, bimodal
- [[concepts/action-look-crosstab|Action × Look Cross-Tabulation]] — methodology and key tables from the JAAD/PIE cross-tabulation analysis
- [[concepts/compositional-labels|Compositional Labels]] — ROAD++'s five-level hierarchy: agent → action → location → duplex → triplet
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — t-norm loss for enforcing logical requirements on model predictions; ROAD-R methodology
- [[concepts/vlm-architectures|VLM Architectures]] — three VLM patterns (contrastive, generative, cross-attention fusion) and how DFA, CLIP-ViP, and VLT fit into MCDM

### Broader Perception
- [[concepts/domain-shift|Domain Shift]] — sources of distributional mismatch across JAAD/PIE/ROAD++ and how models fail
- [[concepts/multi-task-perception|Multi-Task Perception]] — joint detection + segmentation + lane detection; end-to-end vs. modular approaches
- [[concepts/bev-fusion|BEV Fusion]] — camera + LiDAR bird's-eye view projection for 3D perception
- [[concepts/occlusion|Occlusion]] — occlusion annotations in JAAD/PIE; an active frontier for occlusion-robust intent models

---

## Papers (20)

### Causal Theory
- [[papers/pearl-2009-causality|Pearl 2009 — Causality]] — DAGs, SCMs, do-calculus, d-separation, counterfactuals; theoretical foundation for DSDAG; includes custom causal graph design thought for ROAD-Waymo

### Pretraining / VLA
- [[papers/covla-2025|CoVLA 2025]] — 10K clips, 6M frames, behavior + reasoning captions; Stage 1 pre-training source for Approach 3 causal head; reasoning captions supervise W (reason mode) (arXiv:2408.10845)

### ROAD / ROAD++
- [[papers/singh-2022-road|Singh 2022 — ROAD]] — foundational ROAD paper; introduces compositional label framework and 3D-RetinaNet baseline (IEEE TPAMI 2022)
- [[papers/marconato-2022-road-r|Marconato 2022 — ROAD-R]] — 243 logic requirements + t-norm constraint loss for verifiable AV predictions (Machine Learning 2023)
- [[papers/salmank-2024-road-waymo|Salmank 2024 — ROAD-Waymo]] — ROAD++ / ROAD-Waymo dataset introduction; verified stats differ from paper claims (arXiv:2411.01683)
- [[papers/eccv24-track1|ECCV 2024 Track 1 — Agent Detection]] — ECCV challenge winner; 30.82% video-mAP; addresses small-object challenges (arXiv:2410.23077)
- [[papers/eccv24-track3|ECCV 2024 Track 3 — Atomic Activity]] — ECCV challenge winner; 69% mAP on 64-class pedestrian action recognition (arXiv:2410.23092)

### JAAD / Both
- [[papers/xu-2025-gtranspdm|Xu 2025 — GTransPDM]] — graph Transformer + positional decoupling; 92% PIE / 87% JAAD / 0.05ms (IEEE SPL 2025)
- [[papers/rasouli-2024-intentformer|Rasouli 2024 — IntentFormer]] — co-learning Transformer; 93% PIE / 92% JAAD (Pattern Recognition 2024)
- [[papers/rasouli-2023-pedformer|Rasouli 2023 — PedFormer]] — cross-modal attention + gated multitask; SOTA at ICRA 2023 (arXiv:2210.07886)
- [[papers/context-aware-multitask-2025|Context-Aware Multitask 2025]] — joint intent + trajectory with JAAD traffic layer context (Transportation Research Part C 2025)
- [[papers/group-crossing-2025|Group Crossing 2025]] — pedestrian group graph model for JAAD/PIE (Scientific Reports 2025)
- [[papers/occlusion-graph-2025|Occlusion Graph 2025]] — graph autoencoder for partial occlusion; +6% on JAAD under occlusion (EAAI 2025)
- [[papers/occlusion-diffusion-2024|Occlusion Diffusion 2024]] — diffusion model for trajectory reconstruction under occlusion; +5% (arXiv:2511.00858)
- [[papers/efficientpie-ijcai-2025|EfficientPIE — IJCAI 2025]] — lightweight single-frame intent model; upstream of local SparseTemporalPIE work

### PIE
- [[papers/rasouli-2024-encore|Rasouli 2024 — ENCORE]] — scale- and motion-aware trajectory prediction; stratified evaluation paradigm (ICRA 2024)
- [[papers/rasouli-2024-deeper|Rasouli 2024 — Deeper]] — intent + action + risk assessment under scenario-stratified metrics (IV 2024)
- [[papers/he-2022-ista-net|He 2022 — ISTA-Net]] — spatio-temporal graph + GAT + Transformer; 89.5% acc / 88.32% F1 on PIE (ICRA 2022)
- [[papers/pip-net-2024|PIP-Net 2024]] — in-the-wild generalization study + domain adaptation on PIE (arXiv:2402.12810)
- [[papers/vlm-pie-intent-2025|VLM PIE Intent 2025]] — CLIP/LLM scene description as context; PIE SOTA mid-2025 (arXiv:2507.04141)
- [[papers/ssl-lanes-2023|SSL-Lanes 2023]] — SSL pretraining on PIE OBD; +4% intent accuracy from self-supervised pretraining (CoRL 2023)

---

## Methods (9)

- [[methods/3d-retinanet|3D-RetinaNet]] — Kinetics-pretrained anchor-based tube detector; ROAD/ROAD++ baseline for all challenge comparisons
- [[methods/pedformer|PedFormer]] — cross-modal Transformer + gated multitask; JAAD/PIE SOTA at ICRA 2023
- [[methods/gtranspdm|GTransPDM]] — graph Transformer + positional decoupling; 92% PIE / 87% JAAD / 0.05ms inference
- [[methods/intentformer|IntentFormer]] — co-learning Transformer over RGB + segmentation + trajectory; 93% PIE / 92% JAAD
- [[methods/ista-net|ISTA-Net]] — spatio-temporal graph (GAT + Transformer) over bbox + optical flow + OBD; 89.5% PIE
- [[methods/sparse-temporal-pie|SparseTemporalPIE]] — multi-frame cross-attention + pose + ctx MLP (local research); 0.926 acc / 0.947 AUC PIE
- [[methods/encore|ENCORE]] — scale-aware encoder + motion-conditioned Transformer decoder for PIE trajectory
- [[methods/smolvlm-road|SmolVLM Baseline]] — generative VLM baseline for ROAD_Reason zero-shot / constraint-prompted / GT-conditioned reasoning
- [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task]] — Approach 3: shared Qwen2.5-VL-7B backbone with three task heads (BDD-X captioning, ROAD-R tube detection + t-norm, CoVLA captioning + trajectory)
- [[methods/multimodal-causal-driving|Multimodal Causal Driving Model]] — full architecture spec for Approach 4: CLIP-ViP → OpenMixer → DSDAG + VLT → logit reweighting → t-norm; includes defense prep

---

## Research Directions (10)

- [[directions/uncertainty-aware-intent|Uncertainty-Aware Intent]] — regression on PIE's intention_prob; no paper has trained calibrated uncertainty matching human disagreement (PIE)
- [[directions/appearance-conditioned-intent|Appearance-Conditioned Intent]] — JAAD's 24 per-frame appearance attributes are unused by all surveyed papers; research opportunity (JAAD)
- [[directions/cross-dataset-generalization|Cross-Dataset Generalization]] — ROAD++ → JAAD → PIE domain shift study; no cross-dataset generalization paper spans all three
- [[directions/logic-constrained-intent|Logic-Constrained Intent]] — ROAD-R methodology applied to JAAD/PIE crossing behavior; unexplored (JAAD + PIE + ROAD++)
- [[directions/state-sequence-modeling|State-Sequence Modeling]] — Transformer over discrete (action, look) state sequences; Wait2X→Xing transition modeling (all three)
- [[directions/scene-context-transfer|Scene Context Transfer]] — ROAD++ compositional labels as pretraining objective for JAAD/PIE intent models; unexplored (all three)
- [[directions/qwen-multitask-finetuning|Qwen2.5-VL Multi-Task Fine-Tuning]] — Approach 3: shared VLM backbone fine-tuned on BDD-X, CoVLA, ROAD-R separately then jointly; SOTA baseline before causal architecture
- [[directions/constrained-vlm-reasoning|Constrained VLM Reasoning]] — OpenMixer + DSDAG + VLT + t-norm on ROAD++; primary thesis contribution (ROAD_Reason Approach 4)
- [[directions/jepa-intent-head|V-JEPA 2 Intent Head]] — V-JEPA 2 encoder + lightweight intent head on ROAD++; novel application (ROAD_Reason Approach 5)
- [[directions/lewm-scene-prediction|LeWM Scene Prediction]] — workstation-feasible world model (15M params, single GPU) for ROAD++ future state prediction (Approach 6)

---

## Findings (4)

- [[findings/jaad-gaze-findings|JAAD Gaze Findings]] — walking+looking → 95.7% cross at decision_point; standing+not-looking → 44.9% (lowest ambiguity pair)
- [[findings/pie-gaze-reversal|PIE Gaze Reversal]] — walking+not-looking → 74.1% cross (higher than walking+looking 56%); gaze sign reverses from JAAD
- [[findings/pie-intention-bimodality|PIE Intention Bimodality]] — mean=0.712, median=0.850; 42.5% of peds at 0.9–1.0; binarization discards calibration
- [[findings/road-ped-tube-statistics|ROAD++ Tube Statistics]] — 9,573 ped tubes, median 58 frames, extreme scale variation; primary detection challenge

---

## Comparisons (3)

- [[comparisons/dataset-comparison|Dataset Comparison]] — side-by-side JAAD vs PIE vs ROAD++ capabilities table from SYNTHESIS Part 3
- [[comparisons/model-comparison|Model Comparison]] — input/output patterns across 11 published models from SYNTHESIS Part 6
- [[comparisons/jaad-vs-pie-gaze|JAAD vs PIE Gaze]] — detailed analysis of the gaze reversal; implications for multi-dataset training
- [[comparisons/openmixer-vs-retinanet|OpenMixer vs 3D-RetinaNet]] — backbone gap analysis: spatial/temporal/semantic trade-offs for MCDM Approach 3
- [[comparisons/bdd-x-vs-covla|BDD-X vs CoVLA]] — Stage 1 pre-training sources: quality vs scale, DSDAG field mapping, joint training strategy

---

## Tools (4)

- [[tools/ros-perception-pipeline|ROS Perception Pipeline]] — multi-node ROS 1 pipeline: YOLOv10 → traffic light + sign classifiers
- [[tools/bdd100k-detection|BDD100K Detection Comparison]] — YOLOPX vs YOLO_BDD vs TwinLiteNet on BDD100K tasks
- [[tools/traffic-light-classification|Traffic Light Classification]] — 7-class EfficientNet-B0 pipeline on LISA dataset
- [[tools/traffic-sign-classification|Traffic Sign Classification]] — 15-class EfficientNet pipeline on Mapillary MTSD v2
