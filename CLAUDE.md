# Research Wiki — Operating Manual

This file governs how Claude Code maintains the wiki at `/data/repos/wiki/`. Read it at the start of every session.

---

## Project Identity

Master's research in **autonomous driving perception and intent prediction** at NC A&T State University, supervised by **Dr. Moradi**. Research spans:
- Pedestrian intent prediction (JAAD, PIE, ROAD++ datasets)
- Logic-constrained scene reasoning (ROAD_Reason, neuro-symbolic VLMs)
- Efficient intent models (EfficientPIE / SparseTemporalPIE)
- Panoptic driving perception (detection, segmentation, lane detection)
- 3D object detection and BEV fusion

---

## Three-Layer Architecture

| Layer | Location | Who writes it |
|-------|----------|--------------|
| Raw sources | `/data/repos/*` | You (immutable — never modify) |
| Wiki | `/data/repos/wiki/` | LLM (Claude) |
| Schema | `/data/repos/wiki/CLAUDE.md` | Co-evolved by user + LLM |

**Raw sources are immutable.** The LLM reads them but never modifies them.

---

## Raw Source Inventory

### PhD Core

| Repo | Description | Key files |
|------|-------------|-----------|
| `PedestrianIntent++/` | 3-dataset synthesis: JAAD, PIE, ROAD++ | `MEMORY.md`, `SYNTHESIS.md`, `JAAD/JAAD_summary.md`, `PIE/PIE_Dataset_Summary.md`, `ROAD_plusplus/ROAD_plusplus_summary.md`, `ROAD_plusplus/APPROACHES.md` |
| `ROAD_Reason/` | Neuro-symbolic VLM for scene reasoning | `docs/CLAUDE.md`, `docs/APPROACHES.md`, `docs/ROAD_plusplus_summary.md` |
| `EfficientPIE/` | Pedestrian intent prediction models | `README.md`, `docs/RESULTS.md`, `docs/SESSION_NOTES_*.md`, `docs/SPARSE_TEMPORAL_PIE.md` |

### Detection / Segmentation

| Repo | Description | Key files |
|------|-------------|-----------|
| `AutoDrivePerception2026/` | ROS YOLOv10 + classifier pipeline | `PLAN.md` |
| `YOLOPX/` | Anchor-free panoptic perception (BDD100K SOTA) | `README.md`, `eval_readme.md` |
| `YOLO_BDD/` | YOLOv10 on BDD100K | `README.md`, `evaluation_summary_*.md` |
| `eff_light_detection/` | EfficientNet traffic light classifier (7 classes) | `README.md`, `results/experiment_summary.md` |
| `eff_sign_detection/` | EfficientNet traffic sign classifier (15 classes) | `README.md` |
| `TwinLiteNet/` | Lightweight dual-task segmentation + lane | `README.md` |

### Foundations

| Repo | Description | Key files |
|------|-------------|-----------|
| `bevfusion/` | Camera + LiDAR BEV fusion (ICRA 2023) | `README.md` |
| `mmdetection3d/` | OpenMMLab 3D detection toolbox (40+ methods) | `README.md` |

### Other

| Repo | Description | Key files |
|------|-------------|-----------|
| `SLURP/` | Spoken language understanding (methodology parallels) | `CLAUDE_INSTRUCTIONS.md`, `E1_results_summary.md` |

---

## Wiki Structure

```
wiki/
├── CLAUDE.md          ← this file
├── index.md           ← content catalog (update on every ingest)
├── log.md             ← append-only operations log
├── projects/          ← one page per repo (12 pages)
├── datasets/          ← dataset entity pages
├── concepts/          ← domain concept pages
├── methods/           ← model/architecture pages
├── papers/            ← per-paper summaries
├── comparisons/       ← cross-cutting analyses
├── directions/        ← research direction pages
├── findings/          ← key empirical results
└── tools/             ← detection/segmentation pipeline pages
```

---

## Page Conventions

### Naming
- All filenames: lowercase, kebab-case, `.md` extension
- No spaces in filenames
- Wikilinks: `[[page-name]]` or `[[page-name|Display Text]]`

### Frontmatter Schema (required on every wiki page)

```yaml
---
type: project | dataset | concept | method | paper | comparison | direction | finding | tool
title: "Human-readable title"
aliases: []
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: []        # paths relative to /data/repos/
tags: []
status: draft | complete | stale
---
```

**Type-specific additional fields:**

For `type: paper`:
```yaml
authors: "Lastname et al."
year: 2024
venue: "CVPR"
arxiv: "2401.00000"
datasets_used: [jaad, pie, road-plusplus]
```

For `type: dataset`:
```yaml
clips: 346
frames: 82032
primary_units: 686
annotation_format: xml | json
```

For `type: direction`:
```yaml
novelty: true | false
feasibility: workstation | partial | cluster
datasets_required: [road-plusplus]
```

---

## Operations

### Ingest — Adding a New Source

1. Read the source file(s)
2. Discuss key takeaways with user (what to emphasize)
3. Identify which wiki pages need creation or update
4. Create/update pages with correct frontmatter
5. Update `index.md` with new page entries
6. Append an entry to `log.md`:
   ```
   ## [YYYY-MM-DD] INGEST — Source Title
   - Pages created: N  |  Pages updated: M
   - Summary: ...
   ```

### Query — Answering a Question

1. Check `index.md` for relevant pages
2. Read those pages
3. Fall back to raw sources only if wiki coverage is insufficient
4. Synthesize answer with wikilink citations
5. If the answer produces valuable synthesis, file it as a new wiki page
6. Append to `log.md`:
   ```
   ## [YYYY-MM-DD] QUERY — Question
   - Pages consulted: ...
   - New page created: yes/no
   ```

### Lint — Health Check

Check for:
- Pages not appearing in `index.md` (orphans)
- Broken wikilinks (`[[name]]` with no matching file)
- Pages missing frontmatter
- Contradictions between wiki pages and raw source verified numbers
- Entities mentioned across pages without their own page
- Stale status (`status: stale`) needing updates

---

## Domain Rules

1. **Intent vs action:** "Intent" = mental state predicting future crossing; "action" = observed walking/standing. These are distinct. PIE's `intention_prob` is intent; JAAD's `cross` label is outcome.

2. **Always cite dataset when citing statistics.** JAAD and PIE have *opposite* gaze findings — never generalize across datasets without specifying which.

3. **Use verified numbers.** The ROAD++ paper overcounts (claims 198K frames / 54K tubes / 3.9M boxes). Verified from JSON: 153,534 annotated frames / 41,935 tubes / 3.3M boxes (798 annotated + 202 unannotated test videos).

4. **Decision moment semantics differ:** JAAD uses `decision_point`; PIE uses `critical_point`. They are conceptually similar but protocol-different — `critical_point` is defined by the human experiment clip end, `decision_point` is annotator-marked.

5. **Appearance attributes in JAAD are unused.** All 24 binary per-frame appearance attributes in `annotations_appearance/` were unused by every paper in the surveyed literature (2022–2025). This is a research opportunity.

---

## Key Verified Numbers

| Dataset | Clips | Frames | Primary Units | Format |
|---------|-------|--------|---------------|--------|
| JAAD | 346 | 82,032 | 686 behavioral peds | 5-layer XML |
| PIE | 53 | 740,901 | 1,842 ped tracks | 3-layer XML (6 sets) |
| ROAD++ | 798 (+202 test) | 153,534 annotated | 41,935 agent tubes | Single JSON |

**ROAD++ agent distribution:** Car 2.2M boxes | Ped 712K | MedVeh 239K | TL 58K | LarVeh 40K

**PIE intention_prob:** mean=0.712, median=0.850 — bimodal (42.5% of peds at 0.9–1.0)

**JAAD gaze at decision_point:** walking+looking → 95.7% cross; standing+not-looking → 44.9%

**PIE gaze reversal at critical_point:** walking+not-looking → 74.1% cross (higher than walking+looking 56.0%) — looking = hesitation signal in PIE

**SparseTemporalPIE v3:** 0.926 accuracy / 0.947 AUC on PIE test set (9.0M params, 2.50ms)

---

## Updating This File

Update CLAUDE.md when:
- New raw sources are added to `/data/repos/`
- Wiki structure changes (new subdirectories or page types)
- Domain conventions are refined
- Verified numbers are updated from new analysis
