# Research Wiki

A structured, append-only research notebook for a master's program in **autonomous driving perception and pedestrian intent prediction** at NC A&T State University (Dr. Moradi's lab). Spans pedestrian intent (JAAD, PIE, ROAD++), neuro-symbolic VLM reasoning, panoptic driving perception, traffic-light/sign classification, BEV fusion, and JEPA-style world models.

The repo is more than notes — it's a graph-shaped knowledge base maintained jointly by a human researcher and an AI assistant, with a thin Python CLI (`wiki.py`) and a small set of agent skills (`/wiki-ingest`, `/wiki-query`, `/wiki-lint`, `/wiki-bud-from-paper`) that enforce structural invariants. Every page is typed, frontmatter-validated, and wikilinked into the graph. Every change is logged. Stats and growth events live in `log.md`; the catalog lives in `index.md`.

---

## At a glance

| Stat | Value |
|---|---|
| Pages | **110** (32 papers • 14 findings • 13 concepts • 12 methods • 12 projects • 10 directions • 7 comparisons • 6 datasets • 4 tools) |
| Wiki languages | Markdown + YAML frontmatter |
| Toolchain | Python 3 stdlib only (`.claude/scripts/wiki.py`), no install step |
| Validators | Type schema, frontmatter, wikilink graph, orphan/stale/count drift |
| Latest growth | See `log.md` — most recent entries first |

---

## Why a wiki

Research builds on itself. A PDF you skimmed last semester needs to be reachable from the experiment you're running today; an architectural decision made on Tuesday needs to surface when a related ablation fails on Friday. Folder-and-PDF hoarding doesn't compose. A flat Notion page becomes a search problem within a month.

The structure here treats research as a **typed graph**:

- **Nodes** are pages: a *paper*, a *dataset*, a *method*, a *concept*, an *experimental finding*, a *direction*, a *project repo*, a *tool*, or a *comparison*.
- **Edges** are wikilinks: `[[chen-2026-vl-jepa]]` from any page resolves to the VL-JEPA paper page, and a lint step verifies every link points at a real page.
- **Metadata** (frontmatter) lets the CLI search, count, and dedupe; lets a future AI agent reason over the catalog; and prevents schema drift.

The friction is intentional: the cost of writing a new page makes you decide whether the work deserves a page at all. The reward is a graph that gets more useful with every ingest.

---

## Three-layer architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Raw sources (immutable)                                 │
│  /data/repos/*  (code repos: PedestrianIntent++, ROAD_Reason,      │
│                  EfficientPIE, YOLOPX, YOLO_BDD, AutoDrive…)        │
│  /data/repos/wiki/raw/  (drop folder for PDFs, exports)             │
│  ~/Downloads/  (papers, slide decks, dataset metadata)              │
│                                                                    │
│  Read by the LLM. Never modified.                                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │  /wiki-ingest reads source, writes pages
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 2 — The wiki (this repo)                                    │
│  projects/  datasets/  concepts/  methods/  papers/                │
│  comparisons/  directions/  findings/  tools/                      │
│  index.md  log.md                                                  │
│                                                                    │
│  Written by the AI assistant under human review.                   │
│  Validated by .claude/scripts/wiki.py lint.                        │
└────────────────────────────────────────────────────────────────────┘
                              │
                              │  CLAUDE.md, .claude/skills/* govern behavior
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 3 — The schema (this repo, but special)                     │
│  CLAUDE.md  — operating manual for the AI assistant                │
│  .claude/scripts/wiki.py  — deterministic CLI                      │
│  .claude/skills/*/SKILL.md  — agent instructions                   │
│                                                                    │
│  Co-evolved by user + AI as conventions are refined.               │
└────────────────────────────────────────────────────────────────────┘
```

**Why three layers?** Raw sources have to stay reproducible; you can't mutate someone else's published paper. The wiki is the interpretation layer — what *I* learned from those sources, how it connects to what I already knew. The schema lets the system police itself; an AI assistant that strays from conventions gets caught by the next lint pass.

---

## Page types

Every page declares `type:` in frontmatter. The type controls required fields, the directory it lives in, and how the lint validates it.

| Type | Folder | Count | Examples | Meaning |
|---|---|---:|---|---|
| `paper` | `papers/` | 32 | [`chen-2026-vl-jepa`](papers/chen-2026-vl-jepa.md), [`pearl-2009-causality`](papers/pearl-2009-causality.md), [`cheng-2025-mcam`](papers/cheng-2025-mcam.md) | A single paper read in depth; carries `authors`, `year`, `venue`, `arxiv` |
| `dataset` | `datasets/` | 6 | [`jaad`](datasets/jaad.md), [`pie`](datasets/pie.md), [`road-plusplus`](datasets/road-plusplus.md), [`bdd100k`](datasets/bdd100k.md) | A benchmark / corpus; carries `clips`, `frames`, `primary_units`, `annotation_format` |
| `method` | `methods/` | 12 | [`sparse-temporal-pie`](methods/sparse-temporal-pie.md), [`wavlm-hier`](methods/wavlm-hier.md), [`multimodal-causal-driving`](methods/multimodal-causal-driving.md) | A model / architecture; explainer pages also live here |
| `project` | `projects/` | 12 | [`road-reason`](projects/road-reason.md), [`efficient-pie`](projects/efficient-pie.md), [`slurp`](projects/slurp.md) | One page per raw-source code repo |
| `concept` | `concepts/` | 13 | [`encoder-collapse`](concepts/encoder-collapse.md), [`vlm-localization-gap`](concepts/vlm-localization-gap.md), [`crossing-intent`](concepts/crossing-intent.md) | A reusable domain idea cited from multiple pages |
| `direction` | `directions/` | 10 | [`jepa-intent-head`](directions/jepa-intent-head.md), [`constrained-vlm-reasoning`](directions/constrained-vlm-reasoning.md) | A research direction; carries `novelty`, `feasibility`, `datasets_required` |
| `finding` | `findings/` | 14 | [`slurp-wavlm-hier-results`](findings/slurp-wavlm-hier-results.md), [`exp2c-frozen-detr`](findings/exp2c-frozen-detr.md), [`yolov10-bdd13-extension`](findings/yolov10-bdd13-extension.md) | A specific empirical result from a run |
| `comparison` | `comparisons/` | 7 | [`slurp-audio-vs-text-oracle`](comparisons/slurp-audio-vs-text-oracle.md), [`fusion-for-detection-lit-review`](comparisons/fusion-for-detection-lit-review.md) | Cross-cutting head-to-head writeups |
| `tool` | `tools/` | 4 | [`ros-perception-pipeline`](tools/ros-perception-pipeline.md), [`traffic-light-classification`](tools/traffic-light-classification.md) | A pipeline / utility / multi-model wrapper |

Plus two singletons: [`index.md`](index.md) (catalog) and [`log.md`](log.md) (append-only operations log).

### Paper buds — the recursive growth mechanism

A **bud** is a stub `type: paper` page with `status: bud`. Buds are seeded automatically by `/wiki-ingest` when it harvests the *cited papers* of an ingested paper. They carry a best-effort title/authors/year/arxiv and a `citing:` list of paper slugs that reference them, plus a `## Cited by` body section.

Today's bud is tomorrow's ingest candidate. When a future paper ingest matches an existing bud (by arXiv ID, then by surname+year, then by fuzzy title), `wiki.py scaffold paper` refuses to overwrite and points to `wiki.py bud-promote <slug>`. Promotion flips `status: bud` → `status: draft`, preserves the `citing:` history and the `created:` seed date, and keeps the slug — so every inbound `[[wikilink]]` written when the page was a bud still resolves after promotion.

Buds are intentionally absent from `index.md`. They're browseable via:

```bash
python3 .claude/scripts/wiki.py bud-list                  # all buds
python3 .claude/scripts/wiki.py bud-list --min-citers 2   # buds with >=2 citers
python3 .claude/scripts/wiki.py bud-list --citing X       # buds cited by paper X
```

See [CLAUDE.md → Paper buds](CLAUDE.md) for the full schema, identity-match precedence, and promotion semantics. See also `.claude/skills/wiki-bud-from-paper/SKILL.md` for the harvest skill's rules (≤15 buds per ingest, body-mention filter, never-invent-arxiv).

---

## Operations

Four agent skills sit on top of the CLI. Each is a thin instruction layer; the deterministic work happens in `wiki.py`.

| Slash command | Skill file | What it does |
|---|---|---|
| `/wiki-ingest` | [`.claude/skills/wiki-ingest/SKILL.md`](.claude/skills/wiki-ingest/SKILL.md) | Read a raw source, plan pages, scaffold, edit, update index + log, lint. For papers: extracts arxiv to enable bud-match, then auto-spawns bud harvest. |
| `/wiki-query` | [`.claude/skills/wiki-query/SKILL.md`](.claude/skills/wiki-query/SKILL.md) | Search the wiki, read top hits, synthesize an answer where every claim carries a `[[wikilink]]` citation. |
| `/wiki-lint` | [`.claude/skills/wiki-lint/SKILL.md`](.claude/skills/wiki-lint/SKILL.md) | Run the health check; triage + fix orphans, broken wikilinks, missing frontmatter; surface soft signals like `stale` / `bud-stale` / `bud-no-citing`. |
| `/wiki-bud-from-paper` | [`.claude/skills/wiki-bud-from-paper/SKILL.md`](.claude/skills/wiki-bud-from-paper/SKILL.md) | Walk an already-ingested paper's References / Related Works / result tables and mint ≤15 bud pages for cited papers in the lab's research themes. |

### CLI (`wiki.py`)

Reusable from the shell or inside a skill. Stdlib only. Auto-detects the wiki root.

```bash
python3 .claude/scripts/wiki.py status                 # counts + dirty + recent log
python3 .claude/scripts/wiki.py list [--type T] [--status S]
python3 .claude/scripts/wiki.py search "<query>" [--limit N]
python3 .claude/scripts/wiki.py scaffold <type> <slug> "<title>" [--sources …] [--arxiv ID]
python3 .claude/scripts/wiki.py lint [--strict] [--json]
python3 .claude/scripts/wiki.py cite <slug>            # print [[wikilink|title]]
python3 .claude/scripts/wiki.py index-check            # orphan-only check
python3 .claude/scripts/wiki.py recount                # fix index.md total_pages

# Paper buds
python3 .claude/scripts/wiki.py bud-add --slug X --title T --citing CITER --context "…" \
                                        [--authors A] [--year YYYY] [--venue V] [--arxiv ID]
python3 .claude/scripts/wiki.py bud-list [--citing <slug>] [--min-citers N]
python3 .claude/scripts/wiki.py bud-promote <slug>     # bud → draft, preserve citing/created
python3 .claude/scripts/wiki.py bud-match --arxiv ID | --author A --year YYYY --title T
                                  [--suggest-slug]
```

The lint catches: missing frontmatter, missing required fields per type, unknown types/statuses, broken `[[wikilinks]]`, duplicate slugs across folders, slug-style violations, total-pages drift, orphan pages (slugs not mentioned in `index.md`), `stale` pages, `bud-no-citing`, and `bud-stale`.

---

## Conventions

### Frontmatter schema

Every page begins with:

```yaml
---
type: project | dataset | concept | method | paper | comparison | direction | finding | tool
title: "Human-readable title"
aliases: []
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: []          # paths relative to /data/repos/
tags: []
status: draft | complete | stale | bud
---
```

Type-specific additions for `paper`:

```yaml
authors: "Lastname et al."
year: 2024
venue: "CVPR"
arxiv: "2401.00000"
datasets_used: [jaad, pie, road-plusplus]
```

For `dataset`: `clips`, `frames`, `primary_units`, `annotation_format`. For `direction`: `novelty`, `feasibility`, `datasets_required`. For `paper` buds: also `citing: [slug, slug, …]`.

### Naming and wikilinks

- All filenames are lowercase, kebab-case, `.md`.
- Wikilinks: `[[slug]]` or `[[folder/slug|Display Text]]`. The folder prefix is optional but improves Obsidian's rendering.
- Paper slugs follow `<surname>-<year>-<keyword>`: `chen-2026-vl-jepa`, `pearl-2009-causality`, `bao-2025-openmixer`. The CLI's `bud-match --suggest-slug` generates this form centrally.

---

## Domain quick reference

Some facts that come up often and that the wiki treats as **load-bearing invariants**:

### Verified dataset counts

| Dataset | Clips | Frames | Primary units | Format |
|---|---|---|---|---|
| JAAD | 346 | 82,032 | 686 behavioral peds | 5-layer XML |
| PIE | 53 | 740,901 | 1,842 ped tracks | 3-layer XML (6 sets) |
| ROAD++ | 798 (+202 test) | 153,534 annotated | 41,935 agent tubes | Single JSON |
| BDD100K | 100,000 | 100,000 | 100K images | JSON |

Note: the **ROAD++ paper overcounts** — it claims 198K frames / 54K tubes / 3.9M boxes. The wiki uses JSON-verified numbers: 153,534 annotated frames / 41,935 tubes / 3.3M boxes (798 annotated + 202 unannotated test videos). See [`datasets/road-plusplus`](datasets/road-plusplus.md).

### Intent ≠ action

PIE's `intention_prob` is **intent** (a crowd-sourced score of "will this pedestrian cross?"). JAAD's `cross` is **outcome** (did they cross or not in the recorded clip). These are not the same and never generalize across datasets without specifying which.

### Gaze findings flip between JAAD and PIE

- **JAAD `decision_point`:** walking + looking → 95.7% cross.
- **PIE `critical_point`:** walking + not-looking → 74.1% cross (higher than walking + looking 56.0%).

In PIE, gaze is a hesitation signal; in JAAD, it's a commitment signal. Never quote a "gaze finding" without naming the dataset.

### Other gotchas

- **JAAD's 24 per-frame appearance attributes** in `annotations_appearance/` are *unused by every paper in the 2022–2025 surveyed literature*. Research opportunity.
- **`decision_point` (JAAD) vs `critical_point` (PIE)** are conceptually similar but protocol-different — `critical_point` is defined by the human-experiment clip end, `decision_point` is annotator-marked.

See [CLAUDE.md → Key Verified Numbers](CLAUDE.md) for the full table.

---

## Reading paths

**If you want to understand the research program:**
1. [`projects/road-reason`](projects/road-reason.md) — the primary thesis project (neuro-symbolic VLM for scene reasoning)
2. [`projects/pedestrian-intent`](projects/pedestrian-intent.md) — the dataset synthesis (JAAD / PIE / ROAD++)
3. [`directions/`](directions/) — the open research directions, with novelty + feasibility flags
4. [`comparisons/`](comparisons/) — cross-cutting analyses tying methods/papers/datasets together

**If you want to understand the methodology / tooling:**
1. [`CLAUDE.md`](CLAUDE.md) — the operating manual (governs how the AI assistant maintains the wiki)
2. [`.claude/scripts/wiki.py`](.claude/scripts/wiki.py) — the deterministic CLI
3. [`.claude/skills/*/SKILL.md`](.claude/skills/) — the four agent skills (ingest, query, lint, bud-from-paper)
4. [`log.md`](log.md) — every ingest/audit/update since the wiki was bootstrapped

**If you want to evaluate a specific paper or method:**
- Start at [`index.md`](index.md) — the catalog has a one-line hook per page.
- Or grep: `python3 .claude/scripts/wiki.py search "your query"`.

---

## Local setup

```bash
git clone https://github.com/bbyrd2021/wiki.git
cd wiki

# That's it. There's no install. The CLI is stdlib-only Python.
python3 .claude/scripts/wiki.py status
```

Optional: open in [Obsidian](https://obsidian.md/) for graph-view rendering. The wiki's `[[wikilink]]` syntax is Obsidian-native; the schema and lint don't require Obsidian to function.

If you're an AI assistant invoked on this repo, **read [CLAUDE.md](CLAUDE.md) first** — it carries the operating manual, the page conventions, the verified numbers, and the rules the lint enforces.

---

## License & contact

Personal research notebook; the prose and structure are mine. Cited papers, datasets, and code remain the property of their respective authors / institutions.

- **Author:** Brandon Byrd (`brandon.byrd011@gmail.com`)
- **Advisor:** Dr. Moradi, NC A&T State University
- **Source:** https://github.com/bbyrd2021/wiki

If you find a structural bug (broken wikilink, orphan page, stale claim), open an issue or just fork it; the lint will flag the same thing for you.
