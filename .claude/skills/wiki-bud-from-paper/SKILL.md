---
name: wiki-bud-from-paper
description: Harvest "paper buds" from an already-ingested paper. Walks its References / Related Works, creates a stub page for each cited paper in the lab's research themes, and registers the citing paper as a backref. Invoke when the user says /wiki-bud-from-paper, "harvest buds from X", "bud-mine X". Also auto-spawned by /wiki-ingest after a paper page is written.
---

# wiki-bud-from-paper

Your job is to look at an *already-ingested* paper page in this wiki and mint lightweight stub pages ("**buds**") for the papers it cites — so the graph compounds over time. Today's bud is tomorrow's ingest candidate. You do not write full paper pages here; you write stubs that get upgraded later via `wiki.py bud-promote` + `/wiki-ingest`.

The CLI does the heavy lifting: identity matching, slug generation, file writes, dedup of citers. Your job is the unstructured-text work the CLI can't do — reading the PDF, picking which citations matter, and assigning each a one-line role.

## Input

A single slug of a paper page in `papers/`. The user usually passes it explicitly. When auto-spawned by `/wiki-ingest`, the calling agent passes the slug of the just-written paper.

## Step 1 — Read the source

```bash
python3 .claude/scripts/wiki.py cite <slug>
```

That prints the page's path and sources. Read the `sources:` PDF (or markdown, if no PDF) with `Read` — you only need the **References** section, **Related Works** section, and any **result tables** / **method comparisons** in the body.

## Step 2 — Pick citations worth a bud

Walk the body. For each citation:

- **Keep** if the paper is *named in the body* — discussed in Related Works, compared in a results table, or cited inline in Methods.
- **Skip** if the paper is bib-only (appears in References but is not named anywhere in the body).
- **Skip** ubiquitous foundational citations (Transformer, ResNet, BatchNorm, AdamW, Adam) *unless* they tie to a specific lab direction.

Then apply a **lab-theme filter** — keep papers in:

- Autonomous driving perception (detection, segmentation, lane, BEV/3D, traffic light/sign)
- Pedestrian intent prediction (JAAD/PIE/PSI, action prediction, crossing intent)
- JEPA / world models / latent prediction
- VLM / VLA (CLIP, LLaVA, Qwen-VL, V-JEPA, SmolVLM, etc.)
- Neuro-symbolic constraints (t-norms, PiShield, ROAD-R)
- Action detection / open-vocabulary detection / video understanding
- Attention / fusion / FiLM / cross-modal alignment

Drop everything else. Quality over quantity. **Cap the run at ≤ 15 buds.** If a paper has 60+ relevant citations, pick the 15 most central to the citing paper's argument.

## Step 3 — Normalize each citation

For each kept citation, extract:

| Field | Source | Notes |
|---|---|---|
| `--title` | citation text | Cap at 120 chars |
| `--authors` | citation text | `"Lastname et al."` form |
| `--year` | citation text | int |
| `--venue` | citation text | "CVPR 2024", "arXiv preprint", etc. |
| `--arxiv` | citation text if printed | e.g. `2401.00000` — **never invent one** |
| slug | computed | call `wiki.py bud-match --suggest-slug ...` (centralizes the convention) |
| `--context` | your prose | One sentence on the role: "Related Works §5 — JEPA baseline", "Tab. 2 method comparison", etc. |

Suggest a slug:

```bash
python3 .claude/scripts/wiki.py bud-match \
  --suggest-slug --author "Assran et al." --year 2023 --title "I-JEPA"
# prints: assran-2023-jepa  (or similar)
```

## Step 4 — Call `bud-add` per citation

```bash
python3 .claude/scripts/wiki.py bud-add \
  --slug assran-2023-jepa \
  --title "I-JEPA: Joint Embedding Predictive Architecture" \
  --authors "Assran et al." --year 2023 --venue "CVPR 2023" \
  --arxiv "2301.08243" \
  --citing chen-2026-vl-jepa \
  --context "Related Works §5 — original image JEPA; cited as the foundation for VL-JEPA's X-Encoder design"
```

The CLI is idempotent:

- If `papers/<slug>.md` doesn't exist → creates a new bud.
- If it exists as a bud → appends the citer (dedup'd) and updates `updated:`.
- If it exists as a full paper → no-op, prints `already a node`.
- If a *different* slug already matches by arXiv ID → errors out; re-run with that slug.

## Step 5 — Summarize

Print one line for the user: `Harvested N buds from <citing-slug>: M new, K already existed (gained a citer), J no-op (already real papers).`

Don't update `index.md` — buds are intentionally not indexed. Don't append to `log.md` — the harvest is recorded as part of the parent `/wiki-ingest` log entry.

## Hard rules

- **≤ 15 buds per invocation.** Lean.
- **Body-mentioned only.** Bib-only entries skipped unless they appear in a results table.
- **Never invent arXiv IDs.** Leave the field empty if the bib entry doesn't print one.
- **Never bud-add the citing paper itself.** The CLI rejects this, but don't make the call.
- **Don't make up `--context`.** If you can't write a specific one-line role, skip the citation — it's not central enough to deserve a bud.
- **Don't bud-mine other buds.** Only ingest-promoted full paper pages should be bud-mined. (`/wiki-ingest` enforces this by passing the just-written slug.)

## What to avoid

- Don't open Bash to grep the citations manually — read the PDF body and use your existing reading capacity.
- Don't try to reconcile two slugs that differ in spelling. `bud-match --suggest-slug` is the source of truth.
- Don't keep going past 15 — it's better to leave the long tail unminted than to flood the wiki.
- Don't write any prose in this skill output other than the summary line. The bud pages are stubs.
