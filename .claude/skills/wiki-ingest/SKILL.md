---
name: wiki-ingest
description: Ingest a new raw source (paper PDF, repo README, experiment log, dataset summary) into the research wiki. Invoke when the user says /wiki-ingest, asks to "ingest X", "summarize Y into the wiki", "add this paper to the wiki", or drops a path/URL/file with no further instruction in this wiki repo.
---

# wiki-ingest

The user is adding a new raw source. Your job is to turn it into wiki pages, link it into the graph, and leave the wiki one clean commit better.

Raw sources live in `/data/repos/*` (immutable code repos), `/data/repos/wiki/raw/` (drop folder), and `~/Downloads/` (papers, PDFs, slide decks). The user may pass an explicit path, a URL, or nothing — in which case you propose candidates.

## Step 1 — Locate the source

Run `python3 .claude/scripts/wiki.py status` first. Always. It tells you the dirty state and recent log entries so you don't duplicate work.

If the user gave a path/URL, use it. Otherwise list recent candidates:

```bash
ls -lt ~/Downloads/ 2>/dev/null | head -10
ls -lt raw/ 2>/dev/null | head -10
```

Then ask the user which one to ingest. Do not guess silently.

## Step 2 — Read it

For PDFs, use `Read` directly (pass `pages: "1-N"` for long ones). For code/repos, read the key files listed in `CLAUDE.md`'s raw-source inventory — don't read everything.

## Step 3 — Plan pages

Decide which pages to **create** and **update**. Typical ingest touches 3–8 pages:

- 1 new page for the primary entity (paper / dataset / finding / method / tool)
- 2–5 update edits to related projects, concepts, or comparison pages that gain a cross-reference
- 1 update to `index.md` (one line per new page, under the right category)
- 1 append to `log.md`

Look up existing pages before creating new ones:

```bash
python3 .claude/scripts/wiki.py search "<topic terms>" --limit 8
python3 .claude/scripts/wiki.py list --type paper
```

If a page already exists for the entity, update it; don't create a duplicate.

## Step 4 — Scaffold new pages

**Papers only — bud-match first.** Before scaffolding a `type: paper` page, extract the arXiv ID from the source PDF and pass it to scaffold. The CLI uses it to detect any existing bud whose identity matches this paper. If a bud matches, the scaffold command refuses with a message pointing to `bud-promote`:

```bash
python3 .claude/scripts/wiki.py scaffold paper chen-2026-vl-jepa "VL-JEPA: …" \
  --arxiv "2512.10942" --sources "wiki/raw/VL-JEPA.pdf"
# If a matching bud already exists, scaffold prints:
#   error: matching bud at papers/<bud-slug>.md (status: bud)
#     use: python3 .claude/scripts/wiki.py bud-promote <bud-slug>
```

If the scaffold succeeds, continue with `Edit`. If it refused, call `bud-promote` instead — **the bud's slug supersedes any slug you would have chosen**. This is what preserves inbound wikilinks:

```bash
python3 .claude/scripts/wiki.py bud-promote <bud-slug>
# Status flips bud → draft; citing list and ## Cited by section preserved.
# Now Edit the body as normal.
```

For non-paper types, scaffold as before:

```bash
python3 .claude/scripts/wiki.py scaffold paper bao-2025-openmixer "OpenMixer: …" \
  --sources "ROAD_Reason/papers/openmixer.pdf"
```

Types: `project | dataset | concept | method | paper | comparison | direction | finding | tool`. The scaffolder adds type-specific fields (paper → authors/year/venue, dataset → clips/frames, direction → novelty/feasibility) and sets `status: draft`.

Then `Edit` the body. Conventions:

- Wikilinks: `[[slug]]` or `[[folder/slug|Display Text]]`. Use slug only — folder is optional but improves Obsidian display.
- Cite the dataset whenever you cite a statistic. JAAD and PIE have opposite gaze findings; never generalize across datasets.
- Use verified numbers from `CLAUDE.md`'s table, not paper claims, when they disagree.
- Set `status: complete` (in the frontmatter) once the body is real prose, not a stub.

## Step 4b — Harvest paper buds (papers only)

After the paper body is written and saved with `status: complete`, spawn an `Agent` (foreground, general-purpose) that invokes the `/wiki-bud-from-paper <new-slug>` skill. It re-reads the source PDF, walks Related Works / body discussion / result tables, and calls `wiki.py bud-add` for each cited paper in the lab's research themes. The CLI handles dedup, slug matching, and "already a node" no-ops.

```text
Agent(
  description: "Harvest buds from <slug>",
  subagent_type: "general-purpose",
  prompt: "Invoke the /wiki-bud-from-paper skill for slug <new-slug>. Read its source PDF, harvest ≤15 paper buds from Related Works / body discussion / result tables. Report a one-line summary."
)
```

Skip this step for non-paper ingests (datasets, findings, tools, methods) — they don't have citation lists to mine.

If the user explicitly says "no buds this time," skip; otherwise spawn it. Note the bud count in the log entry (Step 6).

## Step 5 — Update the index

Add a one-liner under the appropriate category in `index.md`. Match the existing voice — concise hook, not a paragraph. Example:

```
- [[papers/bao-2025-openmixer|Bao 2025 — OpenMixer]] — VLM-localizability for open-vocab action detection; relevant to Exp2 / VLM gap (WACV 2025)
```

## Step 6 — Append to log.md

Append a single entry. Format (keep it consistent with prior entries):

```markdown
## [YYYY-MM-DD] INGEST — <Source Title>
- Pages created: N (`papers/foo.md`, …)
- Pages updated: M (`index.md`, `projects/bar.md`, …)
- Source: <path or URL>
- Summary: 1–2 sentences on the *why* — what this source unlocks for the research, not just what it is.
```

## Step 7 — Validate and commit

```bash
python3 .claude/scripts/wiki.py recount       # fix total_pages + updated date
python3 .claude/scripts/wiki.py lint          # surface broken links / missing fields
```

Fix any issues your ingest introduced (broken wikilinks, missing frontmatter fields). Pre-existing issues — leave them; they belong to `/wiki-lint`.

Commit only when the user asks, or when you've made a self-contained ingest and the user said to commit automatically. Use a message like:

```
Ingest <source>: +N pages, +M updates
```

## Domain rules (always)

- **Intent ≠ action.** PIE's `intention_prob` is intent; JAAD's `cross` is outcome.
- **Always cite the dataset** alongside any statistic.
- **Use verified numbers.** ROAD++ paper overcounts; trust the JSON-verified table in `CLAUDE.md`.
- **JAAD appearance attributes are unused** by all surveyed papers — flag this whenever ingesting new JAAD-related work.
- **Extract the arXiv ID before scaffolding a paper.** Pass it to `scaffold paper --arxiv ...`. The CLI uses it to detect existing buds — if you skip this step, you risk creating a duplicate of a bud someone else's ingest already seeded.

## What to avoid

- Don't create pages for sources you only skimmed. If you can't summarize the key contribution in one sentence, read more before writing.
- Don't add concept pages for entities that have a single mention. Wait for a second mention before creating a page — Karpathy's rule.
- Don't duplicate. Search first.
- Don't fix unrelated lint issues during ingest. That's the lint skill's job. Keep the commit scoped.
