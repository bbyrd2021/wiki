---
name: wiki-query
description: Answer a research question from the wiki with [[wikilink]] citations to every claim. Invoke when the user says /wiki-query, "ask the wiki", "what does the wiki say about X", or asks a research question that the wiki likely covers (pedestrian intent, ROAD++, frozen DETR, SLURP, JAAD/PIE stats, dataset comparisons, etc.).
---

# wiki-query

The wiki is the source of truth for the user's research. When they ask a question, search it, read the hits, and synthesize an answer with citations — not from memory, not from training data.

## Step 1 — Search

Extract 2–5 keyword terms from the question. Drop stopwords. Run:

```bash
python3 .claude/scripts/wiki.py search "<terms>" --limit 10
```

The search ranks by title × 8 + tags × 4 + body × 1. Each hit prints the slug, title, path, a snippet, and the `sources:` list from frontmatter.

If the top result scores below ~10, the wiki probably doesn't have the answer — say so, and offer to ingest the source.

## Step 2 — Read the top hits

Read the top 3–5 hits in full (or the relevant sections). Don't paraphrase from the snippet alone. Use the `sources:` frontmatter to know what raw file the page is summarizing.

## Step 3 — Synthesize with citations

Every factual claim should carry a wikilink citation. Format:

```markdown
PIE's `intention_prob` is bimodal with mean 0.712 — 42.5% of pedestrians score in [0.9, 1.0]
([[concepts/intention-probability]]).
```

For multiple supporting pages:

```markdown
Gaze behaves oppositely on the two datasets ([[concepts/gaze-and-attention]],
[[comparisons/jaad-vs-pie-gaze]]).
```

At the end of the answer, list the pages consulted as a citation block:

```markdown
**Sources consulted:**
- [[concepts/intention-probability]] — PIE's crowd-sourced intent score (bimodal, mean=0.712)
- [[datasets/pie|PIE]] — 53 videos, 1,842 pedestrian tracks
- [[comparisons/jaad-vs-pie-gaze]] — the gaze reversal between datasets
```

If a claim cites a raw-source statistic, also include the source path from the page's frontmatter so the user can verify.

## Step 4 — File the synthesis (only if warranted)

If your answer produces non-trivial synthesis — bridging two pages, a comparison the user hasn't asked before, a new framing — propose filing it as a new page:

```bash
python3 .claude/scripts/wiki.py scaffold comparison <slug> "<title>"
```

Don't do this for every query. The trigger is: "did I just write something that future-me would want to find by search?" If yes, file it. If it's a one-off recall, don't.

## Step 5 — Log the query

Append to `log.md`:

```markdown
## [YYYY-MM-DD] QUERY — <Question>
- Pages consulted: [[a]], [[b]], [[c]]
- New page filed: yes (`comparisons/foo.md`) | no
```

Skip the log entry for trivial lookups ("what's the PIE frame count?"). Log when the question required real synthesis across pages.

## Domain rules (always)

- **Cite the dataset whenever you cite a statistic.** JAAD ≠ PIE — and the gaze finding is opposite. A claim without dataset attribution is wrong.
- **Trust the wiki's verified numbers over paper claims.** The ROAD++ paper overcounts; `CLAUDE.md`'s table is the authority.
- **When the wiki disagrees with raw sources,** flag the disagreement instead of silently picking one. The user wants to know.
- **When the wiki has no coverage,** say so plainly. Offer to ingest. Don't fabricate.

## What to avoid

- Don't answer from training-data memory when the wiki has a page on the topic. Read the page.
- Don't list every page that matched the keyword. Pick the 3–5 that actually contributed to the answer.
- Don't cite a page by file path. Use `[[wikilink]]` — that's the whole point.
- Don't file a new page for every query. Most queries don't produce filable synthesis.
