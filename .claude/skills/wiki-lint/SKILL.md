---
name: wiki-lint
description: Run a health check on the research wiki — orphans, broken wikilinks, missing frontmatter, stale pages, page-count drift. Invoke when the user says /wiki-lint, "lint the wiki", "check wiki health", "find broken links", or asks for an audit of the wiki structure.
---

# wiki-lint

A periodic health check. The deterministic checks run in Python — your job is to triage and fix the issues that need a human-grade judgment call.

## Step 1 — Run the lint

```bash
python3 .claude/scripts/wiki.py lint
```

The output is grouped by issue kind:

| Kind | Severity | Who fixes |
|------|----------|-----------|
| `frontmatter-missing` | hard | you (read file, add frontmatter) |
| `field-missing` | hard | you (fill in the field if known, else ask) |
| `type-field-missing` | hard | you (look up the type-required field in the source) |
| `type-invalid` / `status-invalid` | hard | you (fix the value) |
| `wikilink-broken` | hard | you (fix the slug, or create the missing page) |
| `slug-duplicate` | hard | you (rename one — ask user before renaming) |
| `total-pages-mismatch` | trivial | run `wiki.py recount` |
| `orphan` | soft | you (add a one-liner to `index.md`, or mark intentional) |
| `slug-style` | soft | you (only fix if the file is otherwise being touched) |
| `stale` | soft | review the page; either update content + flip status, or report back to user |
| `bud-no-citing` | soft | report to user — a bud whose citers were all deleted is a candidate for deletion |
| `bud-stale` | soft | report to user — a bud > 365 days old that was never promoted may have been deprioritized |

## Step 2 — Triage

Don't fix everything blindly. Read the lint output, then:

1. **Fix trivial things automatically:** run `wiki.py recount` for `total-pages-mismatch`. Add missing `index.md` entries for orphans whose content is obviously useful.
2. **Investigate broken wikilinks individually.** A broken link usually means one of:
   - typo in slug (fix the link)
   - target page was renamed (fix the link to the new slug)
   - target page was never created (decide: create stub or remove link)
3. **For `field-missing` / `type-field-missing`:** check the page's `sources:` and the raw source on disk. If the field is recoverable from the source, fill it. If not, ask the user.
4. **For `stale` pages:** read the page, check the raw source it points to, and update if obviously possible. Otherwise report the stale page back to the user with a one-line summary of what's likely outdated.

## Step 3 — Re-run and commit

```bash
python3 .claude/scripts/wiki.py lint --strict
```

`--strict` exits nonzero if issues remain. Repeat fix-and-re-lint until clean (or until only soft issues remain that need user input).

When committing, group lint fixes into one commit:

```
Wiki lint: fix N broken wikilinks, add M orphans to index, recount
```

## Domain rules (always)

- **Don't rename pages without asking.** Slugs are the wiki's primary keys — renames break wikilinks across the graph.
- **Don't auto-create stub pages** to silence broken-wikilink errors. A broken link is often the right signal that the link should be removed, not that a page should appear.
- **`status: stale`** is a deliberate user signal that a page is known-out-of-date. Don't auto-flip it to `complete`; surface it back to the user with a diff suggestion.
- **`status: bud`** is meta-tracking, not content. Buds are intentionally absent from `index.md` (orphan-check skips them) and excluded from `total_pages`. `bud-no-citing` and `bud-stale` are soft signals — surface them to the user, don't delete or auto-fix.

## What to avoid

- Don't reformat or restyle pages while linting. Touch only what the lint flagged.
- Don't add fields to frontmatter that aren't required just because you saw them on other pages — keep the schema in `CLAUDE.md` authoritative.
- Don't run lint and then dump the entire JSON output verbatim to the user. Summarize: N issues across K pages, here are the categories, here's what I fixed, here's what needs your call.
