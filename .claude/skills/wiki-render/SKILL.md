---
name: wiki-render
description: Render a synthesis answer or any wiki markdown page to Apple-editorial HTML (minimalist, maroon + purple accents) and open it in Chrome. Invoke when the user says /wiki-render, "render this in HTML", "show this in the browser", or after producing a meaty synthesis they'll want to read in a browser tab. Also good for sharing a single wiki page with someone outside the vault.
---

# wiki-render

Render markdown as a themed HTML document and launch it in Chrome (with graceful fallback to xdg-open).

**Aesthetic: Apple Developer Documentation × Wikipedia (Vector 2022).** A full-page layout, not a narrow column: frosted-glass sticky top bar (`backdrop-filter: blur(20px) saturate(180%)`) with a breadcrumb, a sticky **left TOC rail** auto-built from the page's `h2`/`h3` headings with scroll-spy active highlighting, and a fluid content column (`max-width: 880px`, centered in the remaining space). Near-white `#fbfbfd` background, near-black `#1d1d1f` ink, SF Pro system font stack; no serif body fonts. Headings get hover anchor links; `h2`s carry a hairline underline. Tables are wrapped in a rounded card with horizontal scroll (so the wide architecture grids don't squash) and use tabular numerals. Code blocks rounded 12px, dark fill. YAML frontmatter is stripped before render so the page opens at its `# H1`. Inspired by developer.apple.com/documentation and Wikipedia's Vector 2022 contents sidebar.

**Color** is restrained accent only. Maroon `#7a0e26` for the corner sigil, h4 eyebrow labels, blockquote rule, and link hover state. Purple `#5b2b82` for link rest state with thin underline and 3px offset. Dark mode is automatic via `prefers-color-scheme` (maroon shifts to a lighter `#d97086`, purple to `#b399e0` for readability on black).

Wrap any rendering call around the deterministic CLI at `.claude/scripts/wiki_render.py`. This skill is pure display: no state mutation, nothing written to the wiki tree.

## When to use

- After producing a real synthesis in `/wiki-query` (more than a paragraph, crossing multiple pages) the user will want to **read** it, not just chat it. Render and open.
- When the user asks "show me this in the browser", "render that", or "open it in chrome".
- When the user wants a shareable HTML version of a single wiki page.
- Optionally as a post-step in `/wiki-ingest` for the user to skim what was written — but only if they ask; don't auto-render every ingest.

**Don't use for:** rendering raw paper PDFs (use the PDF tool), live previewing inside Obsidian (Obsidian already does this), or generating slides.

## Inputs

Three forms:

1. **`/wiki-render <path/to/file.md>`** — explicit markdown file.
2. **`/wiki-render <slug>`** — a wiki slug (resolves to `<slug>.md` under the wiki tree by searching `projects/`, `papers/`, `concepts/`, etc.).
3. **`/wiki-render`** *(no arg)* — render the most recent synthesis you produced this session. Write that synthesis to `~/.cache/wiki-render/synthesis-<timestamp>.md` first, then render.

## Step 1 — Locate or write the markdown source

| Input form | What to do |
|---|---|
| Path ending in `.md` | Pass it through. |
| Bare slug (e.g. `papers/chen-2026-vl-jepa`) | Look under `/data/repos/wiki/` for `<slug>.md`. If multiple matches, prefer exact path; if none, run `python3 .claude/scripts/wiki.py search "<slug>" --limit 3` and ask which. |
| No arg / synthesis | Write your synthesis content to a fresh file. Default location: `~/.cache/wiki-render/synthesis-YYYYMMDD-HHMMSS.md`. Make sure the first line is `# <Title>` — that's what the page title and the cream-bar `<h1>` render from. |

## Step 2 — Render and open

```bash
python3 .claude/scripts/wiki_render.py <path.md>
```

The script writes a self-contained HTML file to `/tmp/wiki-render-<timestamp>-<hash>.html` and launches Chrome (falls back to xdg-open if no Chrome family binary is found — looks up `google-chrome`, `chromium`, `brave-browser`, and `~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome`).

Useful flags:

- `--title "Custom Title"` — overrides the first H1
- `--out /path/to/file.html` — fixed output path (default is `/tmp/wiki-render-…html`)
- `--browser /path/to/chrome` — force a specific browser binary
- `--no-open` — write the HTML only, don't launch

## Step 3 — Confirm and stop

Print the path the renderer wrote and the browser that opened it. **Don't** lint, log, or otherwise mutate the wiki — this skill is read-only display.

## What the renderer handles

- GFM markdown (tables, fenced code, task lists) via `marked.js` from CDN.
- **Auto-built left TOC** from `h2`/`h3` headings, generated client-side after parse: each heading is slugged + given an `id`, listed in the sticky rail (h3s indented under their h2), with **scroll-spy** highlighting the section nearest the top as you scroll. Headings also get hover-revealed `#` anchor links.
- **YAML frontmatter stripping** — a leading `---` … `---` block is removed before render so the page starts at its `# H1` rather than dumping `type:`/`tags:` as body text.
- **Wide tables** are wrapped in a rounded `.table-wrap` card that scrolls horizontally — the 7–8-column architecture grids stay readable instead of squashing.
- `[[wikilink]]` and `[[wikilink|Display]]` render as subtle purple inline links with a ◇ glyph and a thin underline. Each points at `obsidian://open?vault=wiki&file=<slug>` so a click in Chrome jumps to the page in Obsidian (if the vault is open).
- ` ```mermaid ` code blocks rendered client-side via `mermaid.js`. Theme variables switch automatically with `prefers-color-scheme`.
- Frosted-glass sticky top bar: ◇ sigil + "Wiki" + a `section / page` breadcrumb on the left, timestamp + source filename on the right.
- Responsive: below 920px the TOC rail collapses behind a ☰ toggle in the bar.
- System fonts only (SF Pro stack with fallbacks — no Google Fonts dependency).

## What the renderer does NOT do

- Doesn't follow wikilinks transitively — only the target page is rendered.
- Doesn't render LaTeX. If the source has `$…$` math, it'll show as literal text. Add MathJax to the template if math support is needed; flag this to the user instead of silently dropping.
- Doesn't bundle CDN assets — Chrome needs internet for `marked.js` and `mermaid.js`. If offline, render still works but markdown won't parse; tell the user.

## Domain rules

- **Maroon + purple is the brand.** Don't restyle without asking.
- **Open path goes to Chrome family first**, then any matching binary in `~/.cache/ms-playwright/chromium-*/`, then xdg-open. Pass `--browser` if the user specifies otherwise.
- If you scaffolded a synthesis file under `~/.cache/wiki-render/`, don't commit it. The cache dir is intentionally outside the repo.
