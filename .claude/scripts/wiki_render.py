#!/usr/bin/env python3
"""
wiki_render.py — render a markdown synthesis (or any wiki page) as themed
HTML and open it in Chrome (falls back to system default browser).

Theme: Apple-docs × Wikipedia. Full-page layout with a sticky auto-built
table-of-contents sidebar (scroll-spy), wide content column, heading anchors,
maroon + purple accents. Dark mode automatic.

Usage:
  python3 wiki_render.py <path-to.md>
  python3 wiki_render.py -                 # read markdown from stdin
  python3 wiki_render.py <path.md> --title "Custom Title"
  python3 wiki_render.py <path.md> --out /tmp/my.html
  python3 wiki_render.py <path.md> --no-open

Output: writes a self-contained HTML file (Markdown rendering + Mermaid happen
client-side via CDN) and launches it in the browser via `start_new_session`
so this script exits immediately.

Stdlib-only. No Python dependencies.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


CHROME_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium-browser",
    "chromium",
    "brave-browser",
    "chrome",
)

# Non-PATH locations to probe if no chrome family binary is on PATH.
EXTRA_CHROME_PATHS = (
    "~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome",
    "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome",
    "/opt/google/chrome/google-chrome",
    "/opt/google/chrome/chrome",
    "/snap/bin/chromium",
)


HTML_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
<style>
/*
 * wiki-render — Apple Developer Documentation × Wikipedia (Vector 2022).
 * Full-page three-region layout: frosted top bar, sticky left TOC rail
 * (auto-built from headings, scroll-spy active state), fluid content column.
 * Background near-white, ink near-black, maroon + purple used surgically.
 */
:root {
  --bg:            #fbfbfd;
  --bg-elev:       #ffffff;
  --rail-bg:       #f5f5f7;
  --ink:           #1d1d1f;
  --ink-sub:       #6e6e73;
  --ink-dim:       #86868b;
  --hairline:      rgba(0, 0, 0, 0.08);
  --hairline-firm: rgba(0, 0, 0, 0.14);
  --hover-tint:    rgba(0, 0, 0, 0.025);
  --code-bg:       #f5f5f7;
  --code-bg-dark:  #1d1d1f;
  --code-ink-dark: #f5f5f7;
  --maroon:        #7a0e26;
  --maroon-soft:   rgba(122, 14, 38, 0.55);
  --purple:        #5b2b82;
  --purple-soft:   rgba(91, 43, 130, 0.38);
  --purple-wash:   rgba(91, 43, 130, 0.08);
  --selection:     rgba(91, 43, 130, 0.16);
  --rail-w:        300px;
  --content-max:   880px;
  --bar-h:         52px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg:            #000000;
    --bg-elev:       #1d1d1f;
    --rail-bg:       #0a0a0b;
    --ink:           #f5f5f7;
    --ink-sub:       #a1a1a6;
    --ink-dim:       #6e6e73;
    --hairline:      rgba(255, 255, 255, 0.10);
    --hairline-firm: rgba(255, 255, 255, 0.18);
    --hover-tint:    rgba(255, 255, 255, 0.04);
    --code-bg:       #1d1d1f;
    --code-bg-dark:  #161618;
    --code-ink-dark: #f5f5f7;
    --maroon:        #d97086;
    --maroon-soft:   rgba(217, 112, 134, 0.55);
    --purple:        #b399e0;
    --purple-soft:   rgba(179, 153, 224, 0.40);
    --purple-wash:   rgba(179, 153, 224, 0.12);
    --selection:     rgba(179, 153, 224, 0.28);
  }
}

* { box-sizing: border-box; }

html { background: var(--bg); scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
               "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 18px;
  line-height: 1.6;
  letter-spacing: -0.003em;
  font-weight: 400;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

::selection { background: var(--selection); color: var(--ink); }

/* ── Sticky frosted top bar ────────────────────────────────────────── */
header.bar {
  position: sticky;
  top: 0;
  z-index: 50;
  height: var(--bar-h);
  background: color-mix(in srgb, var(--bg) 72%, transparent);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--hairline);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  font-size: 13px;
}

@supports not (background: color-mix(in srgb, white 50%, transparent)) {
  header.bar { background: rgba(251, 251, 253, 0.82); }
}

header .brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--ink);
  font-weight: 600;
  letter-spacing: -0.005em;
}
header .brand .sigil { color: var(--maroon); font-size: 15px; line-height: 1; }
header .brand .crumb { color: var(--ink-dim); font-weight: 400; }
header .meta {
  font-size: 12px; color: var(--ink-sub);
  font-feature-settings: "tnum" 1;
}

/* ── Page grid: [ TOC rail | content ] ─────────────────────────────── */
.layout {
  display: grid;
  grid-template-columns: var(--rail-w) minmax(0, 1fr);
  align-items: start;
}

/* Left rail — sticky, scrollable, Wikipedia/Apple-docs contents nav */
nav.toc {
  position: sticky;
  top: var(--bar-h);
  height: calc(100vh - var(--bar-h));
  overflow-y: auto;
  padding: 32px 20px 60px 28px;
  background: var(--rail-bg);
  border-right: 1px solid var(--hairline);
  font-size: 13.5px;
  line-height: 1.4;
}
nav.toc .toc-label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--ink-dim); font-weight: 600; margin: 0 0 14px 10px;
}
nav.toc ul { list-style: none; margin: 0; padding: 0; }
nav.toc li { margin: 0; }
nav.toc a {
  display: block;
  padding: 5px 10px;
  margin: 1px 0;
  color: var(--ink-sub);
  text-decoration: none;
  border-radius: 6px;
  border-left: 2px solid transparent;
  transition: color 120ms ease, background 120ms ease, border-color 120ms ease;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
nav.toc a.lvl-3 { padding-left: 24px; font-size: 12.8px; color: var(--ink-dim); }
nav.toc a:hover { color: var(--ink); background: var(--hover-tint); }
nav.toc a.active {
  color: var(--maroon);
  background: var(--purple-wash);
  border-left-color: var(--maroon);
  font-weight: 600;
}

/* Content column — fluid, centered, generous */
main {
  min-width: 0;
  padding: 56px clamp(28px, 5vw, 72px) 160px;
}
article { max-width: var(--content-max); margin: 0 auto; }

/* Headings + hover anchor (Wikipedia ¶ / Apple-docs link) */
article :is(h1,h2,h3) { scroll-margin-top: calc(var(--bar-h) + 16px); position: relative; }
article .anchor {
  position: absolute; left: -0.85em; top: 0;
  opacity: 0; text-decoration: none; color: var(--maroon);
  font-weight: 400; transition: opacity 120ms ease;
}
article :is(h1,h2,h3):hover .anchor { opacity: 0.55; }
article .anchor:hover { opacity: 1 !important; }

article h1 {
  font-size: clamp(2.2rem, 4.4vw, 3.1rem);
  line-height: 1.07; letter-spacing: -0.025em; font-weight: 700;
  color: var(--ink); margin: 0 0 0.5em;
}
article h2 {
  font-size: clamp(1.5rem, 2.6vw, 1.95rem);
  line-height: 1.16; letter-spacing: -0.018em; font-weight: 700;
  color: var(--ink); margin: 2.2em 0 0.5em;
  padding-bottom: 0.32em; border-bottom: 1px solid var(--hairline);
}
article h3 {
  font-size: 1.28rem; line-height: 1.3; letter-spacing: -0.012em;
  font-weight: 600; color: var(--ink); margin: 1.8em 0 0.35em;
}
article h4 {
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.085em;
  color: var(--maroon); font-weight: 600; margin: 1.7em 0 0.25em;
}

article p { margin: 1em 0; color: var(--ink); }
article strong { font-weight: 600; color: var(--ink); }
article em     { font-style: italic; color: var(--ink); }

/* Links */
article a {
  color: var(--purple);
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
  text-decoration-color: var(--purple-soft);
  transition: color 150ms ease, text-decoration-color 150ms ease;
}
article a:hover { color: var(--maroon); text-decoration-color: var(--maroon); }

/* Wikilinks — quiet diamond glyph */
article a.wikilink {
  color: var(--purple); text-decoration: none;
  border-bottom: 1px solid var(--purple-soft);
  padding-bottom: 1px; white-space: nowrap;
  transition: color 150ms ease, border-color 150ms ease;
}
article a.wikilink:hover { color: var(--maroon); border-bottom-color: var(--maroon); }
article a.wikilink::before {
  content: "◇\00a0"; color: var(--maroon); opacity: 0.6; font-size: 0.82em;
}

/* Code */
article :not(pre) > code {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 0.84em; font-feature-settings: "calt" 0;
  background: var(--code-bg); color: var(--ink);
  padding: 1.5px 6px; border-radius: 5px;
}
article pre {
  background: var(--code-bg-dark); color: var(--code-ink-dark);
  padding: 20px 24px; border-radius: 12px; overflow-x: auto;
  line-height: 1.55; margin: 1.6em 0; font-size: 0.86em;
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  -webkit-overflow-scrolling: touch;
}
article pre code { background: transparent; color: inherit; padding: 0; font-size: inherit; }

/* Blockquote */
article blockquote {
  margin: 1.6em 0; padding: 0 0 0 1.4em;
  color: var(--ink-sub); font-style: italic;
  border-left: 2px solid var(--maroon); font-size: 1.05em;
}
article blockquote p { margin: 0.5em 0; }

/* Tables — wrapped for horizontal scroll on wide grids */
.table-wrap {
  overflow-x: auto; margin: 1.8em 0;
  border: 1px solid var(--hairline); border-radius: 12px;
  background: var(--bg-elev);
  -webkit-overflow-scrolling: touch;
}
.table-wrap table { margin: 0; }
article table {
  width: 100%; border-collapse: collapse;
  font-size: 0.9em; font-feature-settings: "tnum" 1;
}
article thead tr { background: var(--code-bg); }
article th {
  text-align: left; font-weight: 600; color: var(--ink-sub);
  padding: 13px 16px; font-size: 0.8em;
  text-transform: uppercase; letter-spacing: 0.04em;
  border-bottom: 1px solid var(--hairline-firm); white-space: nowrap;
}
article td {
  padding: 13px 16px; border-bottom: 1px solid var(--hairline);
  vertical-align: top; color: var(--ink);
}
article tbody tr:last-child td { border-bottom: none; }
article tbody tr:hover { background: var(--hover-tint); }

/* Rule */
article hr { border: none; height: 1px; background: var(--hairline); margin: 3em 0; }

article ul, article ol { padding-left: 1.4em; margin: 1em 0; }
article li { margin: 0.4em 0; }
article li::marker { color: var(--ink-dim); }

/* Mermaid */
article .mermaid {
  text-align: center; margin: 2em 0; padding: 28px 20px;
  background: var(--bg-elev); border: 1px solid var(--hairline); border-radius: 14px;
}
article .mermaid svg { max-width: 100%; height: auto; }

/* Footer */
footer { text-align: center; padding: 40px 0 80px; color: var(--ink-dim); }
footer .sigil {
  display: block; color: var(--maroon);
  letter-spacing: 0.5em; font-size: 13px; margin-bottom: 10px;
}
footer .credit { font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; }

/* Mobile — collapse the rail */
.toc-toggle { display: none; }
@media (max-width: 920px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc {
    position: fixed; top: var(--bar-h); left: 0; bottom: 0;
    width: min(82vw, 320px); z-index: 40;
    transform: translateX(-101%); transition: transform 220ms ease;
    box-shadow: 2px 0 24px rgba(0,0,0,0.18);
  }
  body.toc-open nav.toc { transform: translateX(0); }
  .toc-toggle {
    display: inline-flex; align-items: center; gap: 7px;
    background: none; border: none; cursor: pointer;
    color: var(--ink); font: inherit; font-size: 13px; padding: 0;
  }
  .toc-toggle .bars { color: var(--maroon); font-size: 16px; }
}
</style>
</head>
<body>

<header class="bar">
  <span class="brand">
    <button class="toc-toggle" aria-label="Toggle contents"><span class="bars">☰</span></button>
    <span class="sigil">◇</span>Wiki<span class="crumb">__CRUMB__</span>
  </span>
  <span class="meta">__META__</span>
</header>

<div class="layout">
  <nav class="toc" aria-label="Contents">
    <p class="toc-label">On this page</p>
    <ul id="toc-list"></ul>
  </nav>
  <main>
    <article id="content"></article>
    <footer>
      <span class="sigil">◇</span>
      <span class="credit">Wiki render</span>
    </footer>
  </main>
</div>

<script id="md-payload" type="application/octet-stream">__PAYLOAD__</script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

  // Mermaid theme variables matched to the page color scheme at load time.
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const m = dark ? {
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, sans-serif',
    primaryColor: '#1d1d1f', primaryTextColor: '#f5f5f7', primaryBorderColor: '#d97086',
    lineColor: '#a1a1a6', secondaryColor: '#000000', tertiaryColor: '#1d1d1f',
    mainBkg: '#1d1d1f', background: '#000000', nodeTextColor: '#f5f5f7',
    edgeLabelBackground: '#000000',
  } : {
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", Segoe UI, sans-serif',
    primaryColor: '#ffffff', primaryTextColor: '#1d1d1f', primaryBorderColor: '#1d1d1f',
    lineColor: '#86868b', secondaryColor: '#fbfbfd', tertiaryColor: '#ffffff',
    mainBkg: '#ffffff', background: '#fbfbfd', nodeTextColor: '#1d1d1f',
    edgeLabelBackground: '#ffffff',
  };
  mermaid.initialize({
    startOnLoad: false, theme: 'base', themeVariables: m,
    flowchart: { curve: 'basis', useMaxWidth: true },
  });

  // Decode base64 → utf8 markdown
  const b64 = document.getElementById('md-payload').textContent.trim();
  const bin = atob(b64);
  const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
  const raw = new TextDecoder('utf-8').decode(bytes);

  // Wikilinks: [[slug]] / [[slug|display]] → styled <a> to obsidian://
  const wikified = raw.replace(
    /\[\[([^\]|\n]+?)(?:\|([^\]\n]+?))?\]\]/g,
    (_, slug, display) => {
      const file = slug.trim();
      const text = (display || file).trim();
      const url = 'obsidian://open?vault=wiki&file=' + encodeURIComponent(file);
      return `<a class="wikilink" href="${url}">${text}</a>`;
    }
  );

  // Pull ```mermaid blocks out before markdown parse; restore as <div> after.
  const mermaidExtracted = [];
  const preExtracted = wikified.replace(
    /```mermaid\n([\s\S]*?)```/g,
    (_, body) => {
      const idx = mermaidExtracted.length;
      mermaidExtracted.push(body);
      return `\n\n<div class="mermaid" data-mermaid-idx="${idx}"></div>\n\n`;
    }
  );

  marked.setOptions({ gfm: true, breaks: false });
  document.getElementById('content').innerHTML = marked.parse(preExtracted);

  // Render mermaid bodies
  document.querySelectorAll('.mermaid[data-mermaid-idx]').forEach(async (el) => {
    const idx = +el.dataset.mermaidIdx;
    try {
      const { svg } = await mermaid.render('mermaid-svg-' + idx, mermaidExtracted[idx]);
      el.innerHTML = svg;
    } catch (e) {
      el.innerHTML = `<pre style="color:#7a0e26">Mermaid error: ${e.message}</pre>`;
    }
  });

  // Wrap wide tables so they scroll horizontally instead of squashing.
  document.querySelectorAll('#content table').forEach((t) => {
    const w = document.createElement('div');
    w.className = 'table-wrap';
    t.parentNode.insertBefore(w, t);
    w.appendChild(t);
  });

  // Slugify + assign ids to headings; build the TOC; add hover anchors.
  const article = document.getElementById('content');
  const used = new Set();
  const slugify = (s) => {
    let base = s.toLowerCase().trim()
      .replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-')
      .replace(/^-|-$/g, '') || 'section';
    let slug = base, n = 2;
    while (used.has(slug)) slug = base + '-' + (n++);
    used.add(slug);
    return slug;
  };

  const tocList = document.getElementById('toc-list');
  const headings = [...article.querySelectorAll('h2, h3')];
  headings.forEach((h) => {
    const id = slugify(h.textContent);
    h.id = id;
    const a = document.createElement('a');
    a.className = 'anchor'; a.href = '#' + id; a.textContent = '#';
    a.setAttribute('aria-hidden', 'true');
    h.prepend(a);

    const li = document.createElement('li');
    const link = document.createElement('a');
    link.href = '#' + id;
    link.textContent = h.textContent.replace(/^#/, '');
    link.className = h.tagName === 'H3' ? 'lvl-3' : 'lvl-2';
    link.dataset.target = id;
    li.appendChild(link);
    tocList.appendChild(li);
  });

  // Scroll-spy: highlight the TOC entry for the heading nearest the top.
  const links = new Map([...tocList.querySelectorAll('a')].map(a => [a.dataset.target, a]));
  let activeId = null;
  const setActive = (id) => {
    if (id === activeId) return;
    if (activeId && links.get(activeId)) links.get(activeId).classList.remove('active');
    activeId = id;
    const a = links.get(id);
    if (a) { a.classList.add('active'); a.scrollIntoView({ block: 'nearest' }); }
  };
  if (headings.length) {
    const barH = parseInt(getComputedStyle(document.documentElement)
                  .getPropertyValue('--bar-h')) || 52;
    const onScroll = () => {
      let current = headings[0].id;
      for (const h of headings) {
        if (h.getBoundingClientRect().top <= barH + 24) current = h.id;
        else break;
      }
      setActive(current);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // Mobile TOC toggle
  const toggle = document.querySelector('.toc-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => document.body.classList.toggle('toc-open'));
    tocList.addEventListener('click', (e) => {
      if (e.target.tagName === 'A') document.body.classList.remove('toc-open');
    });
  }

  // Title sync from first h1
  const firstH1 = article.querySelector('h1');
  if (firstH1) document.title = firstH1.textContent + ' — Wiki';
</script>

</body>
</html>
"""


def find_browser(override: str | None = None) -> tuple[str | None, str]:
    """Return (binary_path, kind). kind ∈ {'chrome', 'override', 'xdg', 'none'}."""
    import glob
    if override:
        p = shutil.which(override) or (override if Path(override).exists() else None)
        if p:
            return p, "override"
        return None, "none"
    for binary in CHROME_CANDIDATES:
        p = shutil.which(binary)
        if p:
            return p, "chrome"
    for pattern in EXTRA_CHROME_PATHS:
        for hit in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            if os.access(hit, os.X_OK):
                return hit, "chrome"
    xdg = shutil.which("xdg-open")
    if xdg:
        return xdg, "xdg"
    return None, "none"


def strip_frontmatter(md_text: str) -> str:
    """Drop a leading YAML frontmatter block (``---`` … ``---``) if present.

    Wiki pages start with frontmatter that should not render as body text.
    Only strips when the very first line is ``---`` and a closing ``---`` is
    found; otherwise returns the text untouched.
    """
    if not md_text.startswith("---"):
        return md_text
    lines = md_text.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return md_text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            rest = lines[i + 1:]
            # drop leading blank lines after the closing fence
            while rest and rest[0].strip() == "":
                rest.pop(0)
            return "".join(rest)
    return md_text  # no closing fence — leave as-is


def derive_title(md_text: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Wiki Synthesis"


def build_html(md_text: str, title: str, source_path: Path | None) -> str:
    payload = base64.b64encode(md_text.encode("utf-8")).decode("ascii")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_parts = [ts]
    if source_path is not None:
        meta_parts.append(source_path.name)
    meta = " · ".join(meta_parts)

    # Breadcrumb in the top bar: "section / page" from the source path.
    if source_path is not None:
        parent = source_path.parent.name
        crumb_text = f"{parent} / {source_path.stem}" if parent else source_path.stem
        crumb = f"&nbsp;/&nbsp;{html.escape(crumb_text)}"
    else:
        crumb = ""

    return (
        HTML_SHELL
        .replace("__TITLE__", html.escape(title))
        .replace("__CRUMB__", crumb)
        .replace("__META__", html.escape(meta))
        .replace("__PAYLOAD__", payload)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("source", help="path to markdown file, or '-' for stdin")
    ap.add_argument("--title", default=None, help="page title (default: first H1)")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--no-open", action="store_true",
                    help="skip opening the browser")
    ap.add_argument("--browser", default=None,
                    help="browser binary to use (path or name on PATH)")
    args = ap.parse_args()

    if args.source == "-":
        md = sys.stdin.read()
        source_path = None
    else:
        src = Path(args.source)
        if not src.exists():
            print(f"[wiki-render] error: {src} does not exist", file=sys.stderr)
            return 2
        md = src.read_text(encoding="utf-8")
        source_path = src

    md = strip_frontmatter(md)
    title = args.title or derive_title(md)
    doc = build_html(md, title, source_path)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        h = hashlib.sha1(md.encode("utf-8")).hexdigest()[:8]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(tempfile.gettempdir()) / f"wiki-render-{ts}-{h}.html"

    out.write_text(doc, encoding="utf-8")
    print(f"[wiki-render] wrote {out}")

    if args.no_open:
        return 0

    binary, kind = find_browser(args.browser)
    if binary is None:
        print(f"[wiki-render] no browser found — open {out} manually",
              file=sys.stderr)
        return 0

    try:
        subprocess.Popen(
            [binary, str(out)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        print(f"[wiki-render] failed to launch {binary}: {e}", file=sys.stderr)
        return 1

    label = {"chrome": "chrome", "override": "browser", "xdg": "system default"}[kind]
    print(f"[wiki-render] opened in {label} ({binary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
