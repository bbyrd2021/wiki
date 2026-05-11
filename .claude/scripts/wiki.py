#!/usr/bin/env python3
"""
wiki.py — deterministic operations for the research wiki at /data/repos/wiki/.

Subcommands:
  status                          page counts, dirty files, recent log
  list   [--type T] [--status S]  enumerate pages
  lint   [--json] [--strict]      orphans, broken wikilinks, frontmatter
  search "<query>"                ranked hits with citations
  scaffold <type> <slug> "<title>" [--sources path,path] [--arxiv ID]
                                  create a new page stub with frontmatter
                                  (paper type: refuses if a matching bud exists)
  cite <slug>                     print [[wikilink]] citation for a page
  index-check                     verify index.md covers every page
  recount                         rewrite total_pages in index.md frontmatter

  bud-add     create or update a bud page (idempotent citer-append)
  bud-list    list buds [--citing <slug>] [--min-citers N]
  bud-promote flip status: bud → draft, preserving citing/created
  bud-match   find matching paper or bud by --arxiv | --author --year --title
              (or --suggest-slug to generate a slug)

Stdlib only. Run from anywhere — wiki root is auto-detected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Wiki conventions
# ---------------------------------------------------------------------------

VALID_TYPES = {
    "project", "dataset", "concept", "method", "paper",
    "comparison", "direction", "finding", "tool",
    "index", "log",  # special singletons
}
VALID_STATUS = {"draft", "complete", "stale", "bud"}

# Stopwords used when generating slugs from titles and for title fuzzy matching
TITLE_STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "to", "with", "from", "and", "or",
    "by", "is", "are", "be", "was", "were", "as", "at", "this", "that", "via",
}

REQUIRED_FIELDS = {"type", "title", "created", "updated", "status"}
TYPE_REQUIRED = {
    "paper":    {"authors", "year", "venue"},
    "dataset":  {"clips", "frames", "primary_units", "annotation_format"},
    "direction": {"novelty", "feasibility", "datasets_required"},
}

# Directories under wiki/ that hold pages
PAGE_DIRS = [
    "projects", "datasets", "concepts", "methods",
    "papers", "comparisons", "directions", "findings", "tools",
]
# Things to skip when walking
SKIP_DIRS = {".git", ".obsidian", ".claude", "raw", "worktrees", "node_modules"}

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_wiki_root(start: Path | None = None) -> Path:
    """Walk up until we find a dir containing CLAUDE.md and index.md."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "CLAUDE.md").exists() and (candidate / "index.md").exists():
            return candidate
    # Fallback: hardcoded path
    fallback = Path("/data/repos/wiki")
    if (fallback / "CLAUDE.md").exists():
        return fallback
    sys.exit("error: wiki root not found (no CLAUDE.md + index.md in parents)")


def parse_frontmatter(text: str) -> tuple[dict, int]:
    """Minimal YAML-ish frontmatter parser. Returns (fields, body_offset)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, 0
    raw = m.group(1)
    fields: dict = {}
    key = None
    for line in raw.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Indented continuation (list item or sub-value)
        if line.startswith("  ") or line.startswith("\t"):
            if key is not None and isinstance(fields.get(key), list):
                item = line.strip().lstrip("- ").strip().strip('"\'')
                if item:
                    fields[key].append(item)
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        k = k.strip()
        v = v.strip()
        key = k
        if v == "" or v == "[]":
            fields[k] = []
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            fields[k] = [x.strip().strip('"\'') for x in inner.split(",") if x.strip()] if inner else []
        else:
            fields[k] = v.strip('"\'')
    return fields, m.end()


def iter_pages(root: Path):
    """Yield (path, slug, frontmatter, body) for every page in the wiki."""
    for sub in PAGE_DIRS:
        d = root / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.md")):
            text = p.read_text(encoding="utf-8", errors="replace")
            fm, off = parse_frontmatter(text)
            slug = p.stem
            yield p, slug, fm, text[off:]
    # Singletons
    for name in ("index.md", "log.md"):
        p = root / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            fm, off = parse_frontmatter(text)
            yield p, p.stem, fm, text[off:]


def all_slugs(root: Path) -> set[str]:
    return {slug for _, slug, _, _ in iter_pages(root)}


def relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def git(root: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=False,
    )
    return res.stdout


# ---------------------------------------------------------------------------
# Paper-bud helpers: identity matching, slug generation, paper index
# ---------------------------------------------------------------------------

def _normalize_surname(authors: str) -> str:
    """'Chen et al.' -> 'chen'; 'Pearl' -> 'pearl'; 'Yann LeCun' -> 'lecun'."""
    s = (authors or "").strip()
    s = re.sub(r"\s+et\s+al\.?", "", s, flags=re.IGNORECASE)
    s = s.split(",")[0].strip()
    parts = s.split()
    if parts:
        s = parts[-1]
    s = re.sub(r"[^A-Za-z0-9]+", "", s).lower()
    return s


def _title_tokens(title: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", title or "")
            if t.lower() not in TITLE_STOPWORDS and len(t) > 1]


def _title_keyword(title: str) -> str:
    """Pick a short keyword for slug generation. Prefer text before ':' or '—'."""
    if not title:
        return ""
    head = re.split(r"[:—]", title, maxsplit=1)[0].strip()
    tokens = _title_tokens(head)
    if not tokens:
        tokens = _title_tokens(title)
    return "-".join(tokens[:3])


def _slug_for_citation(authors: str, year, title: str) -> str:
    surname = _normalize_surname(authors)
    year_s = str(year or "").strip()
    keyword = _title_keyword(title)
    parts = [p for p in (surname, year_s, keyword) if p]
    return "-".join(parts)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _paper_index(root: Path) -> dict:
    """Walk papers/ and build a searchable index of frontmatter metadata."""
    by_arxiv: dict[str, str] = {}
    by_author_year: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_slug: dict[str, dict] = {}
    papers_dir = root / "papers"
    if not papers_dir.exists():
        return {"by_arxiv": by_arxiv, "by_author_year": by_author_year, "by_slug": by_slug}
    for p in sorted(papers_dir.rglob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        fm, _ = parse_frontmatter(text)
        slug = p.stem
        by_slug[slug] = fm
        arxiv = (fm.get("arxiv") or "").strip()
        if arxiv:
            by_arxiv[arxiv] = slug
        surname = _normalize_surname(fm.get("authors") or "")
        year = str(fm.get("year") or "").strip()
        if surname and year and year != "0":
            by_author_year[(surname, year)].append(slug)
    return {"by_arxiv": by_arxiv, "by_author_year": by_author_year, "by_slug": by_slug}


def _match_bud_or_paper(root: Path, *, arxiv: str = "", author: str = "",
                       year: str = "", title: str = "") -> tuple[str, str]:
    """
    Find a matching paper page. Returns (slug, status) or ("", "") if no match.
    Precedence: exact arXiv → exact (surname, year) → fuzzy title (Jaccard ≥ 0.5).
    """
    idx = _paper_index(root)
    if arxiv:
        slug = idx["by_arxiv"].get(arxiv.strip())
        if slug:
            return slug, idx["by_slug"][slug].get("status", "")
    if author and year:
        surname = _normalize_surname(author)
        candidates = idx["by_author_year"].get((surname, str(year).strip()), [])
        if len(candidates) == 1:
            slug = candidates[0]
            return slug, idx["by_slug"][slug].get("status", "")
        if len(candidates) > 1 and title:
            target = set(_title_tokens(title))
            best, best_score = "", 0.0
            for c in sorted(candidates):
                cand = set(_title_tokens(idx["by_slug"][c].get("title", "")))
                s = _jaccard(target, cand)
                if s > best_score:
                    best, best_score = c, s
            if best and best_score >= 0.3:
                return best, idx["by_slug"][best].get("status", "")
    if title:
        target = set(_title_tokens(title))
        if target:
            best, best_score = "", 0.0
            for slug, fm in sorted(idx["by_slug"].items()):
                cand = set(_title_tokens(fm.get("title", "")))
                s = _jaccard(target, cand)
                if s > best_score:
                    best, best_score = slug, s
            if best and best_score >= 0.5:
                return best, idx["by_slug"][best].get("status", "")
    return "", ""


def _yaml_escape(s: str) -> str:
    """Minimal escape for double-quoted YAML string."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _write_bud(path: Path, *, title: str, authors: str, year, venue: str,
               arxiv: str, created: str, updated: str, citing: list[str],
               cited_by_block: str) -> None:
    """Render a bud page deterministically."""
    citing_yaml = "[" + ", ".join(citing) + "]" if citing else "[]"
    lines = [
        "---",
        "type: paper",
        f'title: "{_yaml_escape(title)}"',
        "aliases: []",
        f"created: {created}",
        f"updated: {updated}",
        "sources: []",
        "tags: [paper, bud]",
        "status: bud",
        f'authors: "{_yaml_escape(authors)}"',
        f"year: {int(year) if str(year).strip().isdigit() else 0}",
        f'venue: "{_yaml_escape(venue)}"',
        f'arxiv: "{_yaml_escape(arxiv)}"',
        f"citing: {citing_yaml}",
        "---",
        "",
        f"# {title}",
        "",
        "> **Bud page.** Stub created from a citation. Promoted to a full paper page on ingest.",
        "",
        "## Cited by",
        "",
        cited_by_block.rstrip(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_cited_by(text: str) -> str:
    """Pull the existing '## Cited by' body content (bullets only)."""
    m = re.search(r"^##\s+Cited by\s*$(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(args) -> int:
    root = find_wiki_root()
    by_type: Counter = Counter()
    by_status: Counter = Counter()
    for _, _, fm, _ in iter_pages(root):
        t = fm.get("type", "?")
        if t in ("index", "log"):
            continue
        by_type[t] += 1
        by_status[fm.get("status", "?")] += 1
    bud_count = by_status.get("bud", 0)
    total = sum(by_type.values()) - bud_count
    print(f"wiki: {root}")
    print(f"total pages: {total}" + (f" (+{bud_count} bud{'s' if bud_count != 1 else ''})" if bud_count else ""))
    print("by type:    " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print("by status:  " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    dirty = git(root, "status", "--porcelain")
    if dirty.strip():
        print("dirty:")
        for line in dirty.rstrip("\n").split("\n"):
            print(f"  {line}")
    else:
        print("git: clean")
    print("recent log:")
    log = (root / "log.md").read_text(encoding="utf-8", errors="replace")
    headers = [l for l in log.split("\n") if l.startswith("## [")][:5]
    for h in headers:
        print(f"  {h}")
    return 0


def cmd_list(args) -> int:
    root = find_wiki_root()
    rows = []
    for p, slug, fm, _ in iter_pages(root):
        t = fm.get("type", "?")
        if t in ("index", "log"):
            continue
        if args.type and t != args.type:
            continue
        if args.status and fm.get("status") != args.status:
            continue
        rows.append((t, fm.get("status", "?"), slug, fm.get("title", ""), relpath(p, root)))
    rows.sort()
    for t, s, slug, title, path in rows:
        print(f"{t:11s} {s:8s} {slug:42s} {title}")
    print(f"-- {len(rows)} pages")
    return 0


def cmd_lint(args) -> int:
    root = find_wiki_root()
    issues: list[dict] = []

    pages = list(iter_pages(root))
    slug_to_paths: dict[str, list[Path]] = defaultdict(list)
    for p, slug, _, _ in pages:
        slug_to_paths[slug].append(p)

    valid_slugs = set(slug_to_paths.keys())

    # Index.md content for orphan check
    index_text = (root / "index.md").read_text(encoding="utf-8", errors="replace")
    # Unescape Obsidian table-pipe/bracket escapes before matching wikilinks
    index_scan = index_text.replace("\\|", "|").replace("\\]", "]")
    index_slugs_mentioned = set(WIKILINK_RE.findall(index_scan))
    # The index also references via paths like projects/foo — capture those too
    index_path_mentions = set(re.findall(r"\[\[([a-z0-9\-]+/[a-z0-9\-]+)", index_scan))
    index_slugs_mentioned |= {x.split("/")[-1] for x in index_path_mentions}

    for p, slug, fm, body in pages:
        rel = relpath(p, root)
        t = fm.get("type", "?")

        # Filename style
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
            issues.append({"file": rel, "kind": "slug-style", "msg": f"slug '{slug}' is not lowercase kebab-case"})

        # Singletons: skip the heavy checks
        if t in ("index", "log"):
            continue

        # Required frontmatter
        if not fm:
            issues.append({"file": rel, "kind": "frontmatter-missing", "msg": "no frontmatter"})
            continue
        missing = REQUIRED_FIELDS - fm.keys()
        if missing:
            issues.append({"file": rel, "kind": "field-missing", "msg": f"missing required fields: {sorted(missing)}"})
        if t not in VALID_TYPES:
            issues.append({"file": rel, "kind": "type-invalid", "msg": f"type '{t}' not in {sorted(VALID_TYPES)}"})
        if fm.get("status") and fm["status"] not in VALID_STATUS:
            issues.append({"file": rel, "kind": "status-invalid", "msg": f"status '{fm['status']}' not in {sorted(VALID_STATUS)}"})
        # Type-specific required — skip for buds (best-effort fields)
        if fm.get("status") != "bud":
            for extra in TYPE_REQUIRED.get(t, set()) - fm.keys():
                issues.append({"file": rel, "kind": "type-field-missing", "msg": f"type={t} missing field '{extra}'"})

        # Stale flag
        if fm.get("status") == "stale":
            issues.append({"file": rel, "kind": "stale", "msg": "status: stale — needs review"})

        # Bud-specific soft signals
        if fm.get("status") == "bud":
            citing = fm.get("citing") or []
            if not isinstance(citing, list) or len(citing) == 0:
                issues.append({"file": rel, "kind": "bud-no-citing", "msg": "bud with no citing entries — orphaned bud"})
            created = fm.get("created", "")
            if created:
                try:
                    created_dt = dt.date.fromisoformat(str(created))
                    age_days = (dt.date.today() - created_dt).days
                    if age_days > 365:
                        issues.append({"file": rel, "kind": "bud-stale", "msg": f"bud seeded {age_days}d ago, never promoted"})
                except (ValueError, TypeError):
                    pass

        # Broken wikilinks (unescape Obsidian table escapes first)
        body_scan = body.replace("\\|", "|").replace("\\]", "]")
        for target in set(WIKILINK_RE.findall(body_scan)):
            # Could be "slug" or "folder/slug"
            target_slug = target.split("/")[-1]
            if target_slug not in valid_slugs:
                issues.append({"file": rel, "kind": "wikilink-broken", "msg": f"[[{target}]] → no such page"})

        # Orphan: not mentioned in index.md — buds are intentionally absent
        if fm.get("status") != "bud" and slug not in index_slugs_mentioned:
            issues.append({"file": rel, "kind": "orphan", "msg": f"slug '{slug}' not mentioned in index.md"})

    # Duplicate slugs across folders
    for slug, paths in slug_to_paths.items():
        if len(paths) > 1:
            issues.append({
                "file": ",".join(relpath(p, root) for p in paths),
                "kind": "slug-duplicate",
                "msg": f"slug '{slug}' appears in multiple folders",
            })

    # total_pages claim in index.md — buds are intentionally absent from the index
    idx_fm, _ = parse_frontmatter(index_text)
    claimed = idx_fm.get("total_pages")
    actual = sum(1 for _, _, fm, _ in pages
                 if fm.get("type") not in ("index", "log") and fm.get("status") != "bud")
    if claimed and str(claimed) != str(actual):
        issues.append({
            "file": "index.md",
            "kind": "total-pages-mismatch",
            "msg": f"frontmatter total_pages={claimed} but actual count is {actual}",
        })

    if args.json:
        print(json.dumps(issues, indent=2))
    else:
        by_kind = defaultdict(list)
        for i in issues:
            by_kind[i["kind"]].append(i)
        order = ["frontmatter-missing", "field-missing", "type-field-missing", "type-invalid",
                 "status-invalid", "wikilink-broken", "orphan", "slug-duplicate",
                 "slug-style", "total-pages-mismatch", "stale", "bud-no-citing", "bud-stale"]
        for k in order + sorted(set(by_kind) - set(order)):
            items = by_kind.get(k, [])
            if not items:
                continue
            print(f"\n== {k} ({len(items)}) ==")
            for i in items:
                print(f"  {i['file']}: {i['msg']}")
        print(f"\n{len(issues)} issue(s) across {actual} pages")
    return 1 if (args.strict and issues) else 0


def _tokenize(q: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-+]*", q) if len(t) > 1]


def cmd_search(args) -> int:
    root = find_wiki_root()
    terms = _tokenize(args.query)
    if not terms:
        sys.exit("error: empty query")
    results = []
    for p, slug, fm, body in iter_pages(root):
        if fm.get("type") in ("index", "log"):
            continue
        title = fm.get("title", "")
        aliases = " ".join(fm.get("aliases", []) if isinstance(fm.get("aliases"), list) else [])
        tags = " ".join(fm.get("tags", []) if isinstance(fm.get("tags"), list) else [])
        hay_title = (title + " " + aliases + " " + slug).lower()
        hay_tags = tags.lower()
        hay_body = body.lower()
        score = 0
        snippet = ""
        for t in terms:
            score += 8 * hay_title.count(t)
            score += 4 * hay_tags.count(t)
            score += 1 * hay_body.count(t)
        if score > 0:
            # Build snippet: first body line containing any term
            for line in body.split("\n"):
                ll = line.lower()
                if any(t in ll for t in terms) and line.strip() and not line.startswith("#"):
                    snippet = line.strip()[:160]
                    break
            results.append((score, slug, title, fm.get("type", "?"), relpath(p, root), snippet, fm))
    results.sort(reverse=True)
    n = args.limit
    for score, slug, title, t, path, snip, fm in results[:n]:
        sources = fm.get("sources") or []
        if isinstance(sources, list) and sources:
            cite = f"  sources: {', '.join(sources)}"
        else:
            cite = ""
        print(f"[{score:>4}] {t:9s} [[{slug}]] {title}")
        print(f"        {path}")
        if snip:
            print(f"        {snip}")
        if cite:
            print(cite)
    print(f"-- {len(results)} hits, top {min(n, len(results))} shown")
    return 0


def cmd_scaffold(args) -> int:
    root = find_wiki_root()
    t = args.type
    if t not in VALID_TYPES:
        sys.exit(f"error: type '{t}' not in {sorted(VALID_TYPES)}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", args.slug):
        sys.exit(f"error: slug '{args.slug}' must be lowercase kebab-case")
    # Pluralize: type → folder
    folder_map = {
        "project": "projects", "dataset": "datasets", "concept": "concepts",
        "method": "methods", "paper": "papers", "comparison": "comparisons",
        "direction": "directions", "finding": "findings", "tool": "tools",
    }
    folder = folder_map.get(t)
    if not folder:
        sys.exit(f"error: type '{t}' is a singleton, not a scaffoldable page")
    out = root / folder / f"{args.slug}.md"

    # For papers: detect an existing bud (same slug, OR matching arxiv/title) and
    # refuse to overwrite — direct user to bud-promote instead.
    if t == "paper" and not args.force:
        if out.exists():
            existing_fm, _ = parse_frontmatter(out.read_text(encoding="utf-8", errors="replace"))
            if existing_fm.get("status") == "bud":
                sys.exit(
                    f"error: bud at papers/{args.slug}.md\n"
                    f"  use: python3 .claude/scripts/wiki.py bud-promote {args.slug}"
                )
        matched_slug, matched_status = _match_bud_or_paper(
            root,
            arxiv=(args.arxiv or "").strip(),
            title=args.title or "",
        )
        if matched_slug and matched_slug != args.slug:
            if matched_status == "bud":
                sys.exit(
                    f"error: matching bud at papers/{matched_slug}.md (status: bud)\n"
                    f"  use: python3 .claude/scripts/wiki.py bud-promote {matched_slug}"
                )
            sys.exit(
                f"error: paper already exists at papers/{matched_slug}.md "
                f"(status: {matched_status})"
            )

    if out.exists() and not args.force:
        sys.exit(f"error: {out} already exists (pass --force to overwrite)")
    today = dt.date.today().isoformat()
    sources = []
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    fm_lines = [
        "---",
        f"type: {t}",
        f'title: "{args.title}"',
        "aliases: []",
        f"created: {today}",
        f"updated: {today}",
        "sources:" + (" []" if not sources else ""),
    ]
    for s in sources:
        fm_lines.append(f'  - "{s}"')
    fm_lines += ["tags: []", "status: draft"]
    # Type-specific stubs
    if t == "paper":
        arxiv_init = (args.arxiv or "").strip()
        fm_lines += ['authors: ""', "year: 0", 'venue: ""', f'arxiv: "{arxiv_init}"']
    elif t == "dataset":
        fm_lines += ["clips: 0", "frames: 0", "primary_units: 0", 'annotation_format: ""']
    elif t == "direction":
        fm_lines += ["novelty: true", 'feasibility: "workstation"', "datasets_required: []"]
    fm_lines += ["---", "", f"# {args.title}", "", "<!-- stub: fill in body -->", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(fm_lines), encoding="utf-8")
    print(f"created: {relpath(out, root)}")
    return 0


def cmd_cite(args) -> int:
    root = find_wiki_root()
    for p, slug, fm, _ in iter_pages(root):
        if slug == args.slug:
            title = fm.get("title", slug)
            sources = fm.get("sources") or []
            sline = f" — sources: {', '.join(sources)}" if isinstance(sources, list) and sources else ""
            print(f"[[{slug}|{title}]]{sline}")
            print(f"  path: {relpath(p, root)}")
            return 0
    sys.exit(f"error: no page with slug '{args.slug}'")


def cmd_index_check(args) -> int:
    root = find_wiki_root()
    index_text = (root / "index.md").read_text(encoding="utf-8", errors="replace")
    mentioned = set(WIKILINK_RE.findall(index_text))
    mentioned |= {x.split("/")[-1] for x in re.findall(r"\[\[([a-z0-9\-]+/[a-z0-9\-]+)", index_text)}
    orphans = []
    for _, slug, fm, _ in iter_pages(root):
        if fm.get("type") in ("index", "log"):
            continue
        if slug not in mentioned:
            orphans.append(slug)
    for o in orphans:
        print(f"orphan: {o}")
    print(f"-- {len(orphans)} orphan(s)")
    return 1 if orphans else 0


def cmd_recount(args) -> int:
    root = find_wiki_root()
    actual = sum(1 for _, _, fm, _ in iter_pages(root)
                 if fm.get("type") not in ("index", "log") and fm.get("status") != "bud")
    idx_path = root / "index.md"
    text = idx_path.read_text(encoding="utf-8")
    new_text, n = re.subn(r"^total_pages:\s*\d+", f"total_pages: {actual}", text, count=1, flags=re.MULTILINE)
    if n == 0:
        sys.exit("error: could not find 'total_pages:' line in index.md frontmatter")
    # Also update 'updated:' to today
    today = dt.date.today().isoformat()
    new_text, _ = re.subn(r"^updated:\s*\d{4}-\d{2}-\d{2}", f"updated: {today}", new_text, count=1, flags=re.MULTILINE)
    idx_path.write_text(new_text, encoding="utf-8")
    print(f"index.md: total_pages = {actual} (updated: {today})")
    return 0


# ---------------------------------------------------------------------------
# Bud commands
# ---------------------------------------------------------------------------

def cmd_bud_add(args) -> int:
    root = find_wiki_root()
    slug = args.slug
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
        sys.exit(f"error: slug '{slug}' must be lowercase kebab-case")
    if args.citing and args.citing == slug:
        sys.exit(f"error: a paper cannot cite itself ({slug})")
    papers_dir = root / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    path = papers_dir / f"{slug}.md"
    today = dt.date.today().isoformat()

    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        fm, _ = parse_frontmatter(text)
        if fm.get("status") != "bud":
            print(f"already a node: {slug} (status: {fm.get('status', '?')}) — no-op")
            return 0
        # Append citer to existing bud
        citing = fm.get("citing") or []
        if not isinstance(citing, list):
            citing = []
        if not args.citing:
            print(f"bud exists: {slug} — pass --citing to register a citer")
            return 0
        if args.citing in citing:
            print(f"already cited by {args.citing}: {slug} — no-op")
            return 0
        citing.append(args.citing)
        existing = _extract_cited_by(text)
        ctx_s = f" — {args.context}" if args.context else ""
        new_line = f"- [[papers/{args.citing}]]{ctx_s}"
        cited_by = (existing + "\n" + new_line) if existing else new_line
        _write_bud(path,
                   title=fm.get("title", slug),
                   authors=fm.get("authors", ""),
                   year=fm.get("year", 0),
                   venue=fm.get("venue", ""),
                   arxiv=fm.get("arxiv", ""),
                   created=fm.get("created", today),
                   updated=today,
                   citing=citing,
                   cited_by_block=cited_by)
        print(f"appended: {slug} now cited by {len(citing)} paper(s)")
        return 0

    # New bud — must have a title
    if not args.title:
        sys.exit("error: --title required for new bud")
    if not args.citing:
        sys.exit("error: --citing required for new bud (a bud must be cited by something)")
    # Check for an existing real paper by arxiv/title before creating
    matched_slug, matched_status = _match_bud_or_paper(
        root,
        arxiv=args.arxiv or "",
        author=args.authors or "",
        year=str(args.year or ""),
        title=args.title,
    )
    if matched_slug and matched_status != "bud":
        print(f"already a node: {matched_slug} (status: {matched_status}) — no bud created")
        return 0
    if matched_slug and matched_status == "bud" and matched_slug != slug:
        sys.exit(f"error: matching bud already exists at papers/{matched_slug}.md "
                 f"(slug differs from {slug}). Re-run with --slug {matched_slug}.")

    citing = [args.citing]
    ctx_s = f" — {args.context}" if args.context else ""
    cited_by = f"- [[papers/{args.citing}]]{ctx_s}"
    _write_bud(path,
               title=args.title,
               authors=args.authors or "",
               year=args.year or 0,
               venue=args.venue or "",
               arxiv=args.arxiv or "",
               created=today, updated=today,
               citing=citing, cited_by_block=cited_by)
    print(f"created bud: {slug}")
    return 0


def cmd_bud_list(args) -> int:
    root = find_wiki_root()
    rows = []
    for p, slug, fm, _ in iter_pages(root):
        if fm.get("type") != "paper" or fm.get("status") != "bud":
            continue
        citing = fm.get("citing") or []
        if not isinstance(citing, list):
            citing = []
        if args.citing and args.citing not in citing:
            continue
        if args.min_citers and len(citing) < args.min_citers:
            continue
        rows.append((len(citing), slug, fm.get("title", ""), citing, fm.get("arxiv", "")))
    rows.sort(key=lambda r: (-r[0], r[1]))
    for n, slug, title, citing, arxiv in rows:
        arxiv_s = f" arxiv:{arxiv}" if arxiv else ""
        print(f"[{n} citer{'s' if n != 1 else ''}] {slug:42s} {title}{arxiv_s}")
        print(f"           cited by: {', '.join(citing)}")
    print(f"-- {len(rows)} bud(s)")
    return 0


def cmd_bud_promote(args) -> int:
    root = find_wiki_root()
    path = root / "papers" / f"{args.slug}.md"
    if not path.exists():
        sys.exit(f"error: no paper at papers/{args.slug}.md")
    text = path.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    if fm.get("status") != "bud":
        sys.exit(f"error: papers/{args.slug}.md has status='{fm.get('status', '?')}', not 'bud'")
    today = dt.date.today().isoformat()
    new_text, n_status = re.subn(r"^status:\s*bud\s*$", "status: draft", text,
                                 count=1, flags=re.MULTILINE)
    if n_status == 0:
        sys.exit(f"error: could not find 'status: bud' line to flip in {path}")
    new_text, _ = re.subn(r"^updated:\s*\S+\s*$", f"updated: {today}", new_text,
                         count=1, flags=re.MULTILINE)
    path.write_text(new_text, encoding="utf-8")
    citing = fm.get("citing") or []
    print(f"promoted: {args.slug} (status: bud → draft)")
    if citing:
        print(f"  preserved citing: {', '.join(citing) if isinstance(citing, list) else citing}")
    print(f"  next: edit the body and flip status to 'complete' when done")
    return 0


def cmd_bud_match(args) -> int:
    root = find_wiki_root()
    if args.suggest_slug:
        if not (args.author and args.year and args.title):
            sys.exit("error: --suggest-slug requires --author, --year, --title")
        print(_slug_for_citation(args.author, args.year, args.title))
        return 0
    if not (args.arxiv or (args.author and args.year) or args.title):
        sys.exit("error: provide --arxiv OR (--author AND --year) OR --title")
    slug, status = _match_bud_or_paper(
        root,
        arxiv=args.arxiv or "",
        author=args.author or "",
        year=args.year or "",
        title=args.title or "",
    )
    if slug:
        print(f"match: {slug} (status: {status})")
        return 0
    print("no match")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="wiki", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p_list = sub.add_parser("list")
    p_list.add_argument("--type")
    p_list.add_argument("--status")
    p_list.set_defaults(fn=cmd_list)

    p_lint = sub.add_parser("lint")
    p_lint.add_argument("--json", action="store_true")
    p_lint.add_argument("--strict", action="store_true", help="exit nonzero if any issues")
    p_lint.set_defaults(fn=cmd_lint)

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(fn=cmd_search)

    p_scaf = sub.add_parser("scaffold")
    p_scaf.add_argument("type")
    p_scaf.add_argument("slug")
    p_scaf.add_argument("title")
    p_scaf.add_argument("--sources", help="comma-separated source paths")
    p_scaf.add_argument("--arxiv", default="", help="arxiv ID (paper type only — also used for bud detection)")
    p_scaf.add_argument("--force", action="store_true")
    p_scaf.set_defaults(fn=cmd_scaffold)

    p_cite = sub.add_parser("cite")
    p_cite.add_argument("slug")
    p_cite.set_defaults(fn=cmd_cite)

    sub.add_parser("index-check").set_defaults(fn=cmd_index_check)
    sub.add_parser("recount").set_defaults(fn=cmd_recount)

    p_bud_add = sub.add_parser("bud-add", help="create or update a bud page")
    p_bud_add.add_argument("--slug", required=True)
    p_bud_add.add_argument("--title", default="")
    p_bud_add.add_argument("--authors", default="")
    p_bud_add.add_argument("--year", type=int, default=0)
    p_bud_add.add_argument("--venue", default="")
    p_bud_add.add_argument("--arxiv", default="")
    p_bud_add.add_argument("--citing", default="", help="slug of the paper citing this bud")
    p_bud_add.add_argument("--context", default="", help="one-line role of the citation")
    p_bud_add.set_defaults(fn=cmd_bud_add)

    p_bud_list = sub.add_parser("bud-list", help="list all buds")
    p_bud_list.add_argument("--citing", default="", help="filter to buds cited by this slug")
    p_bud_list.add_argument("--min-citers", type=int, default=0)
    p_bud_list.set_defaults(fn=cmd_bud_list)

    p_bud_promote = sub.add_parser("bud-promote", help="flip status: bud → draft")
    p_bud_promote.add_argument("slug")
    p_bud_promote.set_defaults(fn=cmd_bud_promote)

    p_bud_match = sub.add_parser("bud-match", help="find matching paper/bud or suggest a slug")
    p_bud_match.add_argument("--arxiv", default="")
    p_bud_match.add_argument("--author", default="")
    p_bud_match.add_argument("--year", default="")
    p_bud_match.add_argument("--title", default="")
    p_bud_match.add_argument("--suggest-slug", action="store_true",
                             help="generate slug from --author/--year/--title instead of matching")
    p_bud_match.set_defaults(fn=cmd_bud_match)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
