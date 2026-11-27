"""Build wiki bundle JSON from docs/wiki/_manifest.json. Usage: build_wiki.py <repo> <out.json>"""
import json, os, re, sys
from html.parser import HTMLParser

repo, out = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(repo, "docs/wiki/_manifest.json")))
TODAY = "2026-11-23"


class H2M(HTMLParser):
    """Small HTML -> Markdown for founder plan pages (headings, paragraphs, lists, tables, links)."""
    BLOCK = {"p", "div", "section", "article", "br", "tr", "table", "ul", "ol", "header", "footer"}

    def __init__(self):
        super().__init__(); self.o = []; self.skip = 0; self.href = None; self.cell = 0

    def handle_starttag(self, t, a):
        a = dict(a)
        if t in ("script", "style", "head", "svg", "nav"): self.skip += 1
        elif re.fullmatch(r"h[1-6]", t): self.o.append("\n\n" + "#" * int(t[1]) + " ")
        elif t == "li": self.o.append("\n- ")
        elif t in ("td", "th"): self.o.append(" | " if self.cell else "| "); self.cell += 1
        elif t == "a": self.href = a.get("href")
        elif t in ("strong", "b"): self.o.append("**")
        elif t in self.BLOCK: self.o.append("\n\n" if t != "tr" else "\n"); self.cell = 0

    def handle_endtag(self, t):
        if t in ("script", "style", "head", "svg", "nav"): self.skip = max(0, self.skip - 1)
        elif t in ("strong", "b"): self.o.append("**")
        elif t == "tr": self.o.append(" |"); self.cell = 0
        elif t == "a": self.href = None

    def handle_data(self, d):
        if self.skip: return
        d = re.sub(r"\s+", " ", d)
        self.o.append(d)

    def md(self):
        s = "".join(self.o)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return "\n".join(l.rstrip() for l in s.splitlines()).strip() + "\n"


def path(src):
    p = os.path.expanduser(src)
    return p if os.path.isabs(p) else os.path.join(repo, p)


def stories_index(d):
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".md"): continue
        first = open(os.path.join(d, f)).readline().lstrip("# ").strip()
        rows.append(f"| {first.replace('|', '/')} | `project-tasks/stories/{f}` |")
    return ("# User stories\n\nOne file per story in `project-tasks/stories/` (validated by `scripts/validate_stories.py`). "
            "Each story is also a work package on the board with this text as its description: "
            "https://pm.docbot.space/projects/docbot/work_packages\n\n"
            f"{len(rows)} story files.\n\n| Story | File |\n|---|---|\n" + "\n".join(rows) + "\n")


kids = {}
for p in man:
    kids.setdefault(p["parent_slug"], []).append(p)

pages = []
for p in man:
    k, src, slug = p["kind"], p.get("source"), p["slug"]
    if slug == "architecture-00-diagrams" and not src:
        src = "docs/wiki/architecture/00-diagrams.md"
    if slug == "home":
        text = open(path(src)).read()
    elif k == "section":
        text = f"# {p['wiki_title']}\n\n{p.get('notes') or ''}\n\n" + "\n".join(
            f"- [{c['wiki_title']}]({c['slug']})" for c in kids.get(slug, [])) + "\n"
    elif slug == "user-stories":
        text = stories_index(path(src))
    elif src and os.path.isfile(path(src)):
        raw = open(path(src), encoding="utf-8", errors="replace").read()
        if src.endswith(".html"):
            h = H2M(); h.feed(raw); raw = h.md()
        shown = src.replace(os.path.expanduser("~"), "~")
        where = "repo" if not os.path.isabs(os.path.expanduser(src)) else "founder plans folder"
        head = f"> Imported from `{shown}` ({where}) on {TODAY}. Edit the source, not this page; it is re-imported.\n\n"
        text = head + raw
    else:
        print(f"skip (no source yet): {slug} {src}", file=sys.stderr)
        continue
    if p.get("sensitive"):
        text = "> **Sensitive business document.** Private wiki; do not share outside the founding team.\n\n" + text
    att = ["docbot-architecture.light.png",
           "docbot-architecture.html"] if slug == "architecture-00-diagrams" else []
    pages.append({"title": p["wiki_title"], "slug": slug,
                  "parent": None if p["parent_slug"] in (None, "home") else p["parent_slug"], "text": text,
                  "attachments": att})

# Relative slug links break on the start page (/projects/<id>/wiki has no slug), so make them absolute.
SLUGS = {p["slug"] for p in man}
LINK = re.compile(r"\]\(([a-z0-9][a-z0-9-]*)(#[^)]*)?\)")
for pg in pages:
    pg["text"] = LINK.sub(lambda m: f"](/projects/docbot/wiki/{m.group(1)}{m.group(2) or ''})" if m.group(1) in SLUGS else m.group(0),
                          pg["text"])
assert LINK.sub("", "[x](home)") == "[x", "self-check"
# repo-relative file links in imported repo docs -> GitHub (private repo, founder is logged in)
FILE = re.compile(r"\]\((?:\./)?([A-Za-z0-9_][\w./-]*\.(?:md|py|txt|json|ya?ml)|LICENSE)\)")
for pg in pages:
    pg["text"] = FILE.sub(lambda m: f"](https://github.com/ssinghai6/DocBot/blob/main/{m.group(1)})", pg["text"])
json.dump(pages, open(out, "w"))
print(f"{len(pages)} pages, {sum(len(x['text']) for x in pages)//1024} KB")
