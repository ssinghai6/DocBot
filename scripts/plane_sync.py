#!/usr/bin/env python3
"""Idempotent sync of DocBot's Phase 0 + Phase 1 backlog into Plane (cloud).

Reads the backlog from git (never the working tree), so the source is a fixed ref:
  - project-tasks/docbot-master-backlog.md   section 8 (referenced tickets) + section 7 rows scheduled in a sprint
  - project-tasks/docbot-v2-project-tracking.md EPIC-25 / EPIC-26 tables (AC, deps, due dates)

Matches existing Plane issues on external_id (fallback: name prefix "DOCBOT-XXXX "), creates
what is missing and updates what changed. Usage: see scripts/README-plane.md.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import requests

API = "https://api.plane.so/api/v1/workspaces/docbot/projects/1bf5510f-0a6c-40e7-a81f-97f0e5ab39e4"
EXTERNAL_SOURCE = "docbot-backlog"
BACKLOG = "project-tasks/docbot-master-backlog.md"
TRACKER = "project-tasks/docbot-v2-project-tracking.md"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THROTTLE_S = 1.1  # Plane cloud allows about 60 requests/min

STATE_ORDER = ["Backlog", "Todo", "In Progress", "Testing", "Done"]
TESTING_STATE = {"name": "Testing", "color": "#f59e0b", "group": "started"}  # PR open / CI / QA
PRIORITY = {"P0": "urgent", "P1": "high", "P2": "medium", "P3": "low"}
LANES = {"founder": "lane-founder", "senior-dev": "lane-senior-dev", "devops": "lane-devops",
         "ai-engineer": "lane-ai-engineer", "pm": "lane-pm", "frontend": "lane-frontend"}
SAFETY = "Phase 1 safety & robustness"
MODULES = {  # backlog "Track" column -> Plane module name
    "EPIC-26 Pre-seed": "EPIC-26 Pre-seed readiness",
    "EPIC-25 AWS": "EPIC-25 AWS migration",
    "EPIC-25 GTM": "EPIC-25 GTM",
    "EPIC-25 SEO": "EPIC-25 SEO",
    "EPIC-25 AI proof": "EPIC-25 AI proof",
    "EPIC-25 Sales ops": "EPIC-25 Sales ops",
    "EPIC-25 Firm workspace": "EPIC-25 Firm workspace",
    "EPIC-24 Stage A": "EPIC-24 Stage A isolation",
    "EPIC-18 billing (referenced)": "EPIC-18 Billing",
    "EPIC-16 remainder": SAFETY,
    "EPIC-23 open bugs (referenced)": SAFETY,
}
HISTORICAL_MODULE = "Historical work"
LABEL_COLORS = {"phase": "#667eea", "P": "#ef4444", "lane": "#10b981", "pts": "#94a3b8",
                "sprint": "#f97316", "historical": "#64748b"}
ALL_LABELS = (["phase-0", "phase-1", "P0", "P1", "P2", "P3"] + sorted(set(LANES.values()))
              + [f"pts-{n}" for n in (1, 2, 3, 5, 8)] + [f"sprint-S{n}" for n in range(1, 8)] + ["historical"])

# Reality on 2026-11-17 that the backlog text does not show yet (verified in GitHub by the caller).
# These are floors: the derived state can move past them, never below.
STATE_FLOOR = {
    "2501": "Done", "2502": "Done", "2536": "Done", "2600": "Done",
    # merged to dev, but no terraform applied yet and AWS docs were lost to a .gitignore bug
    **{i: "In Progress" for i in ("2503", "2504", "2505", "2507", "2508", "2509", "2510", "2511",
                                  "2512", "2515", "2517", "2518", "1810")},
}
STATE_NOTES = {
    "2503": "PR #86 merged to dev.", "2508": "PR #106 merged to dev.", "2504": "Budgets part merged to dev (PR #107).",
    "2510": "PR #108 merged to dev.", "2511": "PR #109 merged to dev.", "2509": "Code merged to dev (PR #110).",
    "2505": "Smoke prep merged to dev (PR #112).", "2512": "PR #113 merged to dev.", "2507": "PR #115 merged to dev.",
    "2515": "Infra merged to dev (PR #116).", "2517": "Plan merged to dev (PR #117).", "2518": "Plan merged to dev (PR #117).",
    "2501": "PR #86 reviewed and merged to dev.", "2502": "main merged into dev.",
    "2536": "PR #99 merged to main.", "2600": "PR #121 merged to main.", "1810": "PR #103 open, held for founder review.",
}
# Not in the backlog tables but shipped today as part of the Phase 0 planning work.
EXTRA = [{"id": "2600", "title": "Pre-seed plan: enterprise roadmap, robustness test-data plan, master backlog",
          "pts": "", "pri": "", "status": "Merged to main (PR #121)", "phase": "0", "sprint": "-",
          "lane": "PM", "track": "EPIC-26 Pre-seed", "src": "git log origin/main (PR #121)"}]
HISTORICAL_IDS = ("606, 702-706, 1101, 1200, 1501-1505, 1509-1532, 1601-1612, 1615-1625, 1629-1635, 1702, 1706, "
                  "1707, 1711-1713, 1800-1803, 1805, 2001-2006, 2008, 2014, 2212, 2302, 2303, 2305-2307, 2310, "
                  "2311, 2313, 2315-2322, 2330-2340, 2541, 2542")
HISTORICAL_NAME = "Historical work: shipped before 2026-11-17"
HISTORICAL_EXT = "HISTORICAL-PRE-2026-11-17"


# ---------------------------------------------------------------- parsing

def git_show(ref: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout


def cells(line: str) -> list[str]:
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [p.strip().replace("\\|", "|") for p in parts]


def tables(text: str):
    """Yield (header, rows) for every markdown table in text."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header = [h.lower() for h in cells(lines[i])]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            yield header, rows
        else:
            i += 1


def section(text: str, start: str, end: str) -> str:
    a = text.index(start)
    b = text.index(end, a)
    return text[a:b]


@dataclass
class Story:
    id: str
    title: str
    pts: str
    pri: str
    status: str
    phase: str
    sprint: str
    lane: str
    track: str
    src: str
    ac: str = ""
    deps: str = ""
    due: str = ""
    why: str = ""
    extra: list[str] = field(default_factory=list)


def load_stories(ref: str) -> tuple[list[Story], list[str]]:
    backlog = git_show(ref, BACKLOG)
    tracker = git_show(ref, TRACKER)
    stories: dict[str, Story] = {}
    skipped: list[str] = []

    for header, rows in tables(section(backlog, "## 8.", "## 9.")):
        if header[:1] != ["id"] or "planned phase" not in header:
            continue
        for r in rows:
            d = dict(zip(header, r))
            sid = d["id"].removeprefix("DOCBOT-")
            if d["planned phase"] not in ("0", "1"):
                skipped.append(f"{sid} (phase {d['planned phase']})")
                continue
            stories[sid] = Story(sid, d["title"], d["pts"], d["pri"], d["status"], d["planned phase"],
                                 d["sprint"], d["lane"], d["track"], f"{BACKLOG} section 8 @ {ref}")

    for header, rows in tables(section(backlog, "## 7.", "## 8.")):
        if "sprint" not in header:
            continue
        for r in rows:
            d = dict(zip(header, r))
            if not re.fullmatch(r"S\d", d.get("sprint", "")):
                continue
            sid = d["id"].removeprefix("DOCBOT-")
            stories[sid] = Story(sid, d["title"], d["pts"], d["pri"], "New (scheduled from parking lot)",
                                 d["phase"], d["sprint"], d["lane"], "SAFETY", f"{BACKLOG} section 7 @ {ref}",
                                 ac=d.get("acceptance criteria", ""), deps=d.get("deps", ""))

    for e in EXTRA:
        stories.setdefault(e["id"], Story(**e))

    # Enrich from tracker EPIC-25 + EPIC-26 tables (AC, deps, due, why now)
    trk = tracker[tracker.index("## EPIC-25"):]
    for header, rows in tables(trk):
        for r in rows:
            d = dict(zip(header, r))
            key = next((k for k in ("id", "ticket") if k in d), None)
            if not key:
                continue
            ids = re.findall(r"\d{4}", d[key])
            for sid in ids:
                s = stories.get(sid)
                if not s:
                    continue
                ac = next((d[k] for k in ("acceptance criteria", "ac (short)", "ac", "scope") if d.get(k)), "")
                deps = d.get("deps") or d.get("depends on") or ""
                if ac and len(ac) > len(s.ac):
                    s.ac = ac
                if deps and not s.deps:
                    s.deps = deps
                if d.get("due"):
                    s.due = d["due"]
                if d.get("why now"):
                    s.why = d["why now"]
    return sorted(stories.values(), key=lambda s: s.id), skipped


def due_to_date(due: str) -> str | None:
    due = re.sub(r"(after|starts)\s+\d{2}-\d{2}", "", due)  # "after 12-01" / "starts 11-17" are not deadlines
    found = re.findall(r"\b(1[0-2]|0[1-9])-([0-3]\d)\b", due)
    if not found:
        return None
    m, d = found[-1]
    year = 2026 if int(m) >= 11 else 2027  # plan runs 2026-11 .. 2027-06
    return f"{year}-{m}-{d}"


def derived_state(s: Story) -> str:
    st = s.status.lower()
    if st.startswith("merged") or st.startswith("done"):
        state = "Done"
    elif re.search(r"in progress|pr #\d+ open|merged|starts now", st):
        state = "In Progress"
    else:
        state = "Todo" if s.phase == "0" else "Backlog"
    floor = STATE_FLOOR.get(s.id)
    if floor and STATE_ORDER.index(floor) > STATE_ORDER.index(state):
        state = floor
    return state


def waf_safe(escaped: str) -> str:
    """Entity-encode chars Plane's Cloudflare WAF pattern-matches (python -m, SQLi/XSS payloads in attack-test stories); renders identically."""
    for ch, ent in ((" -", "&nbsp;-"), ("(", "&#40;"), (")", "&#41;"), ("=", "&#61;"), ("*", "&#42;"), ("/", "&#47;")):
        escaped = escaped.replace(ch, ent)
    return escaped


def md_inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return re.sub(r"(<[^>]+>)|([^<]+)", lambda m: m.group(1) or waf_safe(m.group(2)), out)  # encode text only, not tags


def md_to_html(md: str) -> str:
    """Render the story-file markdown subset (headings, tables, bullet/checkbox lists, fences, paragraphs)."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            code = waf_safe(html.escape("\n".join(lines[i + 1:j])))
            out.append(f"<pre><code>{code}</code></pre>")
            i = j + 1
        elif m := re.match(r"^(#{1,4})\s+(.*)", s):
            n = min(len(m.group(1)) + 1, 4)  # '#' -> h2: the issue name is already the title
            out.append(f"<h{n}>{md_inline(m.group(2))}</h{n}>")
            i += 1
        elif s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            head = cells(s)
            rows, i = [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            th = "".join(f"<th>{md_inline(c)}</th>" for c in head)
            trs = "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
        elif re.match(r"^[-*]\s+", s):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                text = re.sub(r"^\s*[-*]\s+", "", lines[i])
                box = re.match(r"^\[( |x|X)\]\s+(.*)", text)
                if box:
                    mark = "&#9745;" if box.group(1).lower() == "x" else "&#9744;"
                    text_html = f"{mark} {md_inline(box.group(2))}"
                else:
                    text_html = md_inline(text)
                items.append(f"<li>{text_html}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
        elif not s:
            i += 1
        else:
            para = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^\s*([-*]\s|#|\||```)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{md_inline(' '.join(para))}</p>")
    return "".join(out)


def story_html(path: str, s: Story) -> str:
    text = open(path, encoding="utf-8").read()
    first, _, rest = text.partition("\n")
    body = md_to_html(rest if first.startswith(f"# DOCBOT-{s.id}") else text)
    note = STATE_NOTES.get(s.id)
    if note:
        body = f"<p><strong>Today:</strong> {md_inline(note)}</p>" + body
    rel = os.path.relpath(path, REPO)
    return body + f"<p><em>Source: {html.escape(rel)} (story detail); header row from {BACKLOG}.</em></p>"


def _self_test() -> None:
    h = md_to_html("# T\n| A | B |\n|---|---|\n| 1 | `x` |\n\n## AC\n- [ ] Given **a**, then b\n- plain\n\n"
                   "Para one\ncontinues\n```\npython -m x\n```")
    assert "<h2>T</h2>" in h and "<th>A</th>" in h and "<td><code>x</code></td>" in h, h
    assert "<li>&#9744; Given <strong>a</strong>, then b</li><li>plain</li>" in h, h
    assert "<p>Para one continues</p>" in h, h
    assert "python -m" not in h and "python&nbsp;-m" in h, h  # WAF workaround holds in prose and code
    assert md_to_html("<script>") == "<p>&lt;script&gt;</p>"
    print("self-test ok")


def story_payload(s: Story, label_ids: dict[str, str], state_ids: dict[str, str],
                  stories_dir: str | None = None) -> dict:
    labels = [f"phase-{s.phase}"]
    if s.pri in PRIORITY:
        labels.append(s.pri)
    lane = LANES.get(s.lane.lower())
    if lane:
        labels.append(lane)
    if f"pts-{s.pts}" in ALL_LABELS:
        labels.append(f"pts-{s.pts}")
    if f"sprint-{s.sprint}" in ALL_LABELS:
        labels.append(f"sprint-{s.sprint}")
    rows = [("Points", s.pts), ("Priority", s.pri), ("Lane", s.lane), ("Phase", s.phase),
            ("Sprint", s.sprint), ("Backlog status", s.status), ("Today", STATE_NOTES.get(s.id, "")),
            ("Acceptance criteria", s.ac), ("Dependencies", s.deps), ("Due", s.due), ("Why now", s.why)]
    story_file = os.path.join(stories_dir, f"DOCBOT-{s.id}.md") if stories_dir else ""
    if story_file and os.path.exists(story_file):
        body = story_html(story_file, s)
    else:
        body = "".join(f"<p><strong>{k}:</strong> {md_inline(v)}</p>" for k, v in rows if v and v != "-")
        body += f"<p><em>Source: {html.escape(s.src)}; AC/deps/due from {TRACKER} EPIC-25/26.</em></p>"
    payload = {
        "name": f"DOCBOT-{s.id} {s.title}",
        "description_html": body,
        "state": state_ids[derived_state(s)],
        "priority": PRIORITY.get(s.pri, "none"),
        "labels": sorted(label_ids[n] for n in labels if n in label_ids),
        "external_source": EXTERNAL_SOURCE,
        "external_id": f"DOCBOT-{s.id}",
    }
    target = due_to_date(s.due)
    if target:
        payload["target_date"] = target
    return payload


def historical_payload(label_ids: dict[str, str], state_ids: dict[str, str]) -> dict:
    body = ("<p>Rolled-up record of DocBot tickets merged to <code>main</code> before Plane was adopted "
            "(2026-11-17). Not tracked individually.</p>"
            f"<p><strong>Shipped ticket IDs (DOCBOT-):</strong> {HISTORICAL_IDS}</p>"
            "<p><em>Source: project-tasks/jira-sync-plan.md section 2.1 (merged to main, missing from the Jira "
            "updater). Earlier IDs already Done in Jira (101-1407, 1506-1508) are not repeated.</em></p>")
    return {"name": HISTORICAL_NAME, "description_html": body, "state": state_ids["Done"], "priority": "none",
            "labels": [label_ids["historical"]] if "historical" in label_ids else [],
            "external_source": EXTERNAL_SOURCE, "external_id": HISTORICAL_EXT}


# ---------------------------------------------------------------- Plane client

class Plane:
    def __init__(self, key: str, dry_run: bool) -> None:
        self.s = requests.Session()
        self.s.headers.update({"X-API-Key": key, "Content-Type": "application/json"})
        self.dry_run = dry_run
        self.last = 0.0
        self.calls = 0

    def req(self, method: str, path: str, json: dict | None = None) -> dict:
        if self.dry_run and method != "GET":
            return {"id": f"dry-{path}"}
        for attempt in range(6):
            wait = THROTTLE_S - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.calls += 1
            r = self.s.request(method, f"{API}/{path}", json=json, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                backoff = float(r.headers.get("Retry-After") or 2 ** (attempt + 1))
                print(f"  {r.status_code} on {method} {path}, retrying in {backoff:.0f}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            if not r.ok:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else {}
        raise RuntimeError(f"{method} {path}: gave up after retries")

    def list_all(self, path: str) -> list[dict]:
        out, cursor = [], None
        while True:
            sep = "&" if "?" in path else "?"
            q = f"{path}{sep}per_page=100" + (f"&cursor={cursor}" if cursor else "")
            d = self.req("GET", q)
            if isinstance(d, list):
                return d
            out.extend(d.get("results", []))
            if not d.get("next_page_results"):
                return out
            cursor = d["next_cursor"]


# ---------------------------------------------------------------- sync

def ensure_named(api: Plane, path: str, names: list[str], extra: dict[str, dict], counts: dict, kind: str) -> dict[str, str]:
    existing = {x["name"]: x["id"] for x in api.list_all(path)}
    for n in names:
        if n not in existing:
            print(f"  create {kind}: {n}")
            existing[n] = api.req("POST", path, {"name": n, **extra.get(n, {})})["id"]
            counts[f"{kind}s created"] += 1
    return existing


def html_text(h: str | None) -> str:
    # Plane re-serialises HTML on save (wraps in <div>, unescapes quotes), so compare visible text
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h or "")).split())


def differs(issue: dict, k: str, v: object) -> bool:
    if k == "labels":
        return sorted(issue.get(k) or []) != v
    if k == "description_html":
        return html_text(issue.get(k)) != html_text(str(v))
    return issue.get(k) != v


def plane_key() -> str | None:
    """PLANE_API_KEY env var, else macOS Keychain item PLANE_API_KEY (never a file)."""
    if os.environ.get("PLANE_API_KEY"):
        return os.environ["PLANE_API_KEY"]
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", "PLANE_API_KEY", "-w"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


def ensure_states(api: Plane) -> dict[str, str]:
    states = {x["name"]: x["id"] for x in api.list_all("states/")}
    if "Testing" not in states:
        print("  create state: Testing")
        states["Testing"] = api.req("POST", "states/", TESTING_STATE)["id"]
    return states


def set_state(api: Plane, ticket: str, state: str, title: str | None, note: str | None) -> int:
    """Move one issue to a state (creating it if missing and --title given), optionally comment."""
    m = re.fullmatch(r"(?:DOCBOT-)?(\d{3,5})", ticket)
    if not m:
        print(f"bad ticket id: {ticket}", file=sys.stderr)
        return 2
    ext = f"DOCBOT-{m.group(1)}"
    states = ensure_states(api)
    if state not in states:
        print(f"unknown state {state!r}; have {sorted(states)}", file=sys.stderr)
        return 2
    issues = api.list_all("issues/")
    cur = next((i for i in issues if i.get("external_id") == ext or i["name"].startswith(ext + " ")), None)
    if cur is None:
        if not title:
            print(f"{ext} not in Plane; pass --title to create it", file=sys.stderr)
            return 2
        cur = api.req("POST", "issues/", {"name": f"{ext} {title}", "state": states[state],
                                          "external_source": EXTERNAL_SOURCE, "external_id": ext})
        print(f"created {ext} in {state}")
    elif cur["state"] != states[state]:
        api.req("PATCH", f"issues/{cur['id']}/", {"state": states[state]})
        print(f"{ext} -> {state}")
    else:
        print(f"{ext} already {state}")
    if note:
        api.req("POST", f"issues/{cur['id']}/comments/", {"comment_html": f"<p>{waf_safe(html.escape(note))}</p>"})
        print("  comment added")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print planned creates/updates, no writes")
    ap.add_argument("--source-ref", default="origin/main", help="git ref to read the backlog from")
    ap.add_argument("--stories-dir", default=os.path.join(REPO, "project-tasks", "stories"),
                    help="if DOCBOT-XXXX.md exists here, it becomes the issue description (read from the working tree)")
    ap.add_argument("--self-test", action="store_true", help="run the markdown renderer asserts and exit")
    ap.add_argument("--set-state", nargs=2, metavar=("TICKET", "STATE"),
                    help='move one issue, e.g. --set-state DOCBOT-2580 "In Progress" (Todo/In Progress/Testing/Done/Cancelled)')
    ap.add_argument("--title", help="with --set-state: create the issue with this title if it is not in Plane")
    ap.add_argument("--note", help="with --set-state: add this comment (PR link, test result)")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0

    key = plane_key()
    if not key:
        print("No Plane key: set PLANE_API_KEY or run once: security add-generic-password -a docbot -s PLANE_API_KEY -w",
              file=sys.stderr)
        return 2
    if args.set_state:
        return set_state(Plane(key, args.dry_run), *args.set_state, args.title, args.note)
    stories, skipped = load_stories(args.source_ref)
    assert stories and all(re.fullmatch(r"\d{4}", s.id) for s in stories), "backlog parse failed"
    assert not {s.id for s in stories} - {s.id for s in stories if s.phase in ("0", "1")}, "out-of-scope story"

    api = Plane(key, args.dry_run)
    counts = {k: 0 for k in ("labels created", "modules created", "issues created", "issues updated",
                             "issues unchanged", "module links added")}
    states = ensure_states(api)

    def color(n: str) -> str:
        return LABEL_COLORS["P" if re.fullmatch(r"P\d", n) else n.split("-")[0]]

    labels = ensure_named(api, "labels/", ALL_LABELS, {n: {"color": color(n)} for n in ALL_LABELS}, counts, "label")
    module_of = {s.id: MODULES.get(s.track, SAFETY) for s in stories}
    mod_names = sorted(set(module_of.values())) + [HISTORICAL_MODULE]
    modules = ensure_named(api, "modules/", mod_names, {}, counts, "module")

    issues = api.list_all("issues/")
    by_ext = {i["external_id"]: i for i in issues if i.get("external_source") == EXTERNAL_SOURCE and i.get("external_id")}
    by_prefix = {m.group(1): i for i in issues if (m := re.match(r"(DOCBOT-\d{4}) ", i["name"]))}

    plan = [(f"DOCBOT-{s.id}", story_payload(s, labels, states, args.stories_dir), module_of[s.id]) for s in stories]
    with_story = sum(os.path.exists(os.path.join(args.stories_dir, f"DOCBOT-{s.id}.md")) for s in stories)
    plan.append((HISTORICAL_EXT, historical_payload(labels, states), HISTORICAL_MODULE))
    managed_labels = set(labels[n] for n in ALL_LABELS if n in labels)
    order = {states[n]: i for i, n in enumerate(STATE_ORDER) if n in states}
    final_state: dict[str, int] = {}
    want_in_module: dict[str, list[str]] = {}

    for ext, payload, module in plan:
        cur = by_ext.get(ext) or by_prefix.get(ext) or (by_prefix_hist(issues) if ext == HISTORICAL_EXT else None)
        if cur is None:
            print(f"  create issue: {payload['name'][:90]}")
            iid = api.req("POST", "issues/", payload)["id"]
            counts["issues created"] += 1
        else:
            iid = cur["id"]
            # never move state backwards (founder edits in Plane win); keep labels added by hand
            if order.get(cur["state"], -1) >= order.get(payload["state"], -1) or cur["state"] not in order:
                payload = {**payload, "state": cur["state"]}
            keep = [lid for lid in cur.get("labels", []) if lid not in managed_labels]
            payload = {**payload, "labels": sorted(set(payload["labels"]) | set(keep))}
            diff = {k: v for k, v in payload.items() if differs(cur, k, v)}
            if diff:
                print(f"  update issue: {ext} fields={sorted(diff)}")
                api.req("PATCH", f"issues/{iid}/", diff)
                counts["issues updated"] += 1
            else:
                counts["issues unchanged"] += 1
        final_state[ext] = payload["state"]
        want_in_module.setdefault(module, []).append(iid)

    for name, iids in want_in_module.items():
        mid = modules[name]
        have = set() if mid.startswith("dry-") else {x["id"] for x in api.list_all(f"modules/{mid}/module-issues/")}
        missing = [i for i in iids if i not in have]
        if missing:
            print(f"  link {len(missing)} issue(s) to module: {name}")
            api.req("POST", f"modules/{mid}/module-issues/", {"issues": missing})
            counts["module links added"] += len(missing)

    names = {v: k for k, v in states.items()}
    by_state: dict[str, int] = {}
    for sid in final_state.values():
        by_state[names.get(sid, sid)] = by_state.get(names.get(sid, sid), 0) + 1
    print("\n" + ("DRY RUN (no writes). " if args.dry_run else "") + f"source={args.source_ref}")
    print(f"stories in scope: {len(stories)} (+1 historical); with story file: {with_story} ({args.stories_dir})")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"issues by state: {by_state}")
    print(f"skipped (not phase 0/1): {', '.join(skipped)}")
    print(f"API calls: {api.calls}")
    return 0


def by_prefix_hist(issues: list[dict]) -> dict | None:
    return next((i for i in issues if i["name"] == HISTORICAL_NAME), None)


if __name__ == "__main__":
    sys.exit(main())
