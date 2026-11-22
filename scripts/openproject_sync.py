#!/usr/bin/env python3
"""Idempotent sync of every DocBot ticket in the repo docs (current backlog + old epics + parking lot) into self-hosted OpenProject (pm.docbot.space).

Hierarchy: Epic -> Feature -> User story. Backlog sprints S1..S7 map to OpenProject sprints (created
once on the server, see scripts/README-openproject.md), so the Backlogs burndown works. Backlog parsing
is shared with plane_sync.py (same source files, same derived states).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import requests

import plane_sync as ps  # shortcut: parser lives in plane_sync.py (validate_stories.py uses it too); move to backlog_parse.py when Plane is deleted

BASE = os.environ.get("OPENPROJECT_URL", "https://pm.docbot.space").rstrip("/")
PROJECT = "docbot"

# Plane-era state names (used by CLAUDE.md, the workflow and the parser) -> OpenProject statuses
STATUS_MAP = {"Backlog": "New", "Todo": "New", "In Progress": "In progress", "Testing": "In testing",
              "Done": "Closed", "Cancelled": "Rejected"}
STATUS_ORDER = ["New", "In progress", "In testing", "Closed"]
PRIORITY = {"P0": "Immediate", "P1": "High", "P2": "Normal", "P3": "Low"}
EPICS = {  # feature (Plane module) name prefix -> epic subject
    "EPIC-26": "EPIC-26 Pre-seed readiness",
    "EPIC-25": "EPIC-25 $5K MRR + 6 paying logos by 2027-02-28",
    "EPIC-24": "EPIC-24 Stage A isolation",
    "EPIC-18": "EPIC-18 Billing",
}
SAFETY_EPIC = "Phase 1 safety & robustness (EPIC-16 rest, EPIC-23 bugs, scheduled parking lot)"
HISTORICAL_EPIC = "Historical work"


def epic_of(feature: str) -> str:
    if feature == ps.HISTORICAL_MODULE:
        return HISTORICAL_EPIC
    m = re.match(r"EPIC-\d+", feature)
    return EPICS.get(m.group(0), SAFETY_EPIC) if m else SAFETY_EPIC


def op_status(state: str) -> str:
    return STATUS_MAP.get(state, state)


def ahead(cur: str, want: str) -> bool:
    """True if the board is at or past `want`. Never move backwards; Rejected / On hold etc. are kept."""
    if cur not in STATUS_ORDER:
        return True
    return want in STATUS_ORDER and STATUS_ORDER.index(cur) >= STATUS_ORDER.index(want)


def story_markdown(s: ps.Story, stories_dir: str | None) -> str:
    note = ps.STATE_NOTES.get(s.id)
    head = f"**Today:** {note}\n\n" if note else ""
    path = os.path.join(stories_dir, f"DOCBOT-{s.id}.md") if stories_dir else ""
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        first, _, rest = text.partition("\n")
        body = rest.strip() if first.startswith(f"# DOCBOT-{s.id}") else text.strip()
        return f"{head}{body}\n\n_Source: {os.path.relpath(path, ps.REPO)}_"
    rows = [("Points", s.pts), ("Priority", s.pri), ("Lane", s.lane), ("Phase", s.phase), ("Sprint", s.sprint),
            ("Backlog status", s.status), ("Acceptance criteria", s.ac), ("Dependencies", s.deps), ("Due", s.due),
            ("Why now", s.why)]
    body = "\n\n".join(f"**{k}:** {v}" for k, v in rows if v and v != "-")
    return f"{head}{body}\n\n_Source: {s.src}_"


def story_spec(s: ps.Story, stories_dir: str | None) -> dict:
    """Wanted values by name; the client resolves names to ids."""
    return {"subject": f"DOCBOT-{s.id} {s.title}", "type": "User story", "description": story_markdown(s, stories_dir),
            "status": op_status(ps.derived_state(s)), "priority": PRIORITY.get(s.pri, "Normal"),
            "storyPoints": int(s.pts) if s.pts.isdigit() else None,
            "sprint": s.sprint if re.fullmatch(r"S\d", s.sprint) else None}


def historical_spec() -> dict:
    return {"subject": ps.HISTORICAL_NAME, "type": "User story", "status": "Closed", "priority": "Normal",
            "storyPoints": None, "sprint": None,
            "description": f"Rolled-up record of DocBot tickets merged to `main` before 2026-11-17. Not tracked "
                           f"individually.\n\n**Shipped ticket IDs (DOCBOT-):** {ps.HISTORICAL_IDS}"}


# ---------------------------------------------------------------- archive: every other ticket in the repo docs

YC_PLAN = "project-tasks/yc-10-day-plan.md"
ROADMAP = "project-tasks/docbot-enterprise-roadmap.md"
OLD_PRIORITY = {"must have": "High", "should have": "Normal", "nice to have": "Low", "could have": "Low"}
UNKNOWN_NOTE = ("**Status not recorded in the repo docs** (not in the shipped list, not in the current backlog). "
                "Parked as On hold: triage to Closed, Rejected or New.")


def shipped_ids() -> set[str]:
    out = set()
    for part in ps.HISTORICAL_IDS.split(","):
        a, _, b = part.strip().partition("-")
        out.update(str(n) for n in range(int(a), int(b or a) + 1))
    return out


def epic_code(tid: str) -> str:
    """DOCBOT-101 -> EPIC-01, DOCBOT-1503 -> EPIC-15, DOCBOT-2704 -> EPIC-27 (the repo numbers tickets by epic)."""
    if tid == "SCRUM-391":  # Jira-era id, written up under EPIC-04
        return "EPIC-04"
    n = re.sub(r"\D", "", tid)
    return f"EPIC-{int(n[:-2]):02d}"


def epic_names(docs: list[str]) -> dict[str, str]:
    """EPIC-NN -> subject, from the first heading that names it ('### 7.1 EPIC-27: SOC 2 ... (18 stories)')."""
    names = dict(EPICS)
    for text in docs:
        for m in re.finditer(r"^#{2,4}\s+(?:[\d.]+\s+)?(?:New\s+)?(EPIC-\d+)\s*[:—–-]\s*(.+)$", text, re.M):
            code, name = m.group(1), re.sub(r"\s*\([^)]*\)\s*$", "", m.group(2)).strip()
            names.setdefault(code, f"{code} {name}")
    return names


def archive_spec(tid: str, title: str, state: str, pts: str, pri: str, desc: str) -> dict:
    status = "On hold" if state == "?" else op_status(state)
    return {"subject": f"{tid} {title}", "type": "User story", "description": desc, "status": status,
            "priority": PRIORITY.get(pri) or OLD_PRIORITY.get(pri.lower(), "Normal"),
            "storyPoints": int(pts) if pts.isdigit() else None, "sprint": None}


def tracker_sections(text: str, ref: str, shipped: set[str]) -> list[tuple[str, dict]]:
    """Full write-ups in docbot-v2-project-tracking.md section 3 (EPIC-01..15)."""
    body = text[text.index("## 3. User Stories"):text.index("## 4. Sprint Plan")]
    out = []
    for m in re.finditer(r"^#### ((?:DOCBOT|SCRUM)-\d+): (.+?)\n(.*?)(?=^#{2,4} |\Z)", body, re.M | re.S):
        tid, title, sec = m.group(1), re.sub(r"\s*✅.*$", "", m.group(2)).strip(), m.group(3)
        field = lambda k: (re.search(rf"\*\*{k}\*\*:\s*(.+)", sec) or [None, ""])[1].strip()
        status = field("Status").lower()
        # no Status line: EPIC-01..15 are all recorded as shipped in CLAUDE.md "Ticket Reference"
        state = ("Done" if not status or "done" in status or tid.removeprefix("DOCBOT-") in shipped
                 else "In Progress" if "progress" in status else "Backlog")
        desc = re.sub(r"\n-{3,}\s*$", "", sec.strip()) + f"\n\n_Source: {ps.TRACKER} @ {ref}_"
        out.append((tid, archive_spec(tid, title, state, field("Story Points"), field("Priority"), desc)))
    return out


def table_rows(text: str, src: str, ref: str, shipped: set[str]) -> list[tuple[str, dict]]:
    out = []
    for header, rows in ps.tables(text):
        if header[0] not in ("id", "ticket", "story"):
            continue
        for r in rows:
            d = dict(zip(header, r))
            m = re.fullmatch(r"DOCBOT-(\d{3,4})", d[header[0]])
            if not m:
                continue
            sid = m.group(1)
            title = d.get("title") or d.get("group (plat range)") or d.get("candidate story") or ""
            status, sprint = d.get("status", ""), d.get("sprint", "")
            if sid in shipped:
                state = "Done"
            elif status:
                state = ps.derived_state(ps.Story(sid, title, "", "", status, d.get("planned phase", "2"),
                                                  "", "", "", ""))
            elif sprint == "parked" or src == ps.BACKLOG:
                state = "Backlog"
            else:
                state = "?"
            skip = {header[0], "title", "pts", "points", "pri"}
            rows_md = [f"**{k.capitalize()}:** {v}" for k, v in d.items() if k not in skip and v and v != "-"]
            desc = "\n\n".join(([UNKNOWN_NOTE] if state == "?" else []) + rows_md + [f"_Source: {src} @ {ref}_"])
            out.append((f"DOCBOT-{sid}", archive_spec(f"DOCBOT-{sid}", title, state, d.get("pts") or d.get("points", ""),
                                                      d.get("pri", ""), desc)))
    return out


def load_archive(ref: str, known: set[str]) -> list[tuple[str, str, dict]]:
    """(key, epic subject, spec) for every ticket in the repo docs that the current backlog does not cover.
    First source wins: full tracker write-ups, then backlog parking lot / referenced table, YC plan, EPIC-25/26."""
    tracker, backlog, yc = (ps.git_show(ref, p) for p in (ps.TRACKER, ps.BACKLOG, YC_PLAN))
    names = epic_names([tracker, yc, backlog, ps.git_show(ref, ROADMAP)])
    shipped = shipped_ids()
    found = (tracker_sections(tracker, ref, shipped)
             + table_rows(ps.section(backlog, "## 7.", "## 9."), ps.BACKLOG, ref, shipped)
             + table_rows(yc, YC_PLAN, ref, shipped)
             + table_rows(tracker[tracker.index("## EPIC-25"):], ps.TRACKER, ref, shipped))
    n_sections = len(tracker_sections(tracker, ref, shipped))
    out, seen = [], set(known)
    for i, (tid, spec) in enumerate(found):
        key = tid if tid.startswith("DOCBOT-") else spec["subject"]  # index() keys only DOCBOT ids
        if tid in seen:
            if i >= n_sections or tid in known:
                continue  # covered by the current backlog or an earlier source
            # two different write-ups share an id (DOCBOT-701): keep both, the second keyed by its subject
            spec = {**spec, "subject": spec["subject"].replace(tid, tid + "-2", 1)}
            key = spec["subject"]
        seen.add(tid)
        code = epic_code(tid)
        out.append((key, names.get(code, code), spec))
    return out


def norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _self_test() -> None:
    assert op_status("Testing") == "In testing" and op_status("Todo") == "New" and op_status("Done") == "Closed"
    assert ahead("Closed", "In progress") and ahead("Rejected", "Closed") and ahead("On hold", "New")
    assert not ahead("New", "In progress")
    assert epic_of("EPIC-25 AWS migration") == EPICS["EPIC-25"] and epic_of(ps.SAFETY) == SAFETY_EPIC
    assert epic_of(ps.HISTORICAL_MODULE) == HISTORICAL_EPIC
    s = ps.Story("9999", "Terraform", "5", "P1", "Open", "1", "S2", "devops", "EPIC-25 AWS", "test")
    sp = story_spec(s, None)
    assert sp["sprint"] == "S2" and sp["storyPoints"] == 5 and sp["priority"] == "High" and sp["status"] == "New"
    assert "**Lane:** devops" in sp["description"]
    sp2 = story_spec(ps.Story("9998", "FAQ", "", "-", "To do", "1", "-", "x", "EPIC-25 SEO", "t"), None)
    assert sp2["sprint"] is None and sp2["storyPoints"] is None and sp2["priority"] == "Normal"
    assert epic_code("DOCBOT-101") == "EPIC-01" and epic_code("DOCBOT-2704") == "EPIC-27"
    assert "1520" in shipped_ids() and "1613" not in shipped_ids()
    assert archive_spec("DOCBOT-1", "x", "?", "3", "Must Have", "")["status"] == "On hold"
    assert archive_spec("DOCBOT-1", "x", "Done", "", "Must Have", "")["priority"] == "High"
    print("self-test ok")


# ---------------------------------------------------------------- client

def api_key() -> str | None:
    """OPENPROJECT_API_KEY env var, else macOS Keychain item of the same name (never a file)."""
    if os.environ.get("OPENPROJECT_API_KEY"):
        return os.environ["OPENPROJECT_API_KEY"].strip()
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", "OPENPROJECT_API_KEY", "-w"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() or None


def link_id(item: dict, rel: str) -> int | None:
    href = (item["_links"].get(rel) or {}).get("href")
    return int(href.rsplit("/", 1)[1]) if href else None


class Op:
    def __init__(self, key: str, dry_run: bool) -> None:
        self.s = requests.Session()
        self.s.auth = ("apikey", key)
        self.s.headers["Content-Type"] = "application/json"
        self.dry_run = dry_run
        self.calls = 0
        self._fake = 0

    def req(self, method: str, path: str, body: dict | None = None) -> dict:
        if self.dry_run and method != "GET":
            self._fake -= 1
            return {"id": self._fake, "lockVersion": 0, "_links": {}}
        for attempt in range(6):
            self.calls += 1
            r = self.s.request(method, f"{BASE}/api/v3{path}", json=body, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                backoff = float(r.headers.get("Retry-After") or 2 ** (attempt + 1))
                print(f"  {r.status_code} on {method} {path}, retrying in {backoff:.0f}s", file=sys.stderr)
                time.sleep(backoff)
                continue
            if r.status_code == 401:
                raise RuntimeError("OpenProject rejected the API key (OPENPROJECT_API_KEY)")
            if not r.ok:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else {}
        raise RuntimeError(f"{method} {path}: gave up after retries")

    def names(self, path: str) -> dict[str, int]:
        return {e["name"]: e["id"] for e in self.req("GET", path)["_embedded"]["elements"]}

    def load_refs(self) -> None:
        self.project_id = self.req("GET", f"/projects/{PROJECT}")["id"]
        self.types = self.names(f"/projects/{PROJECT}/types")
        self.statuses = self.names("/statuses")
        self.priorities = self.names("/priorities")
        self.sprints = self.names(f"/projects/{PROJECT}/sprints?pageSize=100")
        missing = {"Epic", "Feature", "User story"} - set(self.types)
        assert not missing, f"enable types {missing} on project {PROJECT} (see README-openproject.md)"

    def work_packages(self) -> list[dict]:
        out, offset = [], 1
        while True:  # filters=[] so closed items are listed too
            page = self.req("GET", f"/projects/{PROJECT}/work_packages?filters=%5B%5D&pageSize=500&offset={offset}")
            out += page["_embedded"]["elements"]
            if len(out) >= page["total"]:
                return out
            offset += 1

    def body(self, spec: dict, parent_id: int | None = None) -> dict:
        b: dict = {"_links": {}}
        for k in ("subject", "storyPoints"):
            if k in spec:
                b[k] = spec[k]
        if "description" in spec:
            b["description"] = {"format": "markdown", "raw": spec["description"]}
        for rel, coll, table in (("type", "types", self.types), ("status", "statuses", self.statuses),
                                 ("priority", "priorities", self.priorities)):
            if rel in spec:
                b["_links"][rel] = {"href": f"/api/v3/{coll}/{table[spec[rel]]}"}
        if "sprint" in spec:
            sid = self.sprints.get(spec["sprint"]) if spec["sprint"] else None
            b["_links"]["sprint"] = {"href": f"/api/v3/sprints/{sid}" if sid else None}
        if parent_id:
            b["_links"]["parent"] = {"href": f"/api/v3/work_packages/{parent_id}"}
        return b

    def create(self, spec: dict, parent_id: int | None) -> dict:
        return self.req("POST", f"/projects/{PROJECT}/work_packages", self.body(spec, parent_id))

    def update(self, item: dict, spec: dict, parent_id: int | None = None) -> None:
        self.req("PATCH", f"/work_packages/{item['id']}", {**self.body(spec, parent_id), "lockVersion": item["lockVersion"]})

    def comment(self, wid: int, text: str) -> None:
        self.req("POST", f"/work_packages/{wid}/activities", {"comment": {"raw": text}})

    def current(self, item: dict) -> dict:
        """Same keys as a spec, read back from an existing work package."""
        L = item["_links"]
        return {"subject": item["subject"], "type": L["type"]["title"], "status": L["status"]["title"],
                "priority": L["priority"]["title"], "storyPoints": item.get("storyPoints"),
                "sprint": (L.get("sprint") or {}).get("title"),
                "description": (item.get("description") or {}).get("raw", "")}


def index(items: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    """(DOCBOT-XXXX -> user story, (type, subject) -> any item)."""
    stories, named = {}, {}
    for it in items:
        t, kind = it["subject"], it["_links"]["type"]["title"]
        m = re.match(r"(DOCBOT-\d{3,5}) ", t)
        if m and kind == "User story":
            stories[m.group(1)] = it
        named[(kind, t)] = it
    return stories, named


# ---------------------------------------------------------------- commands

def set_state(api: Op, ticket: str, state: str, title: str | None, note: str | None) -> int:
    m = re.fullmatch(r"(?:DOCBOT-)?(\d{3,5})", ticket)
    want = op_status(state)
    if not m or want not in STATUS_ORDER + ["Rejected"]:
        print(f"bad ticket or state: {ticket} {state} (states: {', '.join(STATUS_MAP)})", file=sys.stderr)
        return 2
    ext = f"DOCBOT-{m.group(1)}"
    api.load_refs()
    stories, _ = index(api.work_packages())
    cur = stories.get(ext)
    if cur is None:
        if not title:
            print(f"{ext} not on the board; pass --title to create it", file=sys.stderr)
            return 2
        cur = api.create({"subject": f"{ext} {title}", "type": "User story", "status": want}, None)
        print(f"created {ext} in {want}")
    elif cur["_links"]["status"]["title"] != want:  # explicit moves may go backwards (reopened)
        api.update(cur, {"status": want})
        print(f"{ext} -> {want}")
    else:
        print(f"{ext} already {want}")
    if note:
        api.comment(cur["id"], note)
        print("  comment added")
    return 0


def full_sync(api: Op, args: argparse.Namespace) -> int:
    stories, skipped = ps.load_stories(args.source_ref)
    assert stories and all(re.fullmatch(r"\d{4}", s.id) for s in stories), "backlog parse failed"
    api.load_refs()
    counts = dict.fromkeys(("epics created", "features created", "stories created", "stories updated",
                            "stories unchanged"), 0)
    have, named = index(api.work_packages())

    def ensure(kind: str, subject: str, parent: dict | None, status: str) -> dict:
        if (kind, subject) not in named:
            print(f"  create {kind}: {subject}")
            named[(kind, subject)] = api.create({"subject": subject, "type": kind, "status": status},
                                                parent and parent["id"])
            counts[f"{kind.lower()}s created"] += 1
        return named[(kind, subject)]

    feature_of = {s.id: ps.MODULES.get(s.track, ps.SAFETY) for s in stories}
    plan = [(f"DOCBOT-{s.id}", story_spec(s, args.stories_dir), epic_of(feature_of[s.id]), feature_of[s.id])
            for s in stories]
    plan.append((ps.HISTORICAL_NAME, historical_spec(), HISTORICAL_EPIC, ps.HISTORICAL_MODULE))
    # every other ticket in the repo docs (old epics, parking lot) sits directly under its epic
    plan += [(key, spec, epic, None) for key, epic, spec in load_archive(args.source_ref, {k for k, *_ in plan})]
    by_status: dict[str, int] = {}
    for key, spec, epic_subject, feature in plan:
        done = "Closed" if feature == ps.HISTORICAL_MODULE else "New"
        epic = ensure("Epic", epic_subject, None, done)
        feat = ensure("Feature", feature, epic, done) if feature else epic
        cur = have.get(key) or named.get(("User story", key))
        if spec["sprint"] and spec["sprint"] not in api.sprints:
            print(f"  warning: sprint {spec['sprint']} missing on server; {key} left unscheduled", file=sys.stderr)
            spec = {**spec, "sprint": None}
        if cur is None:
            print(f"  create story: {spec['subject'][:90]}")
            api.create(spec, feat["id"])
            counts["stories created"] += 1
        else:
            now = api.current(cur)
            if ahead(now["status"], spec["status"]):
                spec = {**spec, "status": now["status"]}
            diff = {k: v for k, v in spec.items()
                    if (norm(now[k]) != norm(v) if k == "description" else now[k] != v)}
            parent = feat["id"] if link_id(cur, "parent") is None else None
            if diff or parent:
                print(f"  update story: {key} fields={sorted(diff)}{' +parent' if parent else ''}")
                api.update(cur, diff, parent)
                counts["stories updated"] += 1
            else:
                counts["stories unchanged"] += 1
        by_status[spec["status"]] = by_status.get(spec["status"], 0) + 1

    print("\n" + ("DRY RUN (no writes). " if args.dry_run else "") + f"source={args.source_ref} board={BASE}")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"stories by status: {json.dumps(by_status)}")
    print(f"not in current sprints (loaded under their epic instead): {', '.join(skipped)}")
    print(f"API calls: {api.calls}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="read the board, print planned writes, write nothing")
    ap.add_argument("--source-ref", default="origin/main", help="git ref to read the backlog from")
    ap.add_argument("--stories-dir", default=os.path.join(ps.REPO, "project-tasks", "stories"))
    ap.add_argument("--self-test", action="store_true", help="run mapping asserts and exit (no key needed)")
    ap.add_argument("--set-state", nargs=2, metavar=("TICKET", "STATE"),
                    help='move one story, e.g. --set-state DOCBOT-2586 "In Progress" (Todo/In Progress/Testing/Done/Cancelled)')
    ap.add_argument("--title", help="with --set-state: create the story with this title if missing")
    ap.add_argument("--note", help="with --set-state: add this comment (PR link, test result)")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0
    key = api_key()
    if not key:
        print("No OpenProject key: set OPENPROJECT_API_KEY or run once:\n"
              "  security add-generic-password -a docbot -s OPENPROJECT_API_KEY -w '<token>'", file=sys.stderr)
        return 2
    api = Op(key, args.dry_run)
    if args.set_state:
        return set_state(api, *args.set_state, args.title, args.note)
    return full_sync(api, args)


if __name__ == "__main__":
    sys.exit(main())
