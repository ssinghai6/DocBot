#!/usr/bin/env python3
"""Check project-tasks/stories/DOCBOT-XXXX.md against the story template and the backlog.

Usage: python3 scripts/validate_stories.py [--source-ref REF] [--stories-dir DIR] [--only 2501,2502]
Exit 1 if any in-scope story is missing or fails a check.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plane_sync as ps  # noqa: E402

HEADINGS = ["## User story", "## Why now", "## Scope", "## Acceptance criteria", "## Deliverable and approach",
            "## Test / verification", "## Dependencies", "## Definition of done", "## Open questions"]
HEADER_COLS = ["Epic", "Phase", "Sprint", "Priority", "Points", "Lane", "Due", "Status"]
SECRET_RE = re.compile(r"gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_-]{20,}|sk_live_[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_-]{20,}"
                       r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|plane_api_[A-Za-z0-9]{10,}"
                       r"|whsec_[A-Za-z0-9]{10,}|postgres(?:ql)?://[^\s:@/]+:[^\s@/]+@")


def norm(v: str) -> str:
    v = v.replace("**", "").strip()
    return "-" if v in ("", "-") else v.lower()


def expected(s: ps.Story) -> dict[str, str]:
    return {"Epic": s.track, "Phase": s.phase, "Sprint": s.sprint, "Priority": s.pri, "Points": s.pts,
            "Lane": s.lane, "Due": s.due, "Status": s.status}


def check(s: ps.Story, text: str) -> list[str]:
    errs = []
    if not text.startswith(f"# DOCBOT-{s.id} "):
        errs.append("first line is not '# DOCBOT-XXXX <title>'")
    pos = [text.find("\n" + h + "\n") for h in HEADINGS]
    for h, p in zip(HEADINGS, pos):
        if p < 0:
            errs.append(f"missing heading '{h}'")
    if all(p >= 0 for p in pos) and pos != sorted(pos):
        errs.append("headings out of template order")
    hdr = next(((h, r) for h, r in ps.tables(text) if [c.lower() for c in HEADER_COLS] == h), None)
    if not hdr or len(hdr[1]) != 1:
        errs.append("header table missing or not exactly one row")
    else:
        got = dict(zip(HEADER_COLS, hdr[1][0]))
        for k, want in expected(s).items():
            if k == "Epic" and s.track == "SAFETY":
                continue  # parking-lot rows: the story names its own EPIC-27..40 epic
            if norm(got.get(k, "")) != norm(want):
                errs.append(f"header {k}={got.get(k)!r}, backlog={want or '-'!r}")
    m = re.search(r"\n## Acceptance criteria\n(.*?)\n## ", text, re.S)
    n_ac = len(re.findall(r"^- \[ \] ", m.group(1), re.M)) if m else 0
    if not 3 <= n_ac <= 7:
        errs.append(f"{n_ac} acceptance criteria (want 3-7 '- [ ] ' lines)")
    if m and not re.search(r"given .*?when .*?then ", m.group(1), re.I | re.S):
        errs.append("acceptance criteria not in Given/When/Then form")
    if "—" in text:
        errs.append(f"{text.count(chr(0x2014))} em dash(es)")
    if SECRET_RE.search(text):
        errs.append(f"secret-looking string: {SECRET_RE.search(text).group(0)[:12]}...")
    if not re.search(r"^As (?:an?|the) .+?, I want .+?, so that .+", text, re.M):
        errs.append("user story not in 'As a ..., I want ..., so that ...' form")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-ref", default="origin/docs/DOCBOT-2601-plan-trim")
    ap.add_argument("--stories-dir", default=os.path.join(ps.REPO, "project-tasks", "stories"))
    ap.add_argument("--only", default="", help="comma-separated ids")
    args = ap.parse_args()
    stories, _ = ps.load_stories(args.source_ref)
    only = set(filter(None, args.only.split(",")))
    bad = missing = 0
    for s in stories:
        if only and s.id not in only:
            continue
        path = os.path.join(args.stories_dir, f"DOCBOT-{s.id}.md")
        if not os.path.exists(path):
            print(f"MISSING DOCBOT-{s.id}")
            missing += 1
            continue
        errs = check(s, open(path, encoding="utf-8").read())
        if errs:
            bad += 1
            print(f"FAIL DOCBOT-{s.id}: " + "; ".join(errs))
    n = len(only) if only else len(stories)
    print(f"\n{n} in scope, {missing} missing, {bad} failing")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
