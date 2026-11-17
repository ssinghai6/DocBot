# plane_sync.py

Loads DocBot's Phase 0 + Phase 1 backlog into Plane (project DocBot, `DOCBO`) and keeps it in sync. Safe to re-run.

```bash
# key: PLANE_API_KEY env var, or once per machine (stored in macOS Keychain, never in a file):
security add-generic-password -a docbot -s PLANE_API_KEY -w '<token>'
git fetch origin
python3 scripts/plane_sync.py --dry-run                         # print planned creates/updates, no writes
python3 scripts/plane_sync.py                                   # apply (reads origin/main)
```

## Per-ticket state (see the Plane Sync Rule in CLAUDE.md)
```bash
python3 scripts/plane_sync.py --set-state DOCBOT-2580 "In Progress" --title "Short title"   # creates if missing
python3 scripts/plane_sync.py --set-state DOCBOT-2580 Testing --note "PR #124"             # PR open / CI / founder test
python3 scripts/plane_sync.py --set-state DOCBOT-2580 Done --note "PR #124 merged"
```
States: Backlog, Todo, In Progress, Testing (created by the script if missing), Done, Cancelled.

## What it reads (from git, not the working tree)
- `project-tasks/docbot-master-backlog.md` section 8: every ticket with planned phase 0 or 1; section 7: parking-lot rows scheduled in a sprint (2812, 3501, 3510, 3512, 3513).
- `project-tasks/docbot-v2-project-tracking.md` EPIC-25 / EPIC-26 tables: AC, deps, due dates.

## What it writes
- Labels (if missing): `phase-0/1`, `P0..P3`, `lane-*`, `pts-1/2/3/5/8`, `sprint-S1..S7`, `historical`.
- Modules (if missing, matched by name): one per epic/track, plus `Historical work`.
- One issue per story: `DOCBOT-XXXX <title>`, `external_source=docbot-backlog`, `external_id=DOCBOT-XXXX`, priority P0..P3 -> urgent/high/medium/low, `target_date` from the due date.
- One rolled-up issue for work shipped before 2026-11-17.

## Re-run rules
- Matches on `external_id`, then on the `DOCBOT-XXXX ` name prefix. Updates only changed fields.
- State never moves backwards: if you move an issue forward in Plane, the sync keeps it.
- Labels you add by hand are kept; the managed labels above are recomputed.
- `STATE_FLOOR` in the script holds facts the backlog text did not have on 2026-11-17 (PRs merged to `dev`/`main`). Remove entries once the backlog says the same.
- Throttled to ~1 request/1.1 s with retry on 429/5xx (Plane cloud limit is about 60/min). A full first run takes about 3 minutes.

## Story detail (2026-11-17)
- If `project-tasks/stories/DOCBOT-XXXX.md` exists (working tree, `--stories-dir` to override), it is rendered to HTML (stdlib renderer `md_to_html`: headings, tables, bullet/checkbox lists, code fences, paragraphs) and becomes the issue description. Otherwise the old table-derived text is used.
- The `python -m` WAF workaround (`" -"` -> `"&nbsp;-"`) applies to story text and code fences too.
- `python3 scripts/plane_sync.py --self-test` runs the renderer asserts (no key needed).
- `python3 scripts/validate_stories.py [--only 2501,2502]` checks every in-scope story file: template headings in order, header row equals the backlog (Epic, Phase, Sprint, Priority, Points, Lane, Due, Status; empty = `-`), 3-7 Given/When/Then AC, no em dashes, no secret-looking strings.
