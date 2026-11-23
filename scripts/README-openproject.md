# openproject_sync.py

Loads every DocBot ticket in the repo docs into self-hosted OpenProject at https://pm.docbot.space (project `docbot`) and keeps it in sync. Safe to re-run.

```bash
# key: OPENPROJECT_API_KEY env var, or once per machine (macOS Keychain, never a file).
# Token: board -> My account -> Access tokens -> API.
security add-generic-password -a docbot -s OPENPROJECT_API_KEY -w '<token>'
git fetch origin
python3 scripts/openproject_sync.py --self-test     # mapping asserts, no key needed
python3 scripts/openproject_sync.py --dry-run       # planned creates/updates, no writes
python3 scripts/openproject_sync.py                 # apply (reads origin/main)
```

## Per-ticket state (see the OpenProject Sync Rule in CLAUDE.md)
```bash
python3 scripts/openproject_sync.py --set-state DOCBOT-2586 "In Progress" --title "Short title"   # creates if missing
python3 scripts/openproject_sync.py --set-state DOCBOT-2586 Testing --note "PR #140"
python3 scripts/openproject_sync.py --set-state DOCBOT-2586 Done --note "PR #140 merged"
```
Todo/Backlog -> New, In Progress -> In progress, Testing -> In testing, Done -> Closed, Cancelled -> Rejected.

## What it loads (from git, not the working tree)
- Current backlog (same parser as `plane_sync.py`): `docbot-master-backlog.md` section 8 phase 0/1 + section 7 rows in a sprint, enriched from the EPIC-25/26 tables in `docbot-v2-project-tracking.md`. Hierarchy Epic -> Feature (the old Plane module) -> User story, with sprint, points, priority. Description = `project-tasks/stories/DOCBOT-XXXX.md` if present.
- Everything else, directly under its epic (epic = ticket number prefix: DOCBOT-101 -> EPIC-01, DOCBOT-2704 -> EPIC-27):
  - EPIC-01..15 full write-ups in `docbot-v2-project-tracking.md` section 3 (Closed unless the write-up says otherwise).
  - EPIC-16..23 tables in `yc-10-day-plan.md`.
  - EPIC-27..40 parking lot and later-phase rows in `docbot-master-backlog.md` (New, no sprint).
  - Shipped = `HISTORICAL_IDS` in `plane_sync.py`. A ticket with no recorded status is set to **On hold** with a triage note.

## Re-run rules
- Matches stories on the `DOCBOT-XXXX ` subject prefix (other ids on the full subject). Updates only changed fields.
- Status never moves backwards; Rejected / On hold set by hand are kept. `--set-state` can move backwards.
- A sprint set on the board is kept when the docs give the story no sprint.
- Missing sprint on the server: the story is left unscheduled with a warning.

## Server (AWS Lightsail, us-east-1, 2 GB / $12 month, paid by AWS credits)
- Static IP `docbot-pm-ip` (13.217.175.83); DNS `pm` A record in Vercel (docbot.space). Firewall: 22, 443 (80 for cert renewal).
- SSH: `ssh -i ~/.ssh/docbot-pm.pem ubuntu@13.217.175.83` (key stays in `~/.ssh`, never committed).
- `/opt/openproject/docker-compose.yml`: `openproject/openproject:17` all-in-one (bundled Postgres) on 127.0.0.1:8080 + `caddy:2` for HTTPS. `.env` holds `SECRET_KEY_BASE` (root only, never print it). 3 GB swapfile.
- Restart: `cd /opt/openproject && sudo docker compose up -d`. Logs: `sudo docker compose logs -f openproject`.
- Upgrade: bump the image tag, `sudo docker compose pull && sudo docker compose up -d` (take a snapshot first).
- Backups: Lightsail automatic snapshots, daily 1:00 AM EST, last 7 kept (instance > Snapshots tab).
- Memory is close to 2 GB. If pages take over 5 s or the container restarts, snapshot and resize to the 4 GB plan.

### One-time setup done with `rails runner`
The sprint API is read-only and project types are not settable over the API, so these ran on the server:
```bash
sudo docker exec -w /app openproject-openproject-1 bundle exec rails runner /tmp/x.rb
```
- Enable types: `p.project_types.find_or_create_by!(type_id: Type.find_by!(name: "Epic").id)` (also Feature, User story).
- Sprints: `Sprint.create!(project: p, name: "S1", start_date: ..., finish_date: ..., status: "in_planning")` for S0..S7 (dates from backlog section 5). Mark a sprint `active` / `completed` on the Backlogs page to get burndown and velocity.

### Views and boards (made with `rails runner` on 2026-11-22)
- Public views in Work packages sidebar: `1. Epics: progress`, `2. Roadmap: epic > stories`, `3. Done`, `4. Remaining`, `5. Current sprint`. The default list hides Closed / Rejected; use these.
- Boards: `Status board` (all user stories by status) and `DocBot: S0` (sprint task board).
- Start a sprint from Backlogs > Start sprint (or `Backlogs::Sprints::StartService`), not by setting `status` directly; the service builds the sprint task board.
- Progress is status-based (`work_package_done_ratio = status`), epics roll up as a simple average of their children.
