# Wiki QA pass (2026-11-23)

Scope: `architecture/01..08`, `archify-input.md`, `api/01..04`, checked against each other, against `_manifest.json`, and against code (`api/`). Minimal in-place edits only.

## Fixes

| File | What | Why |
|---|---|---|
| `api/01-current-state.md` | "7 routes under `/v1/*`" -> "6 routes" | Code has 6 `/v1` handlers (`api/index.py:3659-3963`); the page's own table lists 6; architecture-04 lists 6 |
| `architecture/04-endpoint-reference.md` | Route split "60 in `api/index.py`, 22 in routers" -> "63 / 19" (total 82 unchanged) | AST count of route decorators: 63 in `api/index.py`, 19 in `api/routes/*_routes.py` |
| `architecture/01-system-overview.md` | "60 route handlers" / "60 routes still in it" -> 63 | Same count as above |
| `architecture/04-endpoint-reference.md` | Added a "Known gaps" row: `/v1/connectors*` not org-scoped | api-01 documents the cross-tenant bug; architecture-04 listed the routes as plainly "Guarded". Confirmed in code (`api/index.py:3905-3975`: no `org_id` on list/save, no org check on sync) |
| `architecture/06-security-architecture.md` | Same gap added to "14. Known gaps", linking `api-01-current-state` | Security page omitted a confirmed cross-tenant gap |
| `architecture/07-deployment-and-ops.md` | One intro line: today's prod is Vercel + Railway; AWS cutover (DOCBOT-2519) mid-January, links `master-backlog`, `aws-migration-plan` | Page never mentioned the planned migration |
| `architecture/01,02,04,05,06,07` and `api/02,03,04` | 23 lines: sibling refs (`03-data-model.md`, `[01-current-state.md](01-current-state.md)`, `02-public-api-plan.md#anchor`) -> slug links, e.g. `[Data Model](architecture-03-data-model)` | Relative slug links work in the OpenProject wiki; `.md` paths do not. Anchors kept |
| `_manifest.json` | `architecture-00-diagrams` source -> `docs/wiki/architecture/00-diagrams.md`; 12 agent page titles set to their H1s; removed "in progress" notes | Coordinator request; titles must match the pages |
| `00-index.md` | Link texts updated to the new titles | Titles changed |

## Checked, consistent (no edit)

- Auth header: `Authorization: Bearer <key>` everywhere; matches `api/api_key_service.py:379-384`.
- Key prefix: only `dk_live_` exists today; `dk_test_` appears only as proposed in api-02/03 and api-01 says so.
- Scopes: `hybrid:read`, `lineage:read`, `autopilot:run`, `connectors:manage` in both sets; matches `api_key_service.py:84-89`.
- Models: `openai/gpt-oss-20b` (chat/SQL/synthesis), `openai/gpt-oss-120b` (code gen), `gemini-2.5-flash` (extraction/fallback); Llama 3.3 only mentioned as removed. Matches `llm_provider.py:153`.
- Total routes 82: architecture-04 inventory and api-02 section 17 (82 rows) agree.
- Personas: 6 in `api/personas.py`, 4 shown in UI; both sets agree.
- Per-key limiter: 5 calls/s in-memory, sleeps not 429; per-IP limiter POST/DELETE only; both sets agree.
- Cookie names `docbot_session` / `docbot_anon`, free credits default 500: consistent.
- Backlog ids DOCBOT-3901..3907 in api-04 match the master backlog (EPIC-39).

## Secrets / PII scan

Clean. No email addresses, connection strings with credentials, or key-shaped values (`sk_live_`, `gsk_`, `whsec_`, `AKIA`, `ghp_`, JWTs, long `dk_live_`) in any page or in `docbot-architecture.html/.json`. Only IPs are loopback, `0.0.0.0` and the `169.254.169.254` SSRF example. Example keys are truncated placeholders (`dk_live_...`, `dk_live_a1b2c3d4`).

## Unresolved

1. `api/04-implementation-backlog.md` uses 41 `NEW-xx` placeholder ids. They need real DOCBOT ids from the master backlog (next free id) before syncing to OpenProject.
2. Anchor links (`api-02-public-api-plan#6-error-envelope` etc.) assume OpenProject generates GitHub-style heading ids. Verify after import.
3. `archify-input.md` still cites sibling `.md` files. Left as is: it is a diagram spec, not a wiki page.
4. `architecture/` also holds archify render artifacts (`docbot-architecture.html/.json`, 4 PNGs, `visual-check.*`). They are not in the manifest. Upload the PNG(s) as attachments to `architecture-00-diagrams` if needed; ignore the rest.
5. The `/v1/connectors*` cross-tenant bug is a code defect, not a docs issue. It needs a ticket (api-04 may already cover it; not checked line by line).
6. Not verified: line-number citations across all pages (spot-checked only), and the "2,717 of 2,757 tests" figure (not re-run).
