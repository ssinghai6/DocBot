# DOCBOT-2403 Org-namespaced Chroma collections + purge
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-24 Stage A | 1 | S4 | P0 | 3 | senior-dev | - | Not on main |

## User story
As a fractional CFO, I want my uploaded documents stored in a vector collection named for my org and purged when I delete my account, so that no other tenant's query can touch them and nothing lingers after I leave.

## Why now
`_collection_name` in `api/utils/vector_store.py:123` returns `session_{id}` with no org. Retrieval relies only on session ownership checks. Stage A requires `org_{org}_session_{id}` and purge on account delete (with 1708), and 2563 later adds per-workspace collections on top.

## Scope
In:
- Collection name `org_{org}_session_{id}` for org-owned sessions; anonymous sessions keep a distinct `anon_session_{id}` prefix.
- Migration of existing collections: rename/copy via the existing `clone_store` path, or let old ones expire through the 2311 `documents_expired` event.
- `delete_store` and a new purge-by-org helper used by account deletion (1708).

Out:
- RLS, Alembic, pgvector, BYOK and KMS (Stage A exclusions: rebuilt in the platform).
- Moving off Chroma.

## Acceptance criteria
- [ ] Given a logged-in user in org A creates a session, when the PDF is indexed, then the Chroma collection name starts with `org_` and contains A's org id.
- [ ] Given an existing `session_*` collection, when the session is opened after deploy, then it is migrated or the user gets the existing documents-expired event, never another org's data.
- [ ] Given org A has 3 sessions with vectors, when the purge-by-org helper runs, then `list_stored_sessions()` returns none of them and the call is idempotent.
- [ ] Given org B, when it calls `load_store` with A's session id through any route, then the 2401 sweep still passes.
- [ ] Given the change, when the 2516 Chroma rehearsal runs, then the ADR notes the naming change so the EFS copy keeps names.

## Deliverable and approach
- Change `_collection_name(session_id)` to take the owner org; thread org from the session row at the 4 call sites in `vector_store.py`.
- Purge helper filters collection names by prefix via `_get_chroma_client()`.
- Lazy migration on first load to avoid a long boot.

## Test / verification
- Unit tests in `tests/unit/` with a temp persist dir for naming, migration and purge.
- 2401 sweep green.

## Dependencies
Blocked by: 2401, 2402
Blocks: 1708 (purge on delete), 2563, 2575

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Lazy migration on load or one-shot script at deploy? One-shot is simpler but must run before the 2516 Chroma copy.
- Points: migrating existing `session_*` collections and lining up with the 2516 Chroma copy may not fit in 3. Proposed to the founder for re-sizing, not changed here.
