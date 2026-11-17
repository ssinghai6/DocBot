# DOCBOT-2578 NEW: DB_ENCRYPTION_KEY MultiFernet dual-key rotation (needed for 2509)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S1 | P1 | 3 | senior-dev | - | New (2026-11-17) |

## User story
As the founder holding design partners' saved database credentials, I want to rotate `DB_ENCRYPTION_KEY` without making any stored credential unreadable, so that a suspected key leak or the cutover never forces customers to re-enter connection strings.

## Why now
Today one Fernet key encrypts everything: `api/utils/encryption.py:12` builds `Fernet(key)` from `DB_ENCRYPTION_KEY`, and the envelope KEK in `api/utils/envelope_encryption.py` (per-org DEKs, DOCBOT-1529) is also `Fernet(DB_ENCRYPTION_KEY)`. Changing the env var breaks every stored row. 2509 (secrets rotation, S2) needs a dual-key path first.

## Scope
In:
- `DB_ENCRYPTION_KEY` accepts an ordered, comma-separated key list (new key first); a single key keeps working unchanged.
- `MultiFernet` in `api/utils/encryption.py` and for the KEK in `envelope_encryption.py`: encrypt with the first key, decrypt with any.
- Idempotent re-encrypt script (`scripts/` or `api/utils/` CLI) that rotates stored `db_connections` creds blobs and re-wraps per-org DEKs.
- Tests: old-only, both, new-only keys.

Out:
- Rotating per-org DEKs themselves; KMS-backed keys (2523, CMEK 2807).

## Acceptance criteria
- [ ] Given `DB_ENCRYPTION_KEY` holds one key, when the app encrypts and decrypts, then behavior and token format are unchanged (existing tests pass).
- [ ] Given a row encrypted with key A and `DB_ENCRYPTION_KEY=B,A`, when the row is decrypted, then it succeeds, and a new encryption uses key B.
- [ ] Given `DB_ENCRYPTION_KEY=B,A`, when the re-encrypt script runs twice, then the second run reports 0 rows changed, and afterwards every creds blob and wrapped DEK decrypts with `DB_ENCRYPTION_KEY=B` alone.
- [ ] Given a malformed key in the list, when the app boots, then it fails with a clear error that never prints the key.
- [ ] Given the change, when `pytest tests/unit -k encryption` runs in CI, then old-only, both and new-only cases pass.

## Deliverable and approach
Small change in the two utils modules (CLAUDE.md: logic in utils, not `index.py`). The script reads keys only from env and prints counts only. The 2509 runbook documents the order: prepend new key, deploy, run script, remove old key, deploy.

## Test / verification
New tests in `tests/unit/test_encryption.py` and `tests/unit/` for envelope wrapping; integration test of the script against a temp SQLite DB in `tests/integration/`.

## Dependencies
Blocked by: none.
Blocks: 2509.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should the CSV raw-bytes blobs in `db_connections` (dialect `csv`) be re-encrypted by the same script? They use the same creds path, so the default is yes.
