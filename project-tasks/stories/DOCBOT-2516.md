# DOCBOT-2516 Chroma /data ADR + rehearsal
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-25 AWS | 1 | S3 | P1 | 5 | senior-dev | - | Open (Wave B) |

## User story
As a design partner whose board pack is already indexed on Railway, I want my existing sessions to keep answering the same way after the move to AWS, so that I do not have to re-upload documents or get different citations.

## Why now
DocBot keeps only chunks and embeddings in Chroma (no source PDFs), so the vector store must be copied, not rebuilt. ADR `infra/aws/adr/0001-chroma-data.md` on `origin/dev` proposes option A (tar `/data/chroma` onto EFS) and needs founder sign-off and one rehearsal before 2519.

## Scope
In:
- Founder sign-off on ADR 0001 (A: copy files, B: re-embed; B has no source text to re-embed from).
- Rehearsal with `infra/aws/scripts/data_copy.sh`: Railway `/data/chroma` tarball to S3 migration bucket to prod EFS, sha256 checked at both ends.
- Retrieval parity on 20 sampled sessions.
- ADR records `desired_count <= 1` until pgvector (PLAT-0208).

Out:
- pgvector migration; re-embedding.

## Acceptance criteria
- [ ] Given ADR 0001, when the founder signs, then its Status reads accepted with the date.
- [ ] Given the tarball, when `data_copy.sh` extracts it on prod EFS, then the sha256 of the archive matches at source and destination and file ownership is uid/gid 1000.
- [ ] Given 20 sampled session IDs, when the same question is run against Railway and against the restored EFS store, then the top-5 chunk ID sets have Jaccard similarity of at least 0.9 for every session.
- [ ] Given the rehearsal, when it finishes, then copy duration and archive size are recorded in `infra/aws/runbooks/cutover.md`.

## Deliverable and approach
Senior-dev owns the script and parity check (a small Python helper run in the ops task that prints chunk IDs only, never document text); devops provides EFS access. The founder produces the Railway tarball and shares only its S3 key.

## Test / verification
Parity table (session hash, Jaccard) in the ADR appendix; shellcheck on `data_copy.sh` in CI.

## Dependencies
Blocked by: 2514, 2576.
Blocks: 2518, 2519.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Sessions on Railway that expire between rehearsal and cutover: accept the delta, or re-copy at T-0 (the runbook default)?
