# ADR 0001: Move the Chroma store by copying /data to EFS

- Status: proposed (needs founder sign-off and one rehearsal, DOCBOT-2516)
- Date: 2026-11-14
- Owner: senior-dev (script), devops (EFS, rehearsal)

## Context

Railway prod keeps every session's vectors in Chroma's persistent store on the
`/data` volume (`CHROMA_DIR=/data/chroma`, DOCBOT-1001/2311). On AWS the same
image mounts EFS at `/data` through an access point that forces uid/gid 1000
(`appuser`), and `CHROMA_DIR=/data/chroma` is set in the task env.

At cutover the AWS service must answer questions about documents users
uploaded on Railway. Two options:

| | A. Copy the files | B. Re-embed from source |
|---|---|---|
| How | tar `/data/chroma` on Railway, extract onto EFS | Re-chunk and re-embed every document on AWS |
| Needs the source documents | no | **yes** |
| Result | byte-identical store | new store; chunk IDs and ordering can differ |
| Time | minutes (one tarball) | hours of CPU-bound local embedding |
| Risk | Chroma version must match (same image, so it does) | anything that changed in chunking since upload changes answers |

DocBot does **not** keep the uploaded PDFs: upload extracts text, chunks it and
writes only the chunks and embeddings to Chroma. So option B has no source to
re-embed from, except Chroma itself, which makes it a slower copy of A.

## Decision

**Option A.** Copy `/data/chroma` byte for byte with
`infra/aws/scripts/data_copy.sh`:

1. `export`: tar inside the Railway container (`railway ssh`), stream it out as
   base64, check the sha256 taken inside the container against the local file.
2. `import`: upload the tarball to S3, then run a one-off ECS task from the API
   service's own task definition (same image, same EFS mount, same network) that
   downloads it through a 15-minute presigned URL, checks the sha256 again,
   extracts into `/data/.chroma-incoming` and renames it into place. A
   non-empty `/data/chroma` is refused unless `FORCE=1`; the old copy is kept as
   `chroma.bak-<ts>`.

Both ends run the same image, so the Chroma and SQLite versions match and the
embedding model (local ONNX all-MiniLM-L6-v2) is the same.

## Consequences

- Retrieval parity: a matching sha256 means the store is the same bytes, so
  top-k results for a session are identical given the same image (Jaccard 1.0).
  The rehearsal still spot-checks 5 sessions (ask the same question on Railway
  and AWS, compare cited chunks) to catch a wrong `CHROMA_DIR` or mount.
- `/data/chroma` must be quiescent: export only after the write freeze, import
  only with the ECS service at 0 tasks.
- **Single writer.** Chroma's SQLite on EFS is safe for one process only. The
  API service must stay at `desired_count <= 1` until vectors move to pgvector
  (PLAT-0208). Autoscaling the API above 1 before then risks a corrupt store.
- The tarball holds users' document text. It lives only in the operator's
  local temp space and an encrypted S3 bucket; delete the local copy after
  import. For the Railway sunset (DOCBOT-2520) the final tarball is the archive
  copy (S3, KMS, 90-day retention).
- `railway ssh` piping a large base64 stream is the least proven step. If the
  rehearsal shows it truncates or adds banner text (the sha256 check will fail),
  fall back to splitting: `tar ... | split -b 50m` in the container and stream
  each part with the same command.

## Rehearsal checklist (DOCBOT-2516 acceptance)

- [ ] `data_copy.sh export` against Railway prod after hours; record size and time.
- [ ] `data_copy.sh import` into staging/prod EFS with the service at 0.
- [ ] Scale to 1, `smoke.sh`, then 5 sessions answer with the same cited chunks as Railway.
- [ ] Record the duration in the cutover runbook's timing column.
