#!/usr/bin/env bash
# DOCBOT-2515: copy the DocBot Postgres database (Railway) into RDS and prove it.
#
#   SOURCE_DATABASE_URL=... TARGET_DATABASE_URL=... infra/aws/scripts/pg_migrate.sh
#
# Steps: pg_dump -Fc from SOURCE -> pg_restore --no-owner --no-acl --clean
# --if-exists --single-transaction into TARGET -> per-table row count + content
# checksum diff -> trigger list diff -> immutability triggers still block a
# DELETE -> every owned sequence is past max(column).
#
# Exit codes: 0 = verified, 1 = verification mismatch, 2 = usage/preflight/tool error.
#
# Safety:
#   - Connection strings are read from env only, never printed, never put on a
#     command line (each tool gets PGHOST/PGUSER/PGPASSWORD/... in its own env).
#   - pg_restore --clean DROPS the target's copies of every dumped object. The
#     target must be an RDS instance (has the rds_superuser role) so a swapped
#     SOURCE/TARGET can never wipe Railway. ALLOW_ANY_TARGET=1 lifts that
#     (local tests only).
#   - Stop the target app first (ECS desired count 0): the restore is one
#     transaction and the app would hold locks / write during it.
#   - Freeze writes on the source first (cutover T-0), or the post-dump source
#     counts can drift from the dump and the diff will (correctly) fail.
#
# Optional env:
#   DUMP_FILE  keep the dump at this path (cutover: archive it, DOCBOT-2520).
#              Default: a private temp file, deleted on exit.
#   PG_BIN     directory with pg_dump/pg_restore/psql (e.g. a postgresql@16 bin
#              dir) when the PATH ones are the wrong major version.
set +x
set -euo pipefail
umask 077

die() { echo "pg_migrate: $*" >&2; exit 2; }

[ -n "${SOURCE_DATABASE_URL:-}" ] || die "SOURCE_DATABASE_URL is not set"
[ -n "${TARGET_DATABASE_URL:-}" ] || die "TARGET_DATABASE_URL is not set"
[ "$SOURCE_DATABASE_URL" != "$TARGET_DATABASE_URL" ] || die "SOURCE and TARGET are the same database"
BIN="${PG_BIN:+$PG_BIN/}"
for t in "${BIN}pg_dump" "${BIN}pg_restore" "${BIN}psql" python3; do
  command -v "$t" >/dev/null || die "missing tool: $t"
done

# Run a command against SOURCE or TARGET. The URL is parsed into libpq env vars
# in a child process, both URL vars are removed, then the tool is exec'd.
on() {
  local which="$1"; shift
  python3 -c '
import os, sys, urllib.parse as up
which, cmd = sys.argv[1], sys.argv[2:]
url = os.environ[which + "_DATABASE_URL"]
p = up.urlsplit(url)
if p.scheme.split("+")[0] not in ("postgres", "postgresql"):
    sys.exit("pg_migrate: %s_DATABASE_URL is not a postgres:// URL" % which)
env = {k: v for k, v in os.environ.items() if not k.endswith("_DATABASE_URL")}
q = dict(up.parse_qsl(p.query))
for var, val in (("PGHOST", p.hostname), ("PGPORT", str(p.port) if p.port else None),
                 ("PGUSER", up.unquote(p.username or "")), ("PGPASSWORD", up.unquote(p.password or "")),
                 ("PGDATABASE", up.unquote(p.path.lstrip("/"))), ("PGSSLMODE", q.get("sslmode"))):
    if val:
        env[var] = val
env["PGTZ"] = "UTC"  # identical text rendering of timestamptz on both sides
os.execvpe(cmd[0], cmd, env)
' "$which" "$@"
}

major() { on "$1" "${BIN}psql" -X -A -t -q -c "SHOW server_version_num" | cut -c1-2; }
src_major="$(major SOURCE)" || die "cannot connect to SOURCE"
tgt_major="$(major TARGET)" || die "cannot connect to TARGET"
# Only RDS has the rds_superuser role. Checked on the server, not the hostname,
# so it also holds through an SSM port forward (TARGET host = localhost).
is_rds="$(on TARGET "${BIN}psql" -X -A -t -q -c "SELECT count(*) FROM pg_roles WHERE rolname = 'rds_superuser'")"
if [ "${ALLOW_ANY_TARGET:-}" != "1" ] && [ "$is_rds" != "1" ]; then
  die "TARGET is not an RDS instance — refusing (pg_restore --clean drops target objects)"
fi
cli_major="$("${BIN}pg_dump" --version | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
echo "server majors: source=$src_major target=$tgt_major; client=$cli_major"
[ "$cli_major" -ge "$src_major" ] || die "pg_dump $cli_major is older than the source server ($src_major); set PG_BIN"
[ "$tgt_major" -ge "$src_major" ] || die "target major $tgt_major < source major $src_major; restore into an equal or newer major"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
dump="${DUMP_FILE:-$work/docbot.dump}"

# Exact row count + content checksum of every user table, one line per table.
STATS_SQL="
SELECT n.nspname || '.' || c.relname || ' ' ||
       (xpath('/row/c/text()', query_to_xml(format(
         'SELECT count(*) || '':'' || md5(coalesce(string_agg(t::text, E''\n'' ORDER BY t::text), '''')) AS c FROM %I.%I t',
         n.nspname, c.relname), false, true, '')))[1]::text
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND n.nspname NOT LIKE 'pg_toast%' AND n.nspname NOT LIKE 'pg_temp%'
ORDER BY 1;"
TRIGGERS_SQL="SELECT tgrelid::regclass || ' ' || tgname FROM pg_trigger WHERE NOT tgisinternal ORDER BY 1;"

start=$(date +%s)
echo "1/5 dumping source..."
on SOURCE "${BIN}pg_dump" -Fc --no-owner --no-acl -f "$dump"
on SOURCE "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c "$STATS_SQL" > "$work/src.stats"
on SOURCE "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c "$TRIGGERS_SQL" > "$work/src.triggers"
echo "    $(wc -l < "$work/src.stats" | tr -d ' ') tables, dump $(du -h "$dump" | cut -f1)"

echo "2/5 restoring into target (single transaction)..."
# COMMENT ON EXTENSION needs extension ownership, which the RDS master user lacks.
"${BIN}pg_restore" -l "$dump" | grep -v 'COMMENT - EXTENSION' > "$work/restore.list"
on TARGET "${BIN}pg_restore" --no-owner --no-acl --clean --if-exists --single-transaction \
  --exit-on-error -L "$work/restore.list" -d "" "$dump" \
  || { echo "pg_migrate: restore failed and was rolled back; target unchanged" >&2; exit 2; }
on TARGET "${BIN}psql" -X -q -c "ANALYZE" >/dev/null

fail=0
echo "3/5 row counts + checksums..."
on TARGET "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c "$STATS_SQL" > "$work/tgt.stats"
if diff "$work/src.stats" "$work/tgt.stats" > "$work/stats.diff"; then
  echo "    OK: all $(wc -l < "$work/tgt.stats" | tr -d ' ') tables match (table count:md5)"
else
  echo "    MISMATCH (< source, > target):"; sed 's/^/      /' "$work/stats.diff"; fail=1
fi

echo "4/5 triggers..."
on TARGET "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c "$TRIGGERS_SQL" > "$work/tgt.triggers"
if diff "$work/src.triggers" "$work/tgt.triggers" > "$work/trg.diff"; then
  echo "    OK: $(wc -l < "$work/tgt.triggers" | tr -d ' ') triggers match"
else
  echo "    MISMATCH:"; sed 's/^/      /' "$work/trg.diff"; fail=1
fi
# Each *_no_mutate trigger (audit_log, credit_ledger) must still reject a DELETE.
# Run inside a transaction that is always rolled back, so nothing changes even
# if the trigger is missing.
while read -r tbl tg; do
  [ -n "$tbl" ] || continue
  rows="$(on TARGET "${BIN}psql" -X -A -t -q -c "SELECT count(*) FROM (SELECT 1 FROM $tbl LIMIT 1) s")"
  if [ "$rows" = "0" ]; then echo "    skip $tg: $tbl is empty"; continue; fi
  if out="$(on TARGET "${BIN}psql" -X -q -v ON_ERROR_STOP=1 \
       -c "BEGIN" -c "DELETE FROM $tbl WHERE ctid = (SELECT ctid FROM $tbl LIMIT 1)" -c "ROLLBACK" 2>&1)"; then
    echo "    FAIL: DELETE on $tbl was NOT blocked by $tg"; fail=1
  elif grep -q "immutable" <<<"$out"; then
    echo "    OK: $tg blocks DELETE on $tbl"
  else
    echo "    FAIL: DELETE on $tbl errored for another reason: $(head -1 <<<"$out")"; fail=1
  fi
done < <(grep ' .*_no_mutate$' "$work/tgt.triggers" || true)

echo "5/5 sequences..."
SEQ_SQL="
SELECT format('%s.%s', d.refobjid::regclass, a.attname) || ' ' || d.objid::regclass::text
FROM pg_depend d
JOIN pg_class sc ON sc.oid = d.objid AND sc.relkind = 'S'
JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid
WHERE d.deptype IN ('a', 'i') AND d.classid = 'pg_class'::regclass ORDER BY 1;"
while read -r col seq; do
  [ -n "$col" ] || continue
  tbl="${col%.*}"; c="${col##*.}"
  ok="$(on TARGET "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c \
    "SELECT (SELECT CASE WHEN is_called THEN last_value ELSE last_value - 1 END FROM $seq) >= coalesce((SELECT max(\"$c\") FROM $tbl), 0)")"
  if [ "$ok" = "t" ]; then echo "    OK: $seq"; else echo "    FAIL: $seq is behind max($col)"; fail=1; fi
done < <(on TARGET "${BIN}psql" -X -A -t -q -v ON_ERROR_STOP=1 -c "$SEQ_SQL")

echo "duration: $(( $(date +%s) - start ))s"
[ -n "${DUMP_FILE:-}" ] && echo "dump kept at DUMP_FILE (contains encrypted credentials: store under KMS, never commit)"
if [ "$fail" -ne 0 ]; then echo "RESULT: MISMATCH — do not cut over"; exit 1; fi
echo "RESULT: VERIFIED"
