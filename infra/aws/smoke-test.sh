#!/usr/bin/env bash
# Post-deploy smoke test for an AWS-hosted DocBot API (DOCBOT-2505).
# Usage: infra/aws/smoke-test.sh https://api-dev.docbot.space [allowed-origin]
# Checks: TLS + /api/health {status: ok, db: ok} (retries while tasks roll),
# HTTP->HTTPS redirect, CORS for the allowed origin, health latency < 3s.
# Manual parts of 2505 (PDF chat, CSV/E2B, 120s SSE, EFS persistence) need a
# logged-in user and are listed in infra/aws/docs/first-deploy.md.
set -euo pipefail

BASE="${1:?usage: smoke-test.sh <https://api-host> [origin]}"
BASE="${BASE%/}"
ORIGIN="${2:-}"
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

body=""
for _ in $(seq 1 20); do
  if body=$(curl -fsS --max-time 10 "$BASE/api/health"); then break; fi
  body=""; sleep 15
done
[ -n "$body" ] || fail "/api/health never returned 200"
python3 -c 'import json,sys; j=json.loads(sys.argv[1]); assert j.get("status")=="ok" and j.get("db")=="ok", j' "$body" \
  || fail "health payload not ok: $body"
echo "ok   health: $body"

host="${BASE#https://}"
code=$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' "http://$host/api/health")
case "$code" in 30[18]\ https://*) echo "ok   http->https: $code" ;; *) fail "http redirect: $code" ;; esac

t=$(curl -s -o /dev/null -w '%{time_total}' "$BASE/api/health")
python3 -c 'import sys; sys.exit(float(sys.argv[1]) >= 3)' "$t" || fail "health took ${t}s"
echo "ok   latency: ${t}s"

if [ -n "$ORIGIN" ]; then
  acao=$(curl -s -D - -o /dev/null -X OPTIONS -H "Origin: $ORIGIN" -H "Access-Control-Request-Method: POST" "$BASE/api/health" \
    | tr -d '\r' | awk -F': ' 'tolower($1)=="access-control-allow-origin"{print $2}')
  [ "$acao" = "$ORIGIN" ] || fail "CORS: got '$acao' for $ORIGIN"
  echo "ok   cors: $ORIGIN"
fi
echo "SMOKE PASS $BASE"
