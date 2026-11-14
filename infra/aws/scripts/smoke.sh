#!/usr/bin/env bash
# Post-deploy smoke test, no secrets needed (DOCBOT-2505 / 2519).
#
#   infra/aws/scripts/smoke.sh https://api-dev.docbot.space
#   infra/aws/scripts/smoke.sh https://<vercel-preview-or-prod>   # same checks through the Vercel proxy
#
# Checks: /api/health 200 with db ok; CSP header present; /v1 without a key is
# 401; unsigned Stripe webhook is rejected (400 bad signature, or 503 when the
# webhook secret is not configured); anonymous demo init returns a session.
# Exit 0 = all pass, 1 = any failure. The demo init call costs one demo-init
# quota slot for the caller's IP (DEMO_INIT_PER_IP_PER_HOUR).
#
# Test the AWS stack before the DNS flip (cert is for api.docbot.space):
#   CONNECT_TO=<alb-dns-name> smoke.sh https://api.docbot.space
set -uo pipefail

base="${1:?usage: smoke.sh <base_url>}"
base="${base%/}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
fails=0

# req <method> <path> [curl args...] -> sets $code, body in $tmp/body, headers in $tmp/headers
req() {
  local m="$1" p="$2"; shift 2
  local host="${base#*://}"; host="${host%%/*}"
  code="$(curl -sS -m 60 -X "$m" -o "$tmp/body" -D "$tmp/headers" -w '%{http_code}' \
    ${CONNECT_TO:+--connect-to "${host}:443:${CONNECT_TO}:443"} "$@" "$base$p" 2>"$tmp/err")" || code="000"
}
check() {  # check <name> <condition-exit-status>
  if [ "$2" -eq 0 ]; then echo "PASS  $1"; else echo "FAIL  $1 (http $code: $(head -c 200 "$tmp/body" 2>/dev/null; cat "$tmp/err"))"; fails=$((fails + 1)); fi
}

req GET /api/health
grep -Eq '"db" *: *"ok"' "$tmp/body"; check "health 200, db ok" $(( $? != 0 || code != 200 ))
grep -qi '^content-security-policy:' "$tmp/headers"; check "CSP header present" $?

req GET /v1/connectors
[ "$code" = "401" ]; check "/v1 without key -> 401" $?

req POST /api/billing/webhook -H 'Content-Type: application/json' --data '{}'
[ "$code" = "400" ] || [ "$code" = "503" ]; check "unsigned webhook -> 400/503" $?

req POST "/api/demo/init?dataset=quickbite"
grep -q '"session_id"' "$tmp/body"; check "demo init -> session" $(( $? != 0 || code != 200 ))

[ "$fails" -eq 0 ] && echo "SMOKE OK: $base" || echo "SMOKE FAILED: $fails check(s) on $base"
[ "$fails" -eq 0 ]
