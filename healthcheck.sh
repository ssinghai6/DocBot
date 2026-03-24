#!/usr/bin/env bash
# DocBot on-prem health check — DOCBOT-605
# Usage: ./healthcheck.sh
# Exits 0 if all services are healthy, 1 otherwise.

set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

PASS=0
FAIL=0

_check() {
  local label="$1"
  local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo "  ✅  $label"
    PASS=$((PASS + 1))
  else
    echo "  ❌  $label"
    FAIL=$((FAIL + 1))
  fi
}

echo ""
echo "DocBot — Service Health Check"
echo "──────────────────────────────"

# PostgreSQL
_check "PostgreSQL" \
  "docker compose ps postgres | grep -q 'healthy'"

# Backend /api/health
_check "Backend  (http://localhost:${BACKEND_PORT}/api/health)" \
  "curl -sf http://localhost:${BACKEND_PORT}/api/health"

# Frontend root
_check "Frontend (http://localhost:${FRONTEND_PORT})" \
  "curl -sf -o /dev/null http://localhost:${FRONTEND_PORT}"

# Ollama (only checked if profile is active)
if docker compose ps ollama 2>/dev/null | grep -q "running"; then
  _check "Ollama   (http://localhost:11434)" \
    "curl -sf http://localhost:11434"
else
  echo "  ⏭   Ollama (not in active profile — skipped)"
fi

echo "──────────────────────────────"
echo "  Passed: $PASS   Failed: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo "One or more services are unhealthy. Run 'docker compose logs' to investigate."
  exit 1
fi

echo "All services healthy."
exit 0
