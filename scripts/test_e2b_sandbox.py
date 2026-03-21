#!/usr/bin/env python3
"""
Test script for DOCBOT-103: E2B Sandbox Integration.

Tests BOTH:
  (a) Direct function call: from api.sandbox_service import run_python, SandboxResult
  (b) HTTP API call:        POST http://localhost:8000/api/sandbox/execute

Usage:
    E2B_API_KEY=<key> python scripts/test_e2b_sandbox.py

Flags:
    --unit-only     Run only direct function-call tests (skip HTTP)
    --api-only      Run only HTTP integration tests (skip direct import)
    --api-url URL   Override base URL (default: http://localhost:8000)
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Guard: E2B_API_KEY must be set before we import anything from the sandbox
# ---------------------------------------------------------------------------
E2B_API_KEY = os.getenv("E2B_API_KEY")

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def _print_result(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    line = f"{status} {label}"
    if detail:
        line += f" — {detail}"
    print(line)


# ---------------------------------------------------------------------------
# HTTP helper (uses stdlib urllib so no extra deps required)
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def _assert_no_error(result: Any, field: str = "error") -> tuple[bool, str]:
    val = result.get(field) if isinstance(result, dict) else getattr(result, field, None)
    if val:
        return False, f"unexpected error: {val}"
    return True, ""


def _assert_stdout_contains(result: Any, substring: str) -> tuple[bool, str]:
    stdout = result.get("stdout") if isinstance(result, dict) else getattr(result, "stdout", "")
    if substring in stdout:
        return True, f"stdout contains '{substring}'"
    return False, f"stdout was: {stdout!r}"


def _assert_has_charts(result: Any) -> tuple[bool, str]:
    charts = result.get("charts") if isinstance(result, dict) else getattr(result, "charts", [])
    if not charts:
        return False, "charts list is empty"
    first = charts[0]
    try:
        decoded = base64.b64decode(first)
    except Exception as exc:
        return False, f"chart[0] is not valid base64: {exc}"
    # PNG magic bytes: \x89PNG\r\n\x1a\n
    if decoded[:8] != b"\x89PNG\r\n\x1a\n":
        return False, "chart[0] decoded bytes are not a valid PNG (bad magic)"
    return True, f"charts list has {len(charts)} PNG(s)"


def _assert_error_contains(result: Any, substring: str) -> tuple[bool, str]:
    error = result.get("error") if isinstance(result, dict) else getattr(result, "error", None)
    if error and substring.lower() in error.lower():
        return True, f"error contains '{substring}'"
    return False, f"error was: {error!r}"


# ---------------------------------------------------------------------------
# Test code snippets
# ---------------------------------------------------------------------------

CHART_CODE = """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(['Q1', 'Q2', 'Q3'], [100, 150, 120])
ax.set_title('Revenue by Quarter')
plt.show()
"""

TIMEOUT_CODE = "import time; time.sleep(30)"
ERROR_CODE = 'raise ValueError("test error")'
ARITHMETIC_CODE = "result = 2 + 2; print(f'Result: {result}')"
PANDAS_CODE = "import pandas as pd; print(pd.__version__)"


# ---------------------------------------------------------------------------
# Unit tests — call api.sandbox_service.run_python() directly
# ---------------------------------------------------------------------------

async def _run_unit_tests() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    try:
        from api.sandbox_service import run_python  # noqa: PLC0415
    except ImportError as exc:
        print(f"{SKIP} Cannot import api.sandbox_service: {exc}")
        print("       Make sure api/sandbox_service.py is implemented.")
        return results

    # Test 1: pandas availability
    label = "Unit | pandas availability"
    try:
        t0 = time.monotonic()
        res = await run_python(PANDAS_CODE)
        elapsed = int((time.monotonic() - t0) * 1000)
        ok, detail = _assert_no_error(res)
        if ok:
            ok, detail = _assert_stdout_contains(res, ".")
        results.append((label, ok, detail or f"{elapsed}ms"))
    except Exception as exc:
        results.append((label, False, str(exc)))

    # Test 2: matplotlib chart generation
    label = "Unit | matplotlib chart generation"
    try:
        t0 = time.monotonic()
        res = await run_python(CHART_CODE)
        elapsed = int((time.monotonic() - t0) * 1000)
        ok, detail = _assert_has_charts(res)
        results.append((label, ok, detail or f"{elapsed}ms"))
    except Exception as exc:
        results.append((label, False, str(exc)))

    # Test 3: timeout enforcement
    label = "Unit | timeout enforcement (30s sleep should fail)"
    try:
        t0 = time.monotonic()
        res = await run_python(TIMEOUT_CODE)
        elapsed = int((time.monotonic() - t0) * 1000)
        ok, detail = _assert_error_contains(res, "timeout")
        results.append((label, ok, detail or f"{elapsed}ms"))
    except Exception as exc:
        results.append((label, False, str(exc)))

    # Test 4: error handling
    label = "Unit | error handling (ValueError)"
    try:
        res = await run_python(ERROR_CODE)
        ok, detail = _assert_error_contains(res, "ValueError")
        results.append((label, ok, detail))
    except Exception as exc:
        results.append((label, False, str(exc)))

    # Test 5: basic arithmetic
    label = "Unit | basic arithmetic"
    try:
        res = await run_python(ARITHMETIC_CODE)
        ok, detail = _assert_no_error(res)
        if ok:
            ok, detail = _assert_stdout_contains(res, "Result: 4")
        results.append((label, ok, detail))
    except Exception as exc:
        results.append((label, False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# Integration tests — call POST /api/sandbox/execute via HTTP
# ---------------------------------------------------------------------------

def _run_api_tests(base_url: str) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    endpoint = f"{base_url}/api/sandbox/execute"

    test_cases = [
        ("API  | pandas availability", PANDAS_CODE, _assert_no_error, lambda r: _assert_stdout_contains(r, ".")),
        ("API  | matplotlib chart generation", CHART_CODE, _assert_has_charts, None),
        ("API  | timeout enforcement (30s sleep should fail)", TIMEOUT_CODE, lambda r: _assert_error_contains(r, "timeout"), None),
        ("API  | error handling (ValueError)", ERROR_CODE, lambda r: _assert_error_contains(r, "ValueError"), None),
        ("API  | basic arithmetic", ARITHMETIC_CODE, _assert_no_error, lambda r: _assert_stdout_contains(r, "Result: 4")),
    ]

    for label, code, primary_check, secondary_check in test_cases:
        try:
            t0 = time.monotonic()
            res = _http_post(endpoint, {"code": code})
            elapsed = int((time.monotonic() - t0) * 1000)
            ok, detail = primary_check(res)
            if ok and secondary_check:
                ok, detail = secondary_check(res)
            results.append((label, ok, detail or f"{elapsed}ms"))
        except urllib.error.URLError as exc:
            results.append((label, False, f"HTTP error (is server running at {base_url}?): {exc}"))
        except Exception as exc:
            results.append((label, False, str(exc)))

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E2B sandbox integration tests for DocBot")
    parser.add_argument("--unit-only", action="store_true", help="Skip HTTP API tests")
    parser.add_argument("--api-only", action="store_true", help="Skip direct function-call tests")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()

    if not E2B_API_KEY:
        print(f"{SKIP} E2B_API_KEY is not set in the environment.")
        print("       Set it and re-run: E2B_API_KEY=<key> python scripts/test_e2b_sandbox.py")
        sys.exit(0)

    all_results: list[tuple[str, bool, str]] = []

    print("\n=== DocBot DOCBOT-103 — E2B Sandbox Tests ===\n")

    if not args.api_only:
        print("--- Unit Tests (direct function call) ---")
        unit_results = asyncio.run(_run_unit_tests())
        all_results.extend(unit_results)
        for label, passed, detail in unit_results:
            _print_result(label, passed, detail)
        if not unit_results:
            print(f"{SKIP} No unit tests ran (api/sandbox_service.py not yet importable)")
        print()

    if not args.unit_only:
        print(f"--- Integration Tests (HTTP API at {args.api_url}) ---")
        api_results = _run_api_tests(args.api_url)
        all_results.extend(api_results)
        for label, passed, detail in api_results:
            _print_result(label, passed, detail)
        print()

    if all_results:
        passed = sum(1 for _, ok, _ in all_results if ok)
        total = len(all_results)
        print(f"=== Results: {passed}/{total} passed ===\n")
        if passed < total:
            sys.exit(1)
    else:
        print("=== No tests ran. ===\n")


if __name__ == "__main__":
    main()
