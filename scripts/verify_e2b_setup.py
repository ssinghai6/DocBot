#!/usr/bin/env python3
"""
Quick E2B setup verification for DocBot (DOCBOT-103).

Run this before the full test suite to confirm the E2B SDK is installed,
the API key is valid, and a basic sandbox round-trip works.

Usage:
    E2B_API_KEY=<key> python scripts/verify_e2b_setup.py
"""

import asyncio
import os
import sys

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check_api_key() -> bool:
    key = os.getenv("E2B_API_KEY")
    if not key:
        print(f"{FAIL} E2B_API_KEY is not set in the environment.")
        print("       Export it: export E2B_API_KEY=e2b_...")
        return False
    masked = key[:8] + "..." if len(key) > 8 else "***"
    print(f"{PASS} E2B_API_KEY is set ({masked})")
    return True


def check_sdk_importable() -> bool:
    try:
        import e2b_code_interpreter  # noqa: F401
        print(f"{PASS} e2b-code-interpreter is importable")
        return True
    except ImportError as exc:
        print(f"{FAIL} Cannot import e2b_code_interpreter: {exc}")
        print("       Install it: pip install e2b-code-interpreter")
        return False


async def run_ping_test() -> bool:
    """Create a sandbox, run print('hello'), verify output, close."""
    try:
        from e2b_code_interpreter import AsyncSandbox
    except ImportError:
        print(f"{SKIP} Skipping ping test — e2b_code_interpreter not importable")
        return False

    print("       Creating sandbox (this may take 2-5s on first run)...")
    sbx = None
    try:
        sbx = await AsyncSandbox.create()
        execution = await asyncio.wait_for(
            sbx.run_code('print("hello from e2b")'),
            timeout=30.0,
        )
        stdout = "\n".join(execution.logs.stdout)
        if "hello from e2b" in stdout:
            print(f"{PASS} Sandbox ping: stdout = {stdout!r}")
            return True
        else:
            print(f"{FAIL} Sandbox ping: unexpected stdout = {stdout!r}")
            if execution.error:
                print(f"       Error: {execution.error.name}: {execution.error.value}")
            return False
    except asyncio.TimeoutError:
        print(f"{FAIL} Sandbox ping timed out after 30 seconds")
        return False
    except Exception as exc:
        print(f"{FAIL} Sandbox ping raised an exception: {type(exc).__name__}: {exc}")
        return False
    finally:
        if sbx is not None:
            try:
                await sbx.kill()
                print("       Sandbox closed cleanly.")
            except Exception as exc:
                print(f"       Warning: sandbox close failed: {exc}")


def main() -> None:
    print("\n=== E2B Setup Verification for DocBot (DOCBOT-103) ===\n")

    if not check_api_key():
        print("\nSetup status: FAILED — fix E2B_API_KEY and re-run.\n")
        sys.exit(1)

    if not check_sdk_importable():
        print("\nSetup status: FAILED — install e2b-code-interpreter and re-run.\n")
        sys.exit(1)

    print("       Running sandbox ping test...")
    ping_ok = asyncio.run(run_ping_test())

    print()
    if ping_ok:
        print("Setup status: ALL CHECKS PASSED — E2B is ready for DOCBOT-103 tests.\n")
    else:
        print("Setup status: FAILED — sandbox ping did not succeed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
