"""E2B sandbox lockdown policy — DOCBOT-1607 (SEC-04).

Centralises how every E2B sandbox is created so no code path can forget the
hardening flags:

- ``timeout=<lifetime>`` on ``Sandbox.create`` — E2B kills the VM
  server-side even if our process dies before ``kill()``. Raw user code gets
  ``SANDBOX_LIFETIME_SECONDS`` (60 s). Internal LLM-codegen paths get their
  exec timeout plus ``INSTALL_BUDGET_SECONDS`` so boot, CSV upload and a
  runtime pip install don't eat into the run (DOCBOT-2318, ``lifetime_seconds``).
- ``allow_internet_access=False`` by default (when the installed SDK
  supports the kwarg). Arbitrary user code (``POST /api/sandbox/execute``)
  NEVER gets network egress.
- Optional ``E2B_SANDBOX_TEMPLATE`` for a custom template with extra
  packages pre-baked (preferred over runtime pip installs).

Pip trade-off
-------------
E2B's default code-interpreter image already ships pandas, numpy,
matplotlib, scikit-learn, scipy, seaborn and plotly (``PREBAKED_PACKAGES``).
Only ``statsmodels`` / ``xgboost`` / ``lightgbm`` / ``tabulate`` need a
runtime ``pip install``, which needs internet. Egress is therefore granted
only when ALL of the following hold:

1. the caller is an internal LLM-codegen path (CSV query, autopilot, SQL
   chart analysis) — never raw user code;
2. ``SANDBOX_PIP_INTERNET`` is not disabled (default on);
3. the generated code imports a package that is not pre-baked — or the run
   hit ``ModuleNotFoundError``, in which case it is retried once in a fresh
   sandbox with egress (DOCBOT-2318).

Set ``SANDBOX_PIP_INTERNET=false`` (and point ``E2B_SANDBOX_TEMPLATE`` at a
template with statsmodels etc.) to remove egress entirely. Domain-scoped
egress (PyPI only) is not used because the SDK's ``network.allow_out`` only
accepts IPs/CIDRs and PyPI sits behind a rotating CDN.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

SANDBOX_LIFETIME_SECONDS = 60
# Extra seconds for internal codegen runs: boot + upload + pip install (20-40 s) + module retry.
INSTALL_BUDGET_SECONDS = 90

# Packages baked into E2B's default code-interpreter template
# (e2b-dev/code-interpreter template/requirements.txt). Never pip-installed.
PREBAKED_PACKAGES = frozenset({"scikit-learn", "scipy", "seaborn", "plotly"})


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def pip_egress_enabled() -> bool:
    """``SANDBOX_PIP_INTERNET`` — default on; read at call time."""
    return _env_flag("SANDBOX_PIP_INTERNET", True)


def packages_to_install(needed: Iterable[str]) -> list[str]:
    """Drop packages that are already pre-baked in the sandbox image."""
    return [p for p in needed if p not in PREBAKED_PACKAGES]


def internet_allowed(*, allow_pip_egress: bool, install_packages: Iterable[str]) -> bool:
    """Decide egress for one sandbox. Raw user code always passes False."""
    return bool(allow_pip_egress and pip_egress_enabled() and list(install_packages))


def install_budget(*, allow_pip_egress: bool) -> int:
    """Extra seconds an internal codegen run may spend on pip installs (0 if no egress)."""
    return INSTALL_BUDGET_SECONDS if allow_pip_egress and pip_egress_enabled() else 0


def lifetime_seconds(*, allow_pip_egress: bool, exec_timeout_seconds: float) -> int:
    """E2B VM lifetime: 60 s for raw user code; exec timeout + install budget for codegen."""
    if not allow_pip_egress:
        return SANDBOX_LIFETIME_SECONDS
    budget = install_budget(allow_pip_egress=True)
    return max(SANDBOX_LIFETIME_SECONDS, int(exec_timeout_seconds) + budget)


def _accepted_kwargs(create: Callable[..., Any]) -> Optional[set[str]]:
    """Kwarg names ``create`` accepts, or None if it takes ``**kwargs``."""
    try:
        params = inspect.signature(create).parameters.values()
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        names.add(p.name)
    return names


def create_kwargs(
    create: Callable[..., Any],
    *,
    allow_internet: bool,
    lifetime_seconds: int = SANDBOX_LIFETIME_SECONDS,
) -> Dict[str, Any]:
    """Hardened kwargs for ``Sandbox.create``, filtered to what the SDK supports."""
    wanted: Dict[str, Any] = {
        "timeout": lifetime_seconds,
        "allow_internet_access": allow_internet,
    }
    template = os.getenv("E2B_SANDBOX_TEMPLATE", "").strip()
    if template:
        wanted["template"] = template
    accepted = _accepted_kwargs(create)
    if accepted is None:
        return wanted
    kept = {k: v for k, v in wanted.items() if k in accepted}
    dropped = set(wanted) - set(kept)
    if dropped:
        logger.warning("E2B SDK does not support sandbox kwargs %s — lockdown degraded", sorted(dropped))
    return kept


def create_sandbox(
    sandbox_cls: Any,
    *,
    allow_internet: bool,
    lifetime_seconds: int = SANDBOX_LIFETIME_SECONDS,
) -> Any:
    """Create an E2B sandbox with the lockdown kwargs applied."""
    return sandbox_cls.create(
        **create_kwargs(
            sandbox_cls.create, allow_internet=allow_internet, lifetime_seconds=lifetime_seconds
        )
    )


class SandboxHolder:
    """Thread-safe handle so an async timeout can kill a sandbox mid-run.

    ``run_python`` wraps the sync E2B call in ``asyncio.wait_for``; on timeout
    the worker thread keeps running, so the async side calls ``kill()`` here
    instead of waiting for the thread's own ``finally``. ``kill()`` is
    idempotent — the thread's ``finally`` may call it again harmlessly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sandbox: Any = None
        self._killed = False

    def set(self, sandbox: Any) -> None:
        with self._lock:
            self._sandbox = sandbox
            killed = self._killed
        if killed:  # timeout fired before create() returned
            self.kill(force=True)

    def kill(self, *, force: bool = False) -> None:
        with self._lock:
            sandbox = self._sandbox
            if self._killed and not force:
                return
            self._killed = True
            self._sandbox = None
        if sandbox is None:
            return
        try:
            sandbox.kill()
        except Exception as err:  # teardown must never raise
            logger.warning("Failed to kill sandbox cleanly: %s", type(err).__name__)
