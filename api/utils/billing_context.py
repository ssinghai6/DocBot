"""Per-request billing context — DOCBOT-1800 (EPIC-18 contracts).

Three pieces of request-scoped state are carried through the call tree via
``contextvars`` so deep code (``llm_provider._log_llm_call``, the E2B sandbox
runner, upload handlers) can attribute cost without threading parameters
through every function signature:

``Principal``
    *Who* is making the request: user, org, role, auth method, whether the
    email is verified, whether this is a demo session. Set once by the auth
    dependency (DOCBOT-1711 ``require_principal()``).

``KeyContext``
    *Which LLM key* pays for the call: the platform key or (post-submit,
    EPIC-19) the org's BYOK key. Its ``repr``/``str`` never expose key
    material, it refuses to pickle, and the raw key is only reachable through
    an explicit ``reveal()`` call — so an accidental log line, f-string,
    Sentry local-variable capture or cache write cannot leak it.

``RequestMeter``
    *How much* the request consumed: LLM calls, prompt/completion tokens,
    raw provider cost in integer micro-USD, sandbox milliseconds. Mutable and
    lock-protected so worker threads (``run_in_executor``) and concurrent
    asyncio tasks can all record into the same meter. DOCBOT-1802 settles the
    credit reservation from ``meter.snapshot()`` in a ``finally``.

Propagation rules (read before wiring DOCBOT-1801)
--------------------------------------------------
* ``asyncio.create_task`` / ``asyncio.gather`` / ``asyncio.to_thread`` copy
  the current context automatically — nothing to do.
* ``loop.run_in_executor`` and ``ThreadPoolExecutor.submit`` do **not**.
  Wrap the callable: ``loop.run_in_executor(None, copy_ctx_run(fn, *args))``.
  ``copy_ctx_run`` captures the context *at wrap time, on the caller's
  thread* — never call it from inside the worker.
* ``StreamingResponse`` generators may be iterated after the request-scoped
  dependency has reset its tokens. Capture with ``capture_billing_context()``
  before returning the response and re-enter with
  ``use_billing_context(snapshot)`` inside the generator.

Because the meter is a mutable object, a copied context shares the *same*
meter instance (contexts copy references), which is exactly what lets a
worker thread's sandbox seconds land on the request's meter. Re-binding a
var inside a worker never leaks back to the caller.

Units: money is integer micro-USD (1 USD = 1_000_000) end to end — no floats
accumulate on the billing path. See ``api/billing_models.py`` for the
credit unit (1 credit = 1_000 micro-USD retail).
"""

from __future__ import annotations

import contextvars
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any, Callable, Iterator, Literal, Optional, TypeVar

T = TypeVar("T")

AuthMethod = Literal["session", "api_key", "demo", "anonymous", "system"]
KeySource = Literal["platform", "byok"]
Role = Literal["viewer", "analyst", "admin"]

MICRO_USD_PER_USD = 1_000_000

# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """The authenticated (or demo / anonymous) caller of the current request.

    ``org_id`` is the billing account: credits are held and debited per org.
    A ``demo`` principal has ``is_demo=True`` and is capped by
    ``DEMO_DAILY_USD_CAP`` rather than a credit balance (DOCBOT-1711).
    """

    user_id: Optional[str]
    org_id: Optional[str]
    role: Role = "viewer"
    auth_method: AuthMethod = "anonymous"
    email_verified: bool = False
    is_demo: bool = False
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None  # id of the /v1 API key, never the key itself

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None and self.auth_method in ("session", "api_key")

    @property
    def can_spend_credits(self) -> bool:
        """F-2: credits only after email verification, never for demo/anon."""
        return (
            self.is_authenticated
            and self.org_id is not None
            and self.email_verified
            and not self.is_demo
        )


# ---------------------------------------------------------------------------
# KeyContext
# ---------------------------------------------------------------------------

_MIN_LEN_FOR_HINT = 16


class KeyContext:
    """Which LLM key pays for calls in this request.

    The raw key is stored in a name-mangled slot and is only returned by
    :meth:`reveal`. ``repr``, ``str``, ``format`` and equality never touch
    it beyond a last-4 hint (only for keys >= 16 chars, matching the
    "last4 only" storage rule in EPIC-19). Pickling / copying is refused so
    the object cannot be serialized into a cache row or a task queue.
    """

    __slots__ = ("provider", "key_source", "org_id", "_KeyContext__key")

    provider: str
    key_source: KeySource
    org_id: Optional[str]

    def __init__(
        self,
        provider: str,
        key_source: KeySource = "platform",
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> None:
        if key_source not in ("platform", "byok"):
            raise ValueError("key_source must be 'platform' or 'byok'")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "key_source", key_source)
        object.__setattr__(self, "org_id", org_id)
        object.__setattr__(self, "_KeyContext__key", api_key)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("KeyContext is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("KeyContext is immutable")

    # -- explicit access -----------------------------------------------------

    def reveal(self) -> Optional[str]:
        """Return the raw key. Call only at the provider-client boundary."""
        return self.__key

    @property
    def has_key(self) -> bool:
        return bool(self.__key)

    @property
    def key_hint(self) -> str:
        key = self.__key
        if not key:
            return "<none>"
        if len(key) < _MIN_LEN_FOR_HINT:
            return "****"
        return "****" + key[-4:]

    # -- safe rendering ------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KeyContext(provider={self.provider!r}, key_source={self.key_source!r}, "
            f"org_id={self.org_id!r}, key={self.key_hint})"
        )

    __str__ = __repr__

    def __format__(self, format_spec: str) -> str:
        return format(repr(self), format_spec)

    # -- block serialization / copying --------------------------------------

    def __reduce__(self) -> Any:
        raise TypeError("KeyContext must not be pickled or serialized")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("KeyContext must not be pickled or serialized")

    def __copy__(self) -> "KeyContext":
        return self  # immutable — sharing is safe and avoids key duplication

    def __deepcopy__(self, memo: dict[int, Any]) -> "KeyContext":
        return self

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# RequestMeter
# ---------------------------------------------------------------------------


def usd_to_micro_usd(cost_usd: float | Decimal | str | int) -> int:
    """Convert a USD amount to integer micro-USD, rounding *up*.

    Floats go through ``str()`` first so ``0.1`` becomes exactly 100_000
    rather than 100_000.00000000001. Negative / NaN / inf are rejected —
    a meter only ever accumulates consumption.
    """
    try:
        dec = Decimal(str(cost_usd))
    except InvalidOperation as exc:
        raise ValueError(f"invalid cost value: {cost_usd!r}") from exc
    if not dec.is_finite() or dec < 0:
        raise ValueError(f"cost must be a finite non-negative amount, got {cost_usd!r}")
    return int((dec * MICRO_USD_PER_USD).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class MeterSnapshot:
    """Immutable point-in-time view of a RequestMeter."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_micro_usd: int = 0
    sandbox_ms: int = 0
    uploads: int = 0
    upload_bytes: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def sandbox_seconds(self) -> float:
        return self.sandbox_ms / 1000.0


@dataclass
class RequestMeter:
    """Accumulates consumption for one request. Thread- and task-safe.

    All mutators take a ``threading.Lock``; none of them await, so the lock
    is also safe to use from the event loop thread (it is held for a few
    integer additions only).
    """

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_micro_usd: int = 0
    sandbox_ms: int = 0
    uploads: int = 0
    upload_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record_llm(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float | Decimal | str | int = 0,
    ) -> None:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        micro = usd_to_micro_usd(cost_usd)
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.cost_micro_usd += micro

    def record_sandbox(self, seconds: float, cost_usd: float | Decimal | str | int = 0) -> None:
        if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
            raise ValueError("sandbox seconds must be a finite non-negative number")
        ms = int(math.ceil(seconds * 1000))
        micro = usd_to_micro_usd(cost_usd)
        with self._lock:
            self.sandbox_ms += ms
            self.cost_micro_usd += micro

    def record_upload(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("upload size must be non-negative")
        with self._lock:
            self.uploads += 1
            self.upload_bytes += int(size_bytes)

    def snapshot(self) -> MeterSnapshot:
        with self._lock:
            return MeterSnapshot(
                llm_calls=self.llm_calls,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                cost_micro_usd=self.cost_micro_usd,
                sandbox_ms=self.sandbox_ms,
                uploads=self.uploads,
                upload_bytes=self.upload_bytes,
            )


# ---------------------------------------------------------------------------
# ContextVars + accessors
# ---------------------------------------------------------------------------

_principal_var: contextvars.ContextVar[Optional[Principal]] = contextvars.ContextVar(
    "billing_principal", default=None
)
_key_ctx_var: contextvars.ContextVar[Optional[KeyContext]] = contextvars.ContextVar(
    "billing_key_ctx", default=None
)
_meter_var: contextvars.ContextVar[Optional[RequestMeter]] = contextvars.ContextVar(
    "billing_meter", default=None
)


def get_principal() -> Optional[Principal]:
    return _principal_var.get()


def set_principal(principal: Optional[Principal]) -> contextvars.Token[Optional[Principal]]:
    return _principal_var.set(principal)


def reset_principal(token: contextvars.Token[Optional[Principal]]) -> None:
    _principal_var.reset(token)


def get_key_context() -> Optional[KeyContext]:
    return _key_ctx_var.get()


def set_key_context(key_ctx: Optional[KeyContext]) -> contextvars.Token[Optional[KeyContext]]:
    return _key_ctx_var.set(key_ctx)


def reset_key_context(token: contextvars.Token[Optional[KeyContext]]) -> None:
    _key_ctx_var.reset(token)


def get_meter() -> Optional[RequestMeter]:
    return _meter_var.get()


def set_meter(meter: Optional[RequestMeter]) -> contextvars.Token[Optional[RequestMeter]]:
    return _meter_var.set(meter)


def reset_meter(token: contextvars.Token[Optional[RequestMeter]]) -> None:
    _meter_var.reset(token)


def current_key_source() -> KeySource:
    """Key source for usage attribution; ``platform`` when nothing is bound."""
    ctx = _key_ctx_var.get()
    return ctx.key_source if ctx is not None else "platform"


@contextmanager
def bind_principal(principal: Optional[Principal]) -> Iterator[Optional[Principal]]:
    token = _principal_var.set(principal)
    try:
        yield principal
    finally:
        _principal_var.reset(token)


@contextmanager
def bind_key_context(key_ctx: Optional[KeyContext]) -> Iterator[Optional[KeyContext]]:
    token = _key_ctx_var.set(key_ctx)
    try:
        yield key_ctx
    finally:
        _key_ctx_var.reset(token)


@contextmanager
def metered_request(meter: Optional[RequestMeter] = None) -> Iterator[RequestMeter]:
    """Bind a (fresh by default) meter for the duration of the block."""
    active = meter if meter is not None else RequestMeter()
    token = _meter_var.set(active)
    try:
        yield active
    finally:
        _meter_var.reset(token)


# ---------------------------------------------------------------------------
# Snapshot / restore (SSE generators)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BillingContextSnapshot:
    principal: Optional[Principal]
    key_ctx: Optional[KeyContext]
    meter: Optional[RequestMeter]


def capture_billing_context() -> BillingContextSnapshot:
    return BillingContextSnapshot(
        principal=_principal_var.get(),
        key_ctx=_key_ctx_var.get(),
        meter=_meter_var.get(),
    )


@contextmanager
def use_billing_context(snapshot: BillingContextSnapshot) -> Iterator[BillingContextSnapshot]:
    """Re-bind all three vars from a snapshot; resets on exit."""
    p_tok = _principal_var.set(snapshot.principal)
    k_tok = _key_ctx_var.set(snapshot.key_ctx)
    m_tok = _meter_var.set(snapshot.meter)
    try:
        yield snapshot
    finally:
        _meter_var.reset(m_tok)
        _key_ctx_var.reset(k_tok)
        _principal_var.reset(p_tok)


# ---------------------------------------------------------------------------
# Thread-pool propagation
# ---------------------------------------------------------------------------


def copy_ctx_run(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Callable[[], T]:
    """Bind ``fn(*args, **kwargs)`` to a copy of the *current* context.

    Returns a zero-argument callable suitable for ``loop.run_in_executor``
    or ``executor.submit``::

        await loop.run_in_executor(None, copy_ctx_run(_run_in_sandbox, code))

    The context is captured now, on the calling thread. Each invocation runs
    in its own fresh copy of that snapshot, so the returned callable may be
    invoked several times or concurrently without ``RuntimeError: cannot
    enter context``, and vars re-bound inside the worker never leak out.
    Mutable values (the ``RequestMeter``) are shared by reference, so usage
    recorded in the worker lands on the request's meter.
    """
    captured = contextvars.copy_context()

    def _runner() -> T:
        return captured.copy().run(fn, *args, **kwargs)

    _runner.__name__ = getattr(fn, "__name__", "copy_ctx_run")
    _runner.__qualname__ = getattr(fn, "__qualname__", _runner.__name__)
    return _runner


__all__ = [
    "AuthMethod",
    "BillingContextSnapshot",
    "KeyContext",
    "KeySource",
    "MICRO_USD_PER_USD",
    "MeterSnapshot",
    "Principal",
    "RequestMeter",
    "Role",
    "bind_key_context",
    "bind_principal",
    "capture_billing_context",
    "copy_ctx_run",
    "current_key_source",
    "get_key_context",
    "get_meter",
    "get_principal",
    "metered_request",
    "reset_key_context",
    "reset_meter",
    "reset_principal",
    "set_key_context",
    "set_meter",
    "set_principal",
    "use_billing_context",
    "usd_to_micro_usd",
]
