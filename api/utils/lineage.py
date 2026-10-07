"""DOCBOT-1510: per-answer lineage collector.

One ``LineageCollector`` is created per chat request and filled in by the
pipeline (retrieval, SQL, hybrid, autopilot). ``build()`` returns a plain
``Lineage`` model that is (a) streamed to the client as a ``lineage`` SSE
event just before ``done`` and (b) persisted by ``api/lineage_service.py``.

Safety rules
------------
* Metadata only. No prompts, no connection strings, no credentials.
* Source snippets are PII-masked and truncated before they are stored.
* Never raises: lineage is observability and must not break an answer.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from api.utils.pii_masking import mask_pii

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 300
MAX_SOURCES = 40
MAX_STEPS = 60

Mode = Literal["docs", "db", "csv", "hybrid", "autopilot"]
SourceKind = Literal["pdf", "sql_table", "csv", "edgar", "other"]
StepStatus = Literal["ok", "error", "skipped", "retried"]


class LineageStep(BaseModel):
    name: str
    tool: Optional[str] = None
    status: StepStatus = "ok"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None
    lane: Optional[Literal["docs", "db"]] = None


class LineageSource(BaseModel):
    id: str
    kind: SourceKind = "other"
    label: str
    page: Optional[int] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    cited: bool = True
    step_num: Optional[int] = None
    sub_question: Optional[str] = None


class LineageSql(BaseModel):
    sql: Optional[str] = None
    tables_selected: list[str] = Field(default_factory=list)
    tables_considered: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    drift_retry: bool = False
    result_preview: list[dict[str, Any]] = Field(default_factory=list)


class LineageDiscrepancy(BaseModel):
    label: str
    doc_value: Optional[float] = None
    db_value: Optional[float] = None
    delta: Optional[float] = None
    pct: Optional[float] = None


class LineageModelCall(BaseModel):
    caller: Optional[str] = None
    provider: str
    model: str
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    fallback_triggered: bool = False
    success: bool = True


class LineagePii(BaseModel):
    masked: bool = False
    counts: dict[str, int] = Field(default_factory=dict)


class Lineage(BaseModel):
    run_id: str
    mode: Mode
    intent: Optional[str] = None
    question: Optional[str] = None
    standalone_query: Optional[str] = None
    expanded_queries: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    steps: list[LineageStep] = Field(default_factory=list)
    sources: list[LineageSource] = Field(default_factory=list)
    sql: Optional[LineageSql] = None
    discrepancies: list[LineageDiscrepancy] = Field(default_factory=list)
    model_calls: list[LineageModelCall] = Field(default_factory=list)
    pii: LineagePii = Field(default_factory=LineagePii)
    cache_hit: bool = False
    persona: Optional[str] = None
    total_latency_ms: Optional[float] = None


def safe_snippet(text: Optional[str], limit: int = SNIPPET_MAX_CHARS) -> Optional[str]:
    """Collapse whitespace, PII-mask, and truncate a source excerpt."""
    if not text:
        return None
    flat = " ".join(str(text).split())
    try:
        flat = mask_pii(flat)
    except (TypeError, ValueError) as exc:
        logger.debug("lineage: snippet mask failed (%s)", exc)
        return None
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LineageCollector:
    """Mutable per-request accumulator. Every method is exception-safe."""

    def __init__(self, run_id: str, mode: Mode, question: Optional[str] = None) -> None:
        self._started = time.perf_counter()
        self._data = Lineage(run_id=run_id, mode=mode, question=question)
        self._source_ids: set[str] = set()

    @property
    def run_id(self) -> str:
        return self._data.run_id

    # -- simple setters ----------------------------------------------------
    def set_mode(self, mode: Mode) -> None:
        self._data.mode = mode

    def set_intent(self, intent: Optional[str]) -> None:
        self._data.intent = intent

    def set_persona(self, persona: Optional[str]) -> None:
        self._data.persona = persona

    def set_standalone_query(self, standalone: Optional[str]) -> None:
        if standalone and standalone != self._data.question:
            self._data.standalone_query = standalone

    def set_expanded_queries(self, queries: list[str]) -> None:
        self._data.expanded_queries = [q for q in queries if q][:8]

    def set_sub_questions(self, sub_questions: list[str]) -> None:
        self._data.sub_questions = [q for q in sub_questions if q][:12]

    def set_cache_hit(self, hit: bool) -> None:
        self._data.cache_hit = self._data.cache_hit or bool(hit)

    def set_pii(self, masked: bool, counts: Optional[dict[str, int]] = None) -> None:
        self._data.pii = LineagePii(masked=masked, counts=counts or {})

    # -- steps -------------------------------------------------------------
    def add_step(
        self,
        name: str,
        *,
        tool: Optional[str] = None,
        status: StepStatus = "ok",
        latency_ms: Optional[float] = None,
        detail: Optional[str] = None,
        lane: Optional[Literal["docs", "db"]] = None,
    ) -> None:
        if len(self._data.steps) >= MAX_STEPS:
            return
        self._data.steps.append(
            LineageStep(
                name=name,
                tool=tool,
                status=status,
                latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
                detail=safe_snippet(detail, 200),
                lane=lane,
            )
        )

    @contextmanager
    def step(
        self,
        name: str,
        *,
        tool: Optional[str] = None,
        lane: Optional[Literal["docs", "db"]] = None,
    ) -> Iterator[None]:
        """Time a block and record it; records status=error if it raises."""
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            self.add_step(
                name, tool=tool, status="error", lane=lane,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            raise
        self.add_step(
            name, tool=tool, lane=lane, latency_ms=(time.perf_counter() - t0) * 1000
        )

    # -- sources -----------------------------------------------------------
    def add_source(
        self,
        *,
        kind: SourceKind,
        label: str,
        page: Any = None,
        snippet: Optional[str] = None,
        score: Optional[float] = None,
        rerank_score: Optional[float] = None,
        cited: bool = True,
        step_num: Optional[int] = None,
        sub_question: Optional[str] = None,
    ) -> None:
        if len(self._data.sources) >= MAX_SOURCES:
            return
        page_num = _to_int(page)
        source_id = f"{kind}:{label}:{page_num if page_num is not None else '-'}"
        if step_num is not None:
            source_id += f":s{step_num}"
        if source_id in self._source_ids:
            return
        self._source_ids.add(source_id)
        self._data.sources.append(
            LineageSource(
                id=source_id,
                kind=kind,
                label=str(label)[:200],
                page=page_num,
                snippet=safe_snippet(snippet),
                score=score,
                rerank_score=rerank_score,
                cited=cited,
                step_num=step_num,
                sub_question=sub_question[:200] if sub_question else None,
            )
        )

    # -- sql ---------------------------------------------------------------
    def set_sql(
        self,
        *,
        sql: Optional[str],
        tables_selected: Optional[list[str]] = None,
        tables_considered: Optional[list[str]] = None,
        row_count: Optional[int] = None,
        execution_time_ms: Optional[float] = None,
        drift_retry: bool = False,
        result_preview: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._data.sql = LineageSql(
            sql=sql,
            tables_selected=list(tables_selected or []),
            tables_considered=list(tables_considered or []),
            row_count=row_count,
            execution_time_ms=execution_time_ms,
            drift_retry=drift_retry,
            result_preview=list(result_preview or [])[:10],
        )
        for table in tables_selected or []:
            self.add_source(kind="sql_table", label=table, snippet=None)

    # -- discrepancies -----------------------------------------------------
    def add_discrepancy(
        self,
        label: str,
        doc_value: Optional[float],
        db_value: Optional[float],
        delta: Optional[float] = None,
        pct: Optional[float] = None,
    ) -> None:
        self._data.discrepancies.append(
            LineageDiscrepancy(
                label=label[:120],
                doc_value=doc_value,
                db_value=db_value,
                delta=delta,
                pct=pct,
            )
        )

    # -- build -------------------------------------------------------------
    def build(self, model_calls: Optional[list[dict[str, Any]]] = None) -> Lineage:
        """Finalize. ``model_calls`` are rows from ``llm_calls`` for this run_id."""
        self._data.total_latency_ms = round(
            (time.perf_counter() - self._started) * 1000, 1
        )
        if model_calls:
            parsed: list[LineageModelCall] = []
            for row in model_calls:
                try:
                    parsed.append(
                        LineageModelCall(
                            caller=row.get("caller"),
                            provider=str(row.get("provider") or "unknown"),
                            model=str(row.get("model") or "unknown"),
                            latency_ms=row.get("latency_ms"),
                            input_tokens=row.get("input_tokens"),
                            output_tokens=row.get("output_tokens"),
                            estimated_cost_usd=row.get("estimated_cost_usd"),
                            fallback_triggered=bool(row.get("fallback_triggered")),
                            success=bool(row.get("success", True)),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    logger.debug("lineage: skipped malformed llm_call row (%s)", exc)
            self._data.model_calls = parsed
        return self._data
