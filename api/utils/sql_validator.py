"""sqlglot-based SQL validation — AST-level, no regex."""

import re
from typing import Optional
import sqlglot
import sqlglot.expressions as exp
from sqlglot.tokens import TokenType


class QueryValidationError(Exception):
    """Raised when generated SQL fails AST validation."""


# azure_sql is a DocBot dialect name, not a sqlglot one (unknown -> ValueError).
_DIALECT_ALIASES = {"postgresql": "postgres", "azure_sql": "tsql"}

_READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_WRITE_NODES = tuple(
    t for t in (
        getattr(exp, name, None)
        for name in ("Insert", "Update", "Delete", "Merge", "Create", "Drop",
                     "Alter", "AlterTable", "TruncateTable", "Command", "Copy", "Into")
    ) if isinstance(t, type)
)


def validate_and_sanitize_sql(sql: str, dialect: Optional[str] = None) -> str:
    """
    Parse *sql* with sqlglot, enforce SELECT-only, inject LIMIT if absent.

    Returns the sanitized SQL string (same dialect as input).

    Raises:
        QueryValidationError: with a user-safe reason string.
            The *invalid SQL is never included* in the error message.
    """
    dialect = _DIALECT_ALIASES.get(dialect, dialect)
    try:
        statements = sqlglot.parse(sql, dialect=dialect)
    except sqlglot.errors.SqlglotError as exc:  # ParseError, or TokenError on a truncated literal
        raise QueryValidationError("Generated SQL could not be parsed.") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise QueryValidationError(
            "Expected exactly one SQL statement; multiple or empty statements are not allowed."
        )

    stmt = statements[0]

    # Root must be a read query: SELECT (incl. CTEs) or a UNION / INTERSECT /
    # EXCEPT of SELECTs — tie-out comparisons are naturally UNION ALL queries.
    if not isinstance(stmt, _READ_ROOTS):
        raise QueryValidationError(
            "Only SELECT queries are permitted. Write operations are not allowed."
        )
    # No write / DDL node anywhere in the tree (e.g. a data-modifying CTE).
    if any(isinstance(node, _WRITE_NODES) for node in stmt.walk()):
        raise QueryValidationError(
            "Only SELECT queries are permitted. Write operations are not allowed."
        )

    # Execute the text that was validated rather than sqlglot's regeneration:
    # round-tripping is lossy (sqlite drops CTE column lists, so
    # `WITH doc(metric, amount) AS (VALUES ...)` broke with "no such column").
    text = sql.strip().rstrip(";").rstrip()
    if stmt.args.get("limit") is not None or stmt.args.get("fetch") is not None:
        return text
    if dialect == "tsql":
        return stmt.limit(500).sql(dialect=dialect)  # needs TOP, not LIMIT
    # Newline so a trailing "-- comment" can't swallow the LIMIT.
    return f"{text}\nLIMIT 500"


_FENCE_RE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)```", re.DOTALL)
# Query start: a line beginning with SELECT/WITH (or "(" for a parenthesized
# UNION branch), optionally after a short label ("Query:", "SQL:", "sql ").
_SQL_START_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z][\w \t-]{0,40}:[ \t]*|sql(?:ite)?[ \t]+)?(?P<q>\(*[ \t]*(?:WITH|SELECT)\b)",
    re.IGNORECASE | re.MULTILINE,
)
# Unindented prose after the query ("This query ...", "Note: ...", "**Why**").
_PROSE_LINE_RE = re.compile(r"^(?:(?P<word>[A-Z][a-z]+)[ \t:,.]|\*\*|#{1,6}[ \t])", re.MULTILINE)


def _cut_trailing_prose(sql: str, keywords: set) -> str:
    for m in _PROSE_LINE_RE.finditer(sql, 1):
        word = m.group("word")
        if word is None or word.upper() not in keywords:
            return sql[: m.start()]
    return sql


def _split_statements(sql: str, dialect_obj) -> list:
    """Split on top-level ";" using sqlglot's tokenizer (respects literals)."""
    try:
        cuts = [t.start for t in dialect_obj.tokenize(sql) if t.token_type == TokenType.SEMICOLON]
    except sqlglot.errors.SqlglotError:  # unterminated literal, e.g. a truncated reply
        cuts = [sql.find(";")] if ";" in sql else []
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(sql[prev:c])
        prev = c + 1
    pieces.append(sql[prev:])
    return [p.strip() for p in pieces if p.strip()]


def _union_if_trivial(pieces: list, dialect: Optional[str]) -> Optional[str]:
    """Join plain SELECTs with UNION ALL when that is obviously what was meant:
    each is a single SELECT (no CTE/ORDER/LIMIT/*) with the same column count."""
    widths = set()
    for p in pieces:
        try:
            stmt = sqlglot.parse_one(p, dialect=dialect)
        except sqlglot.errors.SqlglotError:
            return None
        if (
            not isinstance(stmt, exp.Select)
            or any(stmt.args.get(k) for k in ("with_", "order", "limit"))
            or any(isinstance(e, exp.Star) for e in stmt.expressions)
        ):
            return None
        widths.add(len(stmt.expressions))
    return "\nUNION ALL\n".join(pieces) if len(widths) == 1 else None


def extract_sql(text: str, dialect: Optional[str] = None) -> str:
    """Pull one SQL statement out of an LLM reply.

    Handles: a fence (any language tag, even unterminated), leading prose or a
    label ("Query:", "SQL:", "sql "), a parenthesized UNION first branch,
    trailing explanations, and several statements (joined with UNION ALL only
    when trivially compatible, else the first). Never makes a non-SELECT
    acceptable: ``validate_and_sanitize_sql`` still parses the result and
    enforces exactly one read-only statement.
    """
    raw = (text or "").strip()
    fenced = _FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    elif raw.startswith("```"):  # truncated reply: opening fence only
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
    match = _SQL_START_RE.search(raw)
    if match:
        raw = raw[match.start("q"):]
    dialect = _DIALECT_ALIASES.get(dialect, dialect)
    try:
        dialect_obj = sqlglot.Dialect.get_or_raise(dialect)
    except ValueError:
        dialect_obj = sqlglot.Dialect.get_or_raise(None)
    keywords = {k.split()[0] for k in dialect_obj.tokenizer_class.KEYWORDS}
    pieces = _split_statements(_cut_trailing_prose(raw, keywords), dialect_obj)
    if not pieces:
        return ""
    if len(pieces) > 1:
        return _union_if_trivial(pieces, dialect) or pieces[0]
    return pieces[0]
