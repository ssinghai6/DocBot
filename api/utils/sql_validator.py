"""sqlglot-based SQL validation — AST-level, no regex."""

from typing import Optional
import sqlglot
import sqlglot.expressions as exp


class QueryValidationError(Exception):
    """Raised when generated SQL fails AST validation."""


_DIALECT_ALIASES = {"postgresql": "postgres"}


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
    except sqlglot.errors.ParseError as exc:
        raise QueryValidationError("Generated SQL could not be parsed.") from exc

    if not statements or len(statements) != 1:
        raise QueryValidationError(
            "Expected exactly one SQL statement; multiple or empty statements are not allowed."
        )

    stmt = statements[0]

    # Root must be a SELECT (includes CTEs whose final clause is SELECT)
    if not isinstance(stmt, exp.Select):
        raise QueryValidationError(
            "Only SELECT queries are permitted. Write operations are not allowed."
        )

    # Inject LIMIT 500 if no LIMIT is present
    if stmt.args.get("limit") is None:
        stmt = stmt.limit(500)

    return stmt.sql(dialect=dialect)
