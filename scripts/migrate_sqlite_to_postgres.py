#!/usr/bin/env python3
"""
Migrate DocBot session data from SQLite to PostgreSQL.

Usage:
    DATABASE_URL=postgresql://user:pass@host/db python scripts/migrate_sqlite_to_postgres.py

Idempotent: rows already in PostgreSQL are skipped.
"""

import os
import sqlite3
import sys

SQLITE_PATH = "/tmp/docbot_sessions.db"


def get_pg_conn():
    import psycopg2
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(db_url)


def ensure_tables(pg) -> None:
    with pg.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                persona TEXT DEFAULT 'Generalist',
                file_count INTEGER DEFAULT 0,
                files_info TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                sources TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    pg.commit()


def migrate_sessions(sqlite_cur, pg) -> int:
    sqlite_cur.execute(
        "SELECT session_id, created_at, updated_at, persona, file_count, files_info FROM sessions"
    )
    rows = sqlite_cur.fetchall()
    if not rows:
        return 0

    from psycopg2.extras import execute_values
    with pg.cursor() as cur:
        cur.execute("SELECT session_id FROM sessions")
        existing = {r[0] for r in cur.fetchall()}
        new_rows = [r for r in rows if r[0] not in existing]
        if new_rows:
            execute_values(
                cur,
                "INSERT INTO sessions (session_id, created_at, updated_at, persona, file_count, files_info) VALUES %s",
                new_rows,
            )
    pg.commit()
    return len(new_rows)


def migrate_messages(sqlite_cur, pg) -> int:
    sqlite_cur.execute(
        "SELECT session_id, role, content, sources, timestamp FROM messages ORDER BY id ASC"
    )
    rows = sqlite_cur.fetchall()
    if not rows:
        return 0

    from psycopg2.extras import execute_values
    with pg.cursor() as cur:
        cur.execute("SELECT session_id, role, timestamp::text FROM messages")
        existing = {(r[0], r[1], r[2]) for r in cur.fetchall()}
        new_rows = [r for r in rows if (r[0], r[1], str(r[4])) not in existing]
        if new_rows:
            execute_values(
                cur,
                "INSERT INTO messages (session_id, role, content, sources, timestamp) VALUES %s",
                new_rows,
            )
    pg.commit()
    return len(new_rows)


def main() -> None:
    if not os.path.exists(SQLITE_PATH):
        print(f"SQLite database not found at {SQLITE_PATH}. Nothing to migrate.")
        sys.exit(0)

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_cur = sqlite_conn.cursor()
    pg = get_pg_conn()
    ensure_tables(pg)

    session_count = migrate_sessions(sqlite_cur, pg)
    message_count = migrate_messages(sqlite_cur, pg)
    print(f"Migrated {session_count} sessions, {message_count} messages.")

    sqlite_cur.close()
    sqlite_conn.close()
    pg.close()


if __name__ == "__main__":
    main()
