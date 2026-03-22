"""
Shared pytest fixtures for DocBot test suite.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet


# ---------------------------------------------------------------------------
# Environment setup — must happen before any api.* imports
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def set_test_env():
    """Inject required env vars for the entire test session."""
    os.environ.setdefault("DB_ENCRYPTION_KEY", Fernet.generate_key().decode())
    os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost:5432/testdb")


@pytest.fixture(scope="session")
def fernet_key() -> str:
    return os.environ["DB_ENCRYPTION_KEY"]


# ---------------------------------------------------------------------------
# SQLite test database
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with orders + products tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE orders "
        "(id INTEGER PRIMARY KEY, customer TEXT, amount REAL, status TEXT, month TEXT)"
    )
    conn.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?)",
        [
            (1, "Alice", 100.0, "paid", "Jan"),
            (2, "Bob", 200.0, "pending", "Jan"),
            (3, "Alice", 50.0, "paid", "Feb"),
            (4, "Charlie", 300.0, "paid", "Feb"),
        ],
    )
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?)",
        [(1, "Widget", 9.99), (2, "Gadget", 49.99)],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sqlite_url(sqlite_db_path: Path) -> str:
    return f"sqlite:///{sqlite_db_path}"


# ---------------------------------------------------------------------------
# CSV bytes fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_csv_bytes() -> bytes:
    return (
        b"product,revenue,units,date\n"
        b"Widget,1000.50,10,2024-01-01\n"
        b"Gadget,2500.00,5,2024-01-02\n"
        b"Widget,1200.75,12,2024-01-03\n"
    )


@pytest.fixture
def bom_csv_bytes() -> bytes:
    return b"\xef\xbb\xbfname,value\nAlice,100\nBob,200\n"


@pytest.fixture
def messy_csv_bytes() -> bytes:
    """CSV with quoted commas, mixed types, extra whitespace."""
    return (
        b'  Product Name ,  Revenue , Notes\n'
        b'"Widget, Pro",1000.5,"Best seller"\n'
        b'"Gadget, Mini",2500,"New"\n'
    )
