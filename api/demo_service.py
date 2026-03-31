"""
DocBot Demo Mode — Sandbox with pre-loaded financial data.

Provides a /api/demo/init endpoint that creates:
  1. A RAG session with hardcoded TechCorp 10-K financial text chunks
  2. A SQLite database with matching financial tables
  3. Returns session_id + connection_id for immediate hybrid analysis

The demo lets new users experience hybrid doc+DB analysis, discrepancy
detection, and Autopilot without uploading their own data.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

from langchain_core.documents import Document
from sqlalchemy import Table, insert

from api.utils.encryption import encrypt_credentials
from api.utils.vector_store import create_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample financial data — TechCorp Inc. (fictitious)
# ---------------------------------------------------------------------------

DEMO_COMPANY = "TechCorp Inc."
DEMO_TICKER = "TCORP"
DEMO_FILING = "Annual Report (10-K) — Fiscal Year 2024"

# These chunks simulate what a real 10-K PDF extraction would produce.
# They contain deliberate discrepancies with the DB data for demo purposes.
DEMO_DOCUMENT_CHUNKS = [
    Document(
        page_content=(
            "TechCorp Inc. — Annual Report (10-K) for the Fiscal Year Ended December 31, 2024.\n\n"
            "PART I — BUSINESS OVERVIEW\n\n"
            "TechCorp Inc. is a global enterprise software company specializing in AI-powered "
            "analytics and cloud infrastructure. Founded in 2015 and headquartered in San Francisco, "
            "the company serves over 2,400 enterprise customers across 45 countries. Our platform "
            "integrates real-time data processing, predictive analytics, and automated reporting "
            "for financial services, healthcare, and technology sectors."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "PART II — FINANCIAL HIGHLIGHTS\n\n"
            "Total Revenue: $5.2 billion (up 18% year-over-year from $4.4 billion in FY2023)\n"
            "Cost of Revenue: $2.08 billion (gross margin: 60%)\n"
            "Operating Expenses: $1.82 billion\n"
            "Operating Income: $1.30 billion (operating margin: 25%)\n"
            "Net Income: $1.14 billion (net margin: 21.9%)\n"
            "Earnings Per Share (diluted): $4.56\n"
            "Free Cash Flow: $1.38 billion\n\n"
            "The company achieved record revenue driven by strong demand for our AI Analytics "
            "platform, which grew 32% year-over-year to $2.1 billion."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 3},
    ),
    Document(
        page_content=(
            "REVENUE BY SEGMENT — FY2024\n\n"
            "AI Analytics Platform: $2,100M (40.4% of total revenue, up 32% YoY)\n"
            "Cloud Infrastructure: $1,560M (30.0%, up 14% YoY)\n"
            "Professional Services: $890M (17.1%, up 8% YoY)\n"
            "Data Integration: $650M (12.5%, up 11% YoY)\n\n"
            "AI Analytics was the fastest growing segment, driven by new enterprise contracts "
            "and expansion of existing customers. Average contract value increased to $1.2M from $950K."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 5},
    ),
    Document(
        page_content=(
            "QUARTERLY PERFORMANCE — FY2024\n\n"
            "Q1 2024: Revenue $1,180M, Net Income $240M, Operating Margin 23%\n"
            "Q2 2024: Revenue $1,260M, Net Income $275M, Operating Margin 24%\n"
            "Q3 2024: Revenue $1,320M, Net Income $295M, Operating Margin 25%\n"
            "Q4 2024: Revenue $1,440M, Net Income $330M, Operating Margin 27%\n\n"
            "The company demonstrated consistent sequential growth throughout the year, "
            "with Q4 representing the strongest quarter driven by enterprise deal closings "
            "and holiday-season cloud consumption."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 7},
    ),
    Document(
        page_content=(
            "BALANCE SHEET HIGHLIGHTS — December 31, 2024\n\n"
            "Total Assets: $12.8 billion\n"
            "Cash and Equivalents: $3.2 billion\n"
            "Total Debt: $1.8 billion (debt-to-equity ratio: 0.28)\n"
            "Stockholders' Equity: $6.4 billion\n"
            "Goodwill and Intangibles: $4.1 billion\n\n"
            "The company maintained a strong balance sheet with a net cash position "
            "of $1.4 billion. During FY2024, TechCorp completed two strategic acquisitions "
            "totaling $1.1 billion to expand AI capabilities."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 9},
    ),
    Document(
        page_content=(
            "RISK FACTORS\n\n"
            "1. Competition: The enterprise AI market is highly competitive with established players "
            "(Microsoft, Google, Salesforce) and well-funded startups entering the space.\n"
            "2. Customer Concentration: Our top 10 customers represent 22% of total revenue. "
            "Loss of any major customer could materially impact results.\n"
            "3. AI Regulation: Evolving AI regulations in the EU (AI Act) and US may require "
            "significant compliance investment and could limit product capabilities.\n"
            "4. Talent Retention: Competition for AI/ML engineers remains intense. "
            "Our engineering turnover rate was 12% in FY2024.\n"
            "5. Foreign Exchange: 35% of revenue is denominated in non-USD currencies, "
            "exposing the company to currency fluctuation risks."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 12},
    ),
    Document(
        page_content=(
            "GUIDANCE — FY2025\n\n"
            "Revenue: $6.0–6.2 billion (15–19% growth)\n"
            "Operating Margin: 25–27%\n"
            "Free Cash Flow: $1.5–1.7 billion\n"
            "Capital Expenditures: $800M–900M (primarily data center expansion)\n\n"
            "Management expects continued strong demand for AI Analytics, with the segment "
            "projected to reach $2.8 billion in FY2025. The company plans to expand its "
            "data center footprint in Europe and Asia-Pacific to support growing "
            "international demand."
        ),
        metadata={"source": "TechCorp-10K-2024.pdf", "page": 14},
    ),
]

# SQLite data intentionally has slight discrepancies vs the 10-K text above
# to demonstrate discrepancy detection (the killer feature).
# Discrepancies: Q4 net income is $325M in DB vs $330M in doc,
# Professional Services revenue is $885M vs $890M in doc.
DEMO_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS financials (
    year INTEGER PRIMARY KEY,
    revenue_m REAL NOT NULL,
    cost_of_revenue_m REAL NOT NULL,
    operating_expenses_m REAL NOT NULL,
    operating_income_m REAL NOT NULL,
    net_income_m REAL NOT NULL,
    eps_diluted REAL,
    free_cash_flow_m REAL
);

CREATE TABLE IF NOT EXISTS quarterly (
    quarter TEXT PRIMARY KEY,
    revenue_m REAL NOT NULL,
    net_income_m REAL NOT NULL,
    operating_margin_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    segment_name TEXT PRIMARY KEY,
    revenue_m REAL NOT NULL,
    pct_of_total REAL NOT NULL,
    yoy_growth_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS balance_sheet (
    year INTEGER PRIMARY KEY,
    total_assets_m REAL NOT NULL,
    cash_equivalents_m REAL NOT NULL,
    total_debt_m REAL NOT NULL,
    stockholders_equity_m REAL NOT NULL,
    goodwill_intangibles_m REAL NOT NULL
);
"""

DEMO_SQL_DATA = """
INSERT INTO financials VALUES (2022, 3700, 1554, 1480, 666, 555, 2.22, 740);
INSERT INTO financials VALUES (2023, 4400, 1848, 1628, 924, 792, 3.17, 1050);
INSERT INTO financials VALUES (2024, 5200, 2080, 1820, 1300, 1140, 4.56, 1380);

INSERT INTO quarterly VALUES ('Q1 2024', 1180, 240, 23.0);
INSERT INTO quarterly VALUES ('Q2 2024', 1260, 275, 24.0);
INSERT INTO quarterly VALUES ('Q3 2024', 1320, 295, 25.0);
INSERT INTO quarterly VALUES ('Q4 2024', 1440, 325, 27.0);

INSERT INTO segments VALUES ('AI Analytics', 2100, 40.4, 32.0);
INSERT INTO segments VALUES ('Cloud Infrastructure', 1560, 30.0, 14.0);
INSERT INTO segments VALUES ('Professional Services', 885, 17.0, 8.0);
INSERT INTO segments VALUES ('Data Integration', 650, 12.5, 11.0);

INSERT INTO balance_sheet VALUES (2022, 8200, 2100, 2200, 3800, 2800);
INSERT INTO balance_sheet VALUES (2023, 10500, 2800, 2000, 5200, 3500);
INSERT INTO balance_sheet VALUES (2024, 12800, 3200, 1800, 6400, 4100);
"""

DEMO_SUGGESTED_QUESTIONS = [
    "What was TechCorp's total revenue and net income in 2024?",
    "Compare the quarterly performance — which quarter was strongest?",
    "Are there any discrepancies between the filing and the database?",
    "What are the main risk factors and how might they impact FY2025 guidance?",
]


# ---------------------------------------------------------------------------
# Demo initialization
# ---------------------------------------------------------------------------


def _create_demo_sqlite() -> Path:
    """Create a temporary SQLite database with sample financial data."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".sqlite", prefix="docbot_demo_", delete=False
    )
    tmp.close()
    db_path = Path(tmp.name)

    conn = sqlite3.connect(str(db_path))
    conn.executescript(DEMO_SQL_SCHEMA)
    conn.executescript(DEMO_SQL_DATA)
    conn.close()

    logger.info("demo: created SQLite at %s", db_path)
    return db_path


async def init_demo_session(
    vector_stores: Dict[str, Any],
    get_embeddings_fn,
    sessions_table: Table,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
) -> Dict[str, Any]:
    """
    Initialize a fresh demo session with sample document + database.

    Returns dict with session_id, connection_id, and suggested questions.
    """
    session_id = f"demo-{uuid.uuid4()}"
    connection_id = f"demo-{uuid.uuid4()}"

    # 1. Create vector store from hardcoded document chunks
    embeddings = get_embeddings_fn()
    store = create_store(session_id, DEMO_DOCUMENT_CHUNKS, embeddings)
    vector_stores[session_id] = store
    logger.info("demo: created vector store for session %s (%d chunks)", session_id, len(DEMO_DOCUMENT_CHUNKS))

    # 2. Insert session record
    files_info = json.dumps([{
        "filename": "TechCorp-10K-2024.pdf",
        "pages": 15,
        "size": 2_048_000,
    }])
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(sessions_table).values(
                    session_id=session_id,
                    persona="Finance Expert",
                    file_count=1,
                    files_info=files_info,
                    source="demo",
                )
            )

    # 3. Create SQLite database with matching financial data
    db_path = _create_demo_sqlite()

    # 4. Register as db_connection (dbname must be the actual file path for SQLite)
    creds_blob = encrypt_credentials({
        "dialect": "sqlite",
        "host": "__local_file__",
        "port": 0,
        "dbname": str(db_path),
        "user": "",
        "password": "",
        "original_filename": "TechCorp-Financials.db",
    })

    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(db_connections_table).values(
                    id=connection_id,
                    session_id=session_id,
                    dialect="sqlite",
                    host="__local_file__",
                    port=0,
                    dbname=str(db_path),
                    credentials_blob=creds_blob,
                )
            )

    # 5. Introspect and cache schema
    from api.db_service import get_schema
    schema = await get_schema(
        connection_id, db_connections_table, schema_cache_table, async_session_factory
    )

    logger.info(
        "demo: initialized session=%s connection=%s tables=%d",
        session_id, connection_id, len(schema),
    )

    return {
        "session_id": session_id,
        "connection_id": connection_id,
        "company": DEMO_COMPANY,
        "filing": DEMO_FILING,
        "schema_summary": {
            "table_count": len(schema),
            "tables": [t["name"] for t in schema],
        },
        "suggested_questions": DEMO_SUGGESTED_QUESTIONS,
    }
