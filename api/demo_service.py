"""
DocBot Demo Mode — Sandbox with pre-loaded financial data.

Provides a /api/demo/init endpoint that creates:
  1. A RAG session with hardcoded QuickBite 10-K financial text chunks
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
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from sqlalchemy import Table, insert

from api.utils.encryption import encrypt_credentials
from api.utils.vector_store import create_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample financial data — QuickBite Inc. (fictitious)
# ---------------------------------------------------------------------------

DEMO_COMPANY = "QuickBite Inc."
DEMO_TICKER = "QBIT"
DEMO_FILING = "Annual Report (10-K) — Fiscal Year 2025"

# These chunks simulate what a real 10-K PDF extraction would produce.
# They contain deliberate discrepancies with the DB data for demo purposes.
DEMO_DOCUMENT_CHUNKS = [
    Document(
        page_content=(
            "QuickBite Inc. — Annual Report for the Fiscal Year Ended December 31, 2025.\n\n"
            "PART I — BUSINESS OVERVIEW\n\n"
            "QuickBite Inc. is an online food-delivery marketplace that connects hungry "
            "customers with local restaurants and grocery stores through its mobile app. "
            "Founded in 2015 and headquartered in San Francisco, QuickBite operates in over "
            "45 cities, works with 320,000 restaurant partners, and fulfills orders through a "
            "network of independent delivery couriers. The company also offers QuickBite+, a "
            "paid subscription with free delivery and member-only discounts."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "PART II — FINANCIAL HIGHLIGHTS\n\n"
            "Total Revenue: $5.2 billion (up 18% year-over-year from $4.4 billion in FY2024)\n"
            "Cost of Revenue: $2.08 billion (gross margin: 60%)\n"
            "Operating Expenses: $1.82 billion\n"
            "Operating Income: $1.30 billion (operating margin: 25%)\n"
            "Net Income: $1.14 billion (net margin: 21.9%)\n"
            "Earnings Per Share (diluted): $4.56\n"
            "Free Cash Flow: $1.38 billion\n\n"
            "The company achieved record revenue driven by strong growth in QuickBite+ "
            "subscriptions, which grew 32% year-over-year to $1.56 billion, and a 21% "
            "increase in total orders to 640 million for the year."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 3},
    ),
    Document(
        page_content=(
            "REVENUE BY SEGMENT — FY2025\n\n"
            "Delivery Fees: $2,100M (40.4% of total revenue, up 16% YoY)\n"
            "QuickBite+ Subscriptions: $1,560M (30.0%, up 32% YoY)\n"
            "Restaurant Advertising: $890M (17.1%, up 8% YoY)\n"
            "Grocery Delivery: $650M (12.5%, up 27% YoY)\n\n"
            "QuickBite+ Subscriptions was the fastest growing segment, driven by new member "
            "sign-ups and higher renewal rates. Average orders per subscriber rose to 4.8 per "
            "month from 3.9 the prior year."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 5},
    ),
    Document(
        page_content=(
            "QUARTERLY PERFORMANCE — FY2025\n\n"
            "Q1 2025: Revenue $1,180M, Net Income $240M, Operating Margin 23%\n"
            "Q2 2025: Revenue $1,260M, Net Income $275M, Operating Margin 24%\n"
            "Q3 2025: Revenue $1,320M, Net Income $295M, Operating Margin 25%\n"
            "Q4 2025: Revenue $1,440M, Net Income $330M, Operating Margin 27%\n\n"
            "The company demonstrated consistent sequential growth throughout the year, "
            "with Q4 the strongest quarter driven by holiday-season ordering and a surge in "
            "grocery-delivery volume."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 7},
    ),
    Document(
        page_content=(
            "BALANCE SHEET HIGHLIGHTS — December 31, 2025\n\n"
            "Total Assets: $12.8 billion\n"
            "Cash and Equivalents: $3.2 billion\n"
            "Total Debt: $1.8 billion (debt-to-equity ratio: 0.28)\n"
            "Stockholders' Equity: $6.4 billion\n"
            "Goodwill and Intangibles: $4.1 billion\n\n"
            "The company maintained a strong balance sheet with a net cash position "
            "of $1.4 billion. During FY2025, QuickBite completed two strategic acquisitions "
            "totaling $1.1 billion to expand into grocery delivery and new regional markets."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 9},
    ),
    Document(
        page_content=(
            "RISK FACTORS\n\n"
            "1. Competition: The food-delivery market is highly competitive with established "
            "players (DoorDash, Uber Eats, Grubhub) competing aggressively on price and selection.\n"
            "2. Courier Classification: Regulations reclassifying independent couriers as "
            "employees could materially increase our delivery costs.\n"
            "3. Restaurant Concentration: Our top 10 restaurant partners represent 22% of total "
            "orders. Loss of a major partner could impact results.\n"
            "4. Delivery-Fee Regulation: Some cities have capped the commissions we can charge "
            "restaurants, which may limit revenue in those markets.\n"
            "5. Foreign Exchange: 35% of revenue is denominated in non-USD currencies, "
            "exposing the company to currency fluctuation risks."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 12},
    ),
    Document(
        page_content=(
            "GUIDANCE — FY2026\n\n"
            "Revenue: $6.0–6.2 billion (15–19% growth)\n"
            "Operating Margin: 25–27%\n"
            "Free Cash Flow: $1.5–1.7 billion\n"
            "Capital Expenditures: $800M–900M (primarily grocery fulfillment and app investment)\n\n"
            "Management expects continued strong demand for QuickBite+, with subscription "
            "revenue projected to reach $2.1 billion in FY2026. The company plans to expand "
            "grocery delivery into new cities and grow its international presence in Europe "
            "and Asia-Pacific."
        ),
        metadata={"source": "QuickBite-10K-2025.pdf", "page": 14},
    ),
]

# SQLite data intentionally has slight discrepancies vs the 10-K text above
# to demonstrate discrepancy detection (the killer feature).
# Discrepancies: Q4 net income is $325M in DB vs $330M in doc,
# Restaurant Advertising revenue is $885M vs $890M in doc.
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
INSERT INTO financials VALUES (2023, 3700, 1554, 1480, 666, 555, 2.22, 740);
INSERT INTO financials VALUES (2024, 4400, 1848, 1628, 924, 792, 3.17, 1050);
INSERT INTO financials VALUES (2025, 5200, 2080, 1820, 1300, 1140, 4.56, 1380);

INSERT INTO quarterly VALUES ('Q1 2025', 1180, 240, 23.0);
INSERT INTO quarterly VALUES ('Q2 2025', 1260, 275, 24.0);
INSERT INTO quarterly VALUES ('Q3 2025', 1320, 295, 25.0);
INSERT INTO quarterly VALUES ('Q4 2025', 1440, 325, 27.0);

INSERT INTO segments VALUES ('Delivery Fees', 2100, 40.4, 16.0);
INSERT INTO segments VALUES ('QuickBite+ Subscriptions', 1560, 30.0, 32.0);
INSERT INTO segments VALUES ('Restaurant Advertising', 885, 17.0, 8.0);
INSERT INTO segments VALUES ('Grocery Delivery', 650, 12.5, 27.0);

INSERT INTO balance_sheet VALUES (2023, 8200, 2100, 2200, 3800, 2800);
INSERT INTO balance_sheet VALUES (2024, 10500, 2800, 2000, 5200, 3500);
INSERT INTO balance_sheet VALUES (2025, 12800, 3200, 1800, 6400, 4100);
"""

DEMO_SUGGESTED_QUESTIONS = [
    "What was QuickBite's total revenue and net income in 2025?",
    "Compare the quarterly performance — which quarter was strongest?",
    "Are there any discrepancies between the filing and the database?",
    "What are the main risk factors and how might they impact FY2026 guidance?",
]


# ---------------------------------------------------------------------------
# Second dataset — Fuel Prices (real policy PDF + 1970–2026 crude oil series)
# Showcases real forecasting (E2B model on 675 monthly points) + document RAG
# with citations. Source: VTPI "Appropriate Response to Rising Fuel Prices".
# ---------------------------------------------------------------------------

_FUEL_CSV_PATH = Path(__file__).parent / "demo_data" / "fuel_prices_1970_2026.csv"

FUEL_COMPANY = "Crude Oil Prices (1970–2026)"
FUEL_FILING = "VTPI Report — Appropriate Response to Rising Fuel Prices"

FUEL_DOCUMENT_CHUNKS = [
    Document(
        page_content=(
            "Appropriate Response to Rising Fuel Prices — Victoria Transport Policy "
            "Institute (Todd Litman, 19 April 2026).\n\n"
            "SUMMARY\n"
            "This paper evaluates policy options for responding to rising fuel prices. "
            "There is popular support for policies that minimize fuel prices through "
            "subsidies and tax reductions, but such policies harm consumers and the "
            "economy overall because they increase total fuel consumption and vehicle "
            "travel, and therefore associated costs such as congestion, infrastructure "
            "costs, crashes, trade imbalances and pollution emissions."
        ),
        metadata={"source": "fuelprice.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "INTRODUCTION\n"
            "Motor vehicle fuel prices have increased significantly in recent years and "
            "are likely to stay high in the future. Between 2003 and 2008 average U.S. "
            "gasoline retail prices more than doubled, from $1.77 to $4.10 per gallon, and "
            "high prices are expected to continue due to growing international demand and "
            "rising production costs. Fuel prices are an emotional issue; motorists often "
            "feel they pay more than is fair and demand price-minimization policies."
        ),
        metadata={"source": "fuelprice.pdf", "page": 2},
    ),
    Document(
        page_content=(
            "FUEL COST TRENDS\n"
            "Current North American fuel prices are relatively low by most standards. U.S. "
            "and Canada fuel prices are lower than most other high-income countries. Norway "
            "and the UK are notable: during recent decades these countries were major "
            "petroleum producers, yet retained high fuel prices as a strategic policy to "
            "encourage energy efficiency. Inflation-adjusted fuel costs per vehicle-mile "
            "declined over recent decades as manufacturers built more efficient vehicles."
        ),
        metadata={"source": "fuelprice.pdf", "page": 3},
    ),
    Document(
        page_content=(
            "FUEL PRICE IMPACTS ON ENERGY CONSUMPTION AND TRAVEL\n"
            "Studies of the price elasticity of fuel indicate that over the long run a 10% "
            "fuel price increase typically causes: a 4–6% reduction in total vehicle fuel "
            "consumption; a 3–4% increase in fuel efficiency; and a 1–3% reduction in "
            "vehicle mileage. High fuel prices are associated with lower per-capita "
            "transportation energy consumption across countries."
        ),
        metadata={"source": "fuelprice.pdf", "page": 6},
    ),
    Document(
        page_content=(
            "PROBLEM DEFINITIONS AND POTENTIAL SOLUTIONS\n"
            "The problems of rising fuel prices can be defined in several ways: fuel "
            "unaffordability (fuel too expensive for lower-income motorists), transportation "
            "unaffordability (total travel too expensive), energy dependence (economic risk "
            "of importing petroleum), and vehicle fuel inefficiency. Each definition points "
            "to different solutions — subsidies address affordability but worsen dependence "
            "and inefficiency."
        ),
        metadata={"source": "fuelprice.pdf", "page": 9},
    ),
    Document(
        page_content=(
            "ECONOMIC IMPACTS\n"
            "People often assume low fuel prices support economic development and that fuel "
            "tax increases reduce economic activity, but this is not necessarily true. "
            "Reducing fuel prices through subsidies tends to be economically harmful because "
            "it imposes costs elsewhere and increases energy consumption. Many of the most "
            "economically successful countries (Japan, Germany, Scandinavia) have high fuel "
            "prices, while economies with low fuel prices are not especially productive."
        ),
        metadata={"source": "fuelprice.pdf", "page": 14},
    ),
    Document(
        page_content=(
            "EQUITY IMPACTS\n"
            "Horizontal equity asks whether people with equal needs are treated equally; "
            "subsidizing fuel violates it by benefiting high energy-consuming consumers at "
            "the expense of efficient ones. Vertical equity holds that disadvantaged people "
            "should receive extra support — better achieved through targeted assistance and "
            "affordable transport options than through broad fuel subsidies."
        ),
        metadata={"source": "fuelprice.pdf", "page": 19},
    ),
    Document(
        page_content=(
            "CONCLUSIONS: RAISE MY FUEL PRICES, PLEASE!\n"
            "Fuel prices are likely to increase in the future. Rather than trying to minimize "
            "fuel prices — which requires subsidies and increases consumption, travel, "
            "pollution, congestion and sprawl — it is better to allow prices to rise and help "
            "consumers, businesses and communities reduce total fuel costs by increasing "
            "vehicle and transport-system efficiency."
        ),
        metadata={"source": "fuelprice.pdf", "page": 23},
    ),
]

FUEL_SUGGESTED_QUESTIONS = [
    "Forecast crude oil prices for the next 12 months",
    "What policies does the report recommend for responding to rising fuel prices?",
    "How have crude oil prices changed since 1970, and what does the report say about the effects?",
    "How do rising fuel prices affect consumer behavior and the environment?",
]


# ---------------------------------------------------------------------------
# Third dataset — Tie-out (DOCBOT-2212): draft 10-K vs trial balance
# Tidewell Instruments, Inc. is entirely fictional. The draft report's primary
# income statement ties to the books; three figures in the MD&A and segment
# note do not. Every other figure in the draft ties exactly.
#
# PLANTED DISCREPANCIES (report vs books, FY2025, USD):
#   | # | Figure                         | Where in draft         | Report       | Books        | Delta      | Delta % |
#   |---|--------------------------------|------------------------|--------------|--------------|------------|---------|
#   | 1 | Total revenue                  | MD&A (p.4)             | 418,200,000  | 412,600,000  | -5,600,000 | -1.34%  |
#   | 2 | Lab Services segment revenue   | Segment note (p.9)     | 104,900,000  | 102,700,000  | -2,200,000 | -2.10%  |
#   | 3 | Sales and marketing expense    | MD&A (p.5)             |  54,900,000  |  55,200,000  |   +300,000 | +0.55%  |
# #3 is the small-but-material one: $0.3M, just above the 0.5% tie-out band.
# ---------------------------------------------------------------------------

TIEOUT_COMPANY = "Tidewell Instruments, Inc. (fictional)"
TIEOUT_FILING = "DRAFT Annual Report (10-K) — Fiscal Year 2025"
_TIEOUT_DOC = "Tidewell-Draft-10K-FY2025.pdf"

# Expected discrepancies: label -> (report value, books value). Tests assert the
# detector flags exactly these and nothing else.
TIEOUT_EXPECTED_DISCREPANCIES: Dict[str, tuple[float, float]] = {
    "Total revenue": (418_200_000, 412_600_000),
    "Lab Services": (104_900_000, 102_700_000),
    "Sales and marketing expense": (54_900_000, 55_200_000),
}

TIEOUT_DOCUMENT_CHUNKS = [
    Document(
        page_content=(
            "Tidewell Instruments, Inc. (a fictional company) — DRAFT Annual Report on "
            "Form 10-K for the Fiscal Year Ended December 31, 2025. DRAFT — NOT FOR "
            "DISTRIBUTION. Subject to completion of the year-end close.\n\n"
            "PART I — BUSINESS\n\n"
            "Tidewell designs and sells precision measurement instruments to industrial and "
            "research laboratories, operates calibration and testing labs, and licenses data "
            "analytics software. The company reports three segments: Commercial Instruments, "
            "Lab Services, and Software and Analytics."
        ),
        metadata={"source": _TIEOUT_DOC, "page": 1},
    ),
    Document(
        page_content=(
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — RESULTS OF OPERATIONS FY2025\n\n"
            "Total revenue: $418.2 million, driven by instrument shipments to "
            "semiconductor customers and higher calibration volumes.\n"
            "Gross profit: $241.3 million.\n"
            "Operating income: $109.0 million.\n"
            "Net income: $82.5 million."
        ),
        metadata={"source": _TIEOUT_DOC, "page": 4},
    ),
    Document(
        page_content=(
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — OPERATING EXPENSES FY2025\n\n"
            "Research and development expense: $41.2 million, reflecting continued "
            "investment in the next-generation spectrometer platform.\n"
            "Sales and marketing expense: $54.9 million.\n"
            "General and administrative expense: $35.9 million."
        ),
        metadata={"source": _TIEOUT_DOC, "page": 5},
    ),
    Document(
        page_content=(
            "CONSOLIDATED STATEMENT OF OPERATIONS — FY2025 (in millions)\n\n"
            "Revenue: $412.6M\n"
            "Cost of revenue: $171.3M\n"
            "Gross profit: $241.3M\n"
            "Research and development: $41.2M\n"
            "Sales and marketing: $55.2M\n"
            "General and administrative: $35.9M\n"
            "Total operating expenses: $132.3M\n"
            "Operating income: $109.0M\n"
            "Interest expense: $4.6M\n"
            "Income before income taxes: $104.4M\n"
            "Income tax expense: $21.9M\n"
            "Net income: $82.5M"
        ),
        metadata={"source": _TIEOUT_DOC, "page": 7},
    ),
    Document(
        page_content=(
            "NOTE 12 — SEGMENT INFORMATION — FY2025\n\n"
            "Segment revenue (in millions):\n"
            "Commercial Instruments: $248.4M\n"
            "Lab Services: $104.9M\n"
            "Software and Analytics: $61.5M\n\n"
            "Segments are managed by product line. Intersegment sales are not material."
        ),
        metadata={"source": _TIEOUT_DOC, "page": 9},
    ),
]

# Books: amounts in whole USD. trial_balance is the source of truth; the two
# detail tables agree with it; income_statement is a view rolled up from it.
TIEOUT_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_balance (
    account TEXT NOT NULL,
    account_type TEXT NOT NULL,
    period TEXT NOT NULL,
    debit REAL NOT NULL DEFAULT 0,
    credit REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (account, period)
);

CREATE TABLE IF NOT EXISTS revenue_by_segment (
    period TEXT NOT NULL,
    segment TEXT NOT NULL,
    amount REAL NOT NULL,
    PRIMARY KEY (period, segment)
);

CREATE TABLE IF NOT EXISTS opex (
    period TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    PRIMARY KEY (period, category)
);

CREATE VIEW IF NOT EXISTS income_statement AS
SELECT period, line_item, amount FROM (
    SELECT period, 1 AS ord, 'Revenue' AS line_item,
           SUM(CASE WHEN account_type = 'revenue' THEN credit - debit ELSE 0 END) AS amount
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 2, 'Cost of revenue',
           SUM(CASE WHEN account_type = 'cogs' THEN debit - credit ELSE 0 END)
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 3, 'Gross profit',
           SUM(CASE WHEN account_type = 'revenue' THEN credit - debit
                    WHEN account_type = 'cogs' THEN credit - debit ELSE 0 END)
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 4, 'Total operating expenses',
           SUM(CASE WHEN account_type = 'opex' THEN debit - credit ELSE 0 END)
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 5, 'Operating income',
           SUM(CASE WHEN account_type IN ('revenue', 'cogs', 'opex') THEN credit - debit ELSE 0 END)
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 6, 'Income before income taxes',
           SUM(CASE WHEN account_type IN ('revenue', 'cogs', 'opex', 'other') THEN credit - debit ELSE 0 END)
      FROM trial_balance GROUP BY period
    UNION ALL
    SELECT period, 7, 'Net income',
           SUM(CASE WHEN account_type IN ('revenue', 'cogs', 'opex', 'other', 'tax') THEN credit - debit ELSE 0 END)
      FROM trial_balance GROUP BY period
) ORDER BY period, ord;
"""

TIEOUT_SQL_DATA = """
INSERT INTO trial_balance VALUES ('1000 Cash', 'asset', 'FY2025', 120000000, 0);
INSERT INTO trial_balance VALUES ('1200 Accounts receivable', 'asset', 'FY2025', 58400000, 0);
INSERT INTO trial_balance VALUES ('1500 Property and equipment', 'asset', 'FY2025', 75000000, 0);
INSERT INTO trial_balance VALUES ('2000 Accounts payable', 'liability', 'FY2025', 0, 27600000);
INSERT INTO trial_balance VALUES ('3000 Common stock and APIC', 'equity', 'FY2025', 0, 50000000);
INSERT INTO trial_balance VALUES ('3100 Retained earnings (opening)', 'equity', 'FY2025', 0, 93300000);
INSERT INTO trial_balance VALUES ('4000 Revenue - Commercial Instruments', 'revenue', 'FY2025', 0, 248400000);
INSERT INTO trial_balance VALUES ('4100 Revenue - Lab Services', 'revenue', 'FY2025', 0, 102700000);
INSERT INTO trial_balance VALUES ('4200 Revenue - Software and Analytics', 'revenue', 'FY2025', 0, 61500000);
INSERT INTO trial_balance VALUES ('5000 Cost of revenue', 'cogs', 'FY2025', 171300000, 0);
INSERT INTO trial_balance VALUES ('6000 Research and development', 'opex', 'FY2025', 41200000, 0);
INSERT INTO trial_balance VALUES ('6100 Sales and marketing', 'opex', 'FY2025', 55200000, 0);
INSERT INTO trial_balance VALUES ('6200 General and administrative', 'opex', 'FY2025', 35900000, 0);
INSERT INTO trial_balance VALUES ('7000 Interest expense', 'other', 'FY2025', 4600000, 0);
INSERT INTO trial_balance VALUES ('8000 Income tax expense', 'tax', 'FY2025', 21900000, 0);

INSERT INTO revenue_by_segment VALUES ('FY2025', 'Commercial Instruments', 248400000);
INSERT INTO revenue_by_segment VALUES ('FY2025', 'Lab Services', 102700000);
INSERT INTO revenue_by_segment VALUES ('FY2025', 'Software and Analytics', 61500000);

INSERT INTO opex VALUES ('FY2025', 'Research and development', 41200000);
INSERT INTO opex VALUES ('FY2025', 'Sales and marketing', 55200000);
INSERT INTO opex VALUES ('FY2025', 'General and administrative', 35900000);
"""

# One figure per question: each reliably runs the books query and flags its
# planted mismatch on prod; broad "tie out everything" prompts often do not.
TIEOUT_SUGGESTED_QUESTIONS = [
    "Does total revenue in the draft 10-K agree with the books?",
    "Does segment revenue in Note 12 agree with the books?",
    "Does sales and marketing expense in the MD&A agree with the general ledger?",
]


# ---------------------------------------------------------------------------
# Demo initialization
# ---------------------------------------------------------------------------


def _create_demo_sqlite(
    schema: str = DEMO_SQL_SCHEMA, data: str = DEMO_SQL_DATA, prefix: str = "docbot_demo_"
) -> Path:
    """Create a temporary SQLite database from a schema + data script."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", prefix=prefix, delete=False)
    tmp.close()
    db_path = Path(tmp.name)

    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema)
    conn.executescript(data)
    conn.close()

    logger.info("demo: created SQLite at %s", db_path)
    return db_path


def _create_tieout_sqlite() -> Path:
    return _create_demo_sqlite(TIEOUT_SQL_SCHEMA, TIEOUT_SQL_DATA, "docbot_tieout_")


def _create_fuel_sqlite() -> Path:
    """Build a SQLite DB from the committed crude-oil price CSV (1970–2026)."""
    import csv as _csv

    tmp = tempfile.NamedTemporaryFile(
        suffix=".sqlite", prefix="docbot_fuel_", delete=False
    )
    tmp.close()
    db_path = Path(tmp.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS crude_prices ("
        "month TEXT PRIMARY KEY, crude_oil_price REAL NOT NULL)"
    )
    with _FUEL_CSV_PATH.open(newline="") as fh:
        reader = _csv.DictReader(fh)
        rows = [(r["Date"], float(r["Crude_Oil_Price"])) for r in reader]
    conn.executemany("INSERT OR REPLACE INTO crude_prices VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    logger.info("demo: created fuel SQLite at %s (%d rows)", db_path, len(rows))
    return db_path


# Dataset registry — each entry fully describes a one-click demo.
_DATASETS: Dict[str, Dict[str, Any]] = {
    "quickbite": {
        "chunks": DEMO_DOCUMENT_CHUNKS,
        "sqlite_builder": _create_demo_sqlite,
        "doc_filename": "QuickBite-10K-2025.pdf",
        "db_filename": "QuickBite-Financials.db",
        "persona": "Finance Expert",
        "company": DEMO_COMPANY,
        "filing": DEMO_FILING,
        "suggested_questions": DEMO_SUGGESTED_QUESTIONS,
    },
    "fuel": {
        "chunks": FUEL_DOCUMENT_CHUNKS,
        "sqlite_builder": _create_fuel_sqlite,
        "doc_filename": "fuelprice.pdf",
        "db_filename": "FuelPrices-1970-2026.db",
        "persona": "Data Analyst",
        "company": FUEL_COMPANY,
        "filing": FUEL_FILING,
        "suggested_questions": FUEL_SUGGESTED_QUESTIONS,
    },
    "tieout": {
        "chunks": TIEOUT_DOCUMENT_CHUNKS,
        "sqlite_builder": _create_tieout_sqlite,
        "doc_filename": _TIEOUT_DOC,
        "db_filename": "Tidewell-TrialBalance-FY2025.db",
        "persona": "Finance Expert",
        "company": TIEOUT_COMPANY,
        "filing": TIEOUT_FILING,
        "suggested_questions": TIEOUT_SUGGESTED_QUESTIONS,
    },
}


async def init_demo_session(
    vector_stores: Dict[str, Any],
    get_embeddings_fn,
    sessions_table: Table,
    db_connections_table: Table,
    schema_cache_table: Table,
    async_session_factory,
    dataset: str = "quickbite",
    owner: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Initialize a fresh demo session with a sample document + database.

    `dataset` selects which one-click demo to load:
      - "quickbite": synthetic food-delivery 10-K + DB with planted discrepancies
      - "fuel": VTPI fuel-price policy PDF + real 1970–2026 crude oil series
      - "tieout": fictional draft 10-K vs trial balance, 3 planted mismatches

    Returns dict with session_id, connection_id, and suggested questions.
    """
    cfg = _DATASETS.get(dataset, _DATASETS["quickbite"])

    session_id = f"demo-{uuid.uuid4()}"
    connection_id = f"demo-{uuid.uuid4()}"

    # 1. Create vector store from the dataset's document chunks
    embeddings = get_embeddings_fn()
    store = create_store(session_id, cfg["chunks"], embeddings)
    vector_stores[session_id] = store
    logger.info(
        "demo[%s]: created vector store for session %s (%d chunks)",
        dataset, session_id, len(cfg["chunks"]),
    )

    # 2. Insert session record
    files_info = json.dumps([{
        "filename": cfg["doc_filename"],
        "pages": 27 if dataset == "fuel" else 15,
        "size": 651_611 if dataset == "fuel" else 2_048_000,
    }])
    async with async_session_factory() as session:
        async with session.begin():
            await session.execute(
                insert(sessions_table).values(
                    session_id=session_id,
                    persona=cfg["persona"],
                    file_count=1,
                    files_info=files_info,
                    source="demo",
                    # DOCBOT-1601: owner columns from caller_scope.owner_values()
                    **(owner or {}),
                )
            )

    # 3. Build the dataset's SQLite database
    db_path = cfg["sqlite_builder"]()

    # 4. Register as db_connection (dbname must be the actual file path for SQLite)
    creds_blob = encrypt_credentials({
        "dialect": "sqlite",
        "host": "__local_file__",
        "port": 0,
        "dbname": str(db_path),
        "user": "",
        "password": "",
        "original_filename": cfg["db_filename"],
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
                    # DOCBOT-1635: the demo connection belongs to its creator
                    # (user/org, or the anonymous owner cookie digest).
                    **{k: v for k, v in (owner or {}).items() if k in db_connections_table.c},
                )
            )

    # 5. Introspect and cache schema
    from api.db_service import get_schema
    schema = await get_schema(
        connection_id, db_connections_table, schema_cache_table, async_session_factory
    )

    logger.info(
        "demo[%s]: initialized session=%s connection=%s tables=%d",
        dataset, session_id, connection_id, len(schema),
    )

    return {
        "session_id": session_id,
        "connection_id": connection_id,
        "dataset": dataset,
        "company": cfg["company"],
        "filing": cfg["filing"],
        "schema_summary": {
            "table_count": len(schema),
            "tables": [t["name"] for t in schema],
        },
        "suggested_questions": cfg["suggested_questions"],
    }
