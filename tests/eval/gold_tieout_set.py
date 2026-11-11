"""DOCBOT-2541: gold tie-out questions over the demo datasets.

Each question names the figures it ties out. Every figure carries the value the
document reports, the value the books hold (both in absolute USD), the page the
report figure is on, and whether a discrepancy MUST be flagged. Values come
straight from ``api/demo_service.py`` (``TIEOUT_EXPECTED_DISCREPANCIES`` and the
QuickBite planted gaps) -- see ``tests/unit/test_demo_tieout.py`` for the
independent recomputation of every Tidewell figure from the trial balance.

``books_sql`` is the canonical correct query for the books side. Offline mode
runs it instead of the LLM SQL generator; live mode ignores it.
"""

from __future__ import annotations

from dataclasses import dataclass

M = 1_000_000


@dataclass(frozen=True)
class GoldFigure:
    label: str
    report: float          # value stated in the document, USD
    books: float           # value in the database, USD
    page: int              # document page the report figure is on
    flag: bool             # True = a discrepancy MUST be flagged; False = must NOT be
    period: str | None = None


@dataclass(frozen=True)
class TieoutQuestion:
    id: str
    dataset: str           # api/demo_service dataset: "tieout" | "quickbite"
    question: str
    figures: tuple[GoldFigure, ...]
    books_sql: str
    note: str = ""


# Discrepancies the demo plants on purpose. A flag on any other metric is a
# false positive and fails the run.
PLANTED: dict[str, set[str]] = {
    "tieout": {"Total revenue", "Lab Services", "Sales and marketing expense"},
    "quickbite": {"Net Income", "Restaurant Advertising"},
}

_TB = "SELECT account, debit, credit FROM trial_balance WHERE period = 'FY2025' AND account LIKE '%{}'"
_IS = "SELECT line_item, amount FROM income_statement WHERE period = 'FY2025'"

GOLD_TIEOUT: list[TieoutQuestion] = [
    TieoutQuestion(
        id="segment_note12",
        dataset="tieout",
        question="Does segment revenue in Note 12 agree with the books?",
        figures=(
            GoldFigure("Lab Services", 104.9 * M, 102.7 * M, 9, True),
            GoldFigure("Commercial Instruments", 248.4 * M, 248.4 * M, 9, False),
            GoldFigure("Software and Analytics", 61.5 * M, 61.5 * M, 9, False),
        ),
        books_sql="SELECT segment, amount FROM revenue_by_segment WHERE period = 'FY2025'",
        note="The flaky demo question (DOCBOT-2542). Suggested question in ChatArea.tsx.",
    ),
    TieoutQuestion(
        id="total_revenue_mdna",
        dataset="tieout",
        question="Does total revenue in the draft 10-K agree with the books?",
        figures=(GoldFigure("Total revenue", 418.2 * M, 412.6 * M, 4, True),),
        books_sql=_IS,
    ),
    TieoutQuestion(
        id="sales_marketing_mdna",
        dataset="tieout",
        question="Does sales and marketing expense in the MD&A agree with the general ledger?",
        figures=(GoldFigure("Sales and marketing expense", 54.9 * M, 55.2 * M, 5, True),),
        books_sql="SELECT category, amount FROM opex WHERE period = 'FY2025'",
        note="Small-but-material: +0.55%, just above the 0.5% band.",
    ),
    TieoutQuestion(
        id="rnd_mdna",
        dataset="tieout",
        question="Does research and development expense in the MD&A agree with the books?",
        figures=(GoldFigure("Research and development expense", 41.2 * M, 41.2 * M, 5, False),),
        books_sql="SELECT category, amount FROM opex WHERE period = 'FY2025'",
    ),
    TieoutQuestion(
        id="gna_mdna",
        dataset="tieout",
        question="Does general and administrative expense in the MD&A agree with the books?",
        figures=(GoldFigure("General and administrative expense", 35.9 * M, 35.9 * M, 5, False),),
        books_sql="SELECT category, amount FROM opex WHERE period = 'FY2025'",
    ),
    TieoutQuestion(
        id="net_income_mdna",
        dataset="tieout",
        question="Does net income in the MD&A agree with the books?",
        figures=(GoldFigure("Net income", 82.5 * M, 82.5 * M, 4, False),),
        books_sql=_IS,
    ),
    TieoutQuestion(
        id="operating_income_mdna",
        dataset="tieout",
        question="Does operating income in the MD&A agree with the books?",
        figures=(GoldFigure("Operating income", 109.0 * M, 109.0 * M, 4, False),),
        books_sql=_IS,
    ),
    TieoutQuestion(
        id="gross_profit_mdna",
        dataset="tieout",
        question="Does gross profit in the MD&A agree with the books?",
        figures=(GoldFigure("Gross profit", 241.3 * M, 241.3 * M, 4, False),),
        books_sql=_IS,
    ),
    TieoutQuestion(
        id="cost_of_revenue_is",
        dataset="tieout",
        question="Does cost of revenue on the statement of operations agree with the trial balance?",
        figures=(GoldFigure("Cost of revenue", 171.3 * M, 171.3 * M, 7, False),),
        books_sql=_IS,
    ),
    TieoutQuestion(
        id="income_tax_is",
        dataset="tieout",
        question="Does income tax expense on the statement of operations agree with the trial balance?",
        figures=(GoldFigure("Income tax expense", 21.9 * M, 21.9 * M, 7, False),),
        books_sql=_TB.format("Income tax expense"),
    ),
    TieoutQuestion(
        id="q4_net_income",
        dataset="quickbite",
        question="Does the Q4 net income in the 10-K match the database?",
        figures=(GoldFigure("Net Income", 330 * M, 325 * M, 7, True, period="Q4 2025"),),
        books_sql="SELECT quarter, revenue_m, net_income_m FROM quarterly WHERE quarter = 'Q4 2025'",
    ),
    TieoutQuestion(
        id="restaurant_advertising",
        dataset="quickbite",
        question="Does Restaurant Advertising segment revenue in the 10-K match the database?",
        figures=(GoldFigure("Restaurant Advertising", 890 * M, 885 * M, 5, True),),
        books_sql="SELECT segment_name, revenue_m FROM segments",
    ),
]
