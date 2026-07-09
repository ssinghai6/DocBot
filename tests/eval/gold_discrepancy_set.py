"""Gold evaluation set for the hybrid discrepancy detector.

Each case pairs a document snippet with a SQL/DB result and the set of metric
labels where a genuine reporting discrepancy SHOULD be flagged. The detector is
pure code (no API keys), so this set scores precision / recall deterministically
in CI — real, citable numbers for the product's headline feature.

`expected` holds the canonical metric names on which a discrepancy is expected.
An empty set means "no discrepancy should be flagged" (a negative case — matching
values, or a unit/granularity mismatch that must NOT produce noise).
"""

from dataclasses import dataclass, field


@dataclass
class DiscrepancyCase:
    name: str
    doc_context: str
    sql_metadata: dict
    expected: set[str]  # canonical metric labels expected to be flagged
    note: str = ""


# The QuickBite demo data mirrors api/demo_service.py: the DB stores figures in
# millions while the 10-K text quotes them with units ("$330M", "$5.2 billion").
GOLD_CASES: list[DiscrepancyCase] = [
    # --- TRUE POSITIVES: real reporting discrepancies the demo plants ---------
    DiscrepancyCase(
        name="q4_net_income_330_vs_325",
        doc_context="Q4 2025: Revenue $1,440M, Net Income $330M, Operating Margin 27%",
        sql_metadata={"result_preview": [{"quarter": "Q4 2025", "net_income": 325}]},
        expected={"net income"},
        note="Planted: doc $330M vs DB 325M (~1.5%).",
    ),
    DiscrepancyCase(
        name="small_reporting_gap_opex",
        doc_context="Operating Expenses: $1,820M",
        sql_metadata={"result_preview": [{"operating_expenses": 1795}]},
        expected={"operating expenses"},
        note="~1.4% gap — a plausible reporting discrepancy.",
    ),

    # --- TRUE NEGATIVES: matching values, must NOT flag -----------------------
    DiscrepancyCase(
        name="q4_revenue_matches",
        doc_context="Q4 2025: Revenue $1,440M",
        sql_metadata={"result_preview": [{"quarter": "Q4 2025", "total_revenue": 1440}]},
        expected=set(),
        note="Doc $1,440M == DB 1440M after scale alignment. No discrepancy.",
    ),
    DiscrepancyCase(
        name="net_income_exact_match",
        doc_context="Net Income: $1,140M",
        sql_metadata={"result_preview": [{"net_income": 1140}]},
        expected=set(),
    ),

    # --- HARD NEGATIVES: unit / granularity mismatches must be SUPPRESSED ------
    DiscrepancyCase(
        name="unit_mismatch_billions_vs_millions",
        doc_context="Total Revenue: $5.2 billion",
        sql_metadata={"result_preview": [{"total_revenue": 5200}]},  # millions
        expected=set(),
        note="5.2e9 vs 5200M → same value after alignment. Must not flag.",
    ),
    DiscrepancyCase(
        name="annual_vs_quarterly_granularity",
        doc_context="Total Revenue: $5.2 billion",
        sql_metadata={"result_preview": [{"total_revenue": 1180}]},  # one quarter
        expected=set(),
        note="Annual doc figure vs quarterly DB row — huge delta, must suppress.",
    ),
    DiscrepancyCase(
        name="bad_parse_six_dollars",
        doc_context="Revenue guidance is $6.0-6.2 billion for FY2025",
        sql_metadata={"result_preview": [{"total_revenue": 1180}]},
        expected=set(),
        note="Guidance range, not a reported figure — must not produce +19566%.",
    ),
    DiscrepancyCase(
        name="unrelated_labels",
        doc_context="Employee headcount: 4,200",
        sql_metadata={"result_preview": [{"total_revenue": 5200}]},
        expected=set(),
        note="Different metrics — label similarity must reject.",
    ),
]
