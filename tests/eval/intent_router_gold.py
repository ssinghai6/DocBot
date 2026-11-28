"""DOCBOT-2730: gold set for the intent router (api/intent_service.py).

60 labelled questions, each with the source summary the router would see.
Categories: simple lookups, simple viz, keyword-free paraphrased forecasts,
compares, what-ifs, diagnoses, and docs-only forecasts whose sources hold no
series (``data_ready=False``).

Scoring (``score``):
* route accuracy: predicted ``route`` == gold ``route`` over all 60 (gate >= 90%).
* false-autopilot rate: share of ``lookup`` / ``viz`` items routed to autopilot
  (gate <= 5%).
* data-ready accuracy on forecast / what_if / diagnose items (reported only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from api.intent_models import RouteIntentResponse

# (summary, has_doc, has_db)
Src = tuple[str, bool, bool]

TENK: Src = ("Documents: acme_corp_10k_fy2024.pdf", True, False)
POLICY: Src = ("Documents: employee_handbook_2024.pdf, travel_policy.pdf", True, False)
FUEL_DOC: Src = ("Documents: fuelprice_article.pdf", True, False)
CONTRACT: Src = ("Documents: master_services_agreement.pdf", True, False)
SALES_CSV: Src = (
    "CSV file: data(order_date TEXT, region TEXT, product TEXT, units TEXT, revenue TEXT, cac TEXT)", False, True,
)
FUEL_CSV: Src = ("CSV file: data(year TEXT, diesel_price_usd TEXT, gasoline_price_usd TEXT)", False, True)
LEDGER_DB: Src = (
    "postgresql database: trial_balance(period DATE, account TEXT, amount NUMERIC); "
    "revenue_by_segment(period DATE, segment TEXT, amount NUMERIC); "
    "opex(month DATE, department TEXT, amount NUMERIC)",
    False, True,
)
HYBRID: Src = (
    "Documents: tidewell_draft_10k_fy2025.pdf\n"
    "postgresql database: revenue_by_segment(period DATE, segment TEXT, amount NUMERIC); "
    "trial_balance(period DATE, account TEXT, amount NUMERIC)",
    True, True,
)


@dataclass(frozen=True)
class GoldIntent:
    question: str
    src: Src
    route: str
    task: str
    data_ready: bool = True


def _g(q: str, src: Src, route: str, task: str, ready: bool = True) -> GoldIntent:
    return GoldIntent(q, src, route, task, ready)


GOLD_INTENTS: list[GoldIntent] = [
    # ── simple lookups (12) ──
    _g("What was total revenue in fiscal 2024?", TENK, "fast", "lookup"),
    _g("Who is the CEO according to the filing?", TENK, "fast", "lookup"),
    _g("How many vacation days do new employees get?", POLICY, "fast", "lookup"),
    _g("What is the per diem for international travel?", POLICY, "fast", "lookup"),
    _g("When does the services agreement terminate?", CONTRACT, "fast", "lookup"),
    _g("What is the governing law in the contract?", CONTRACT, "fast", "lookup"),
    _g("How many orders are in the file?", SALES_CSV, "fast", "lookup"),
    _g("Which region had the highest revenue?", SALES_CSV, "fast", "lookup"),
    _g("Total opex for March 2025?", LEDGER_DB, "fast", "lookup"),
    _g("List the accounts in the trial balance", LEDGER_DB, "fast", "lookup"),
    _g("What does the 10-K say about risk factors?", HYBRID, "fast", "lookup"),
    _g("What was the diesel price in 2019?", FUEL_CSV, "fast", "lookup"),
    # ── simple viz (8) ──
    _g("Plot revenue by region", SALES_CSV, "fast", "viz"),
    _g("Show a bar chart of units by product", SALES_CSV, "fast", "viz"),
    _g("Chart diesel prices over time", FUEL_CSV, "fast", "viz"),
    _g("Histogram of order revenue", SALES_CSV, "fast", "viz"),
    _g("Graph monthly opex by department", LEDGER_DB, "fast", "viz"),
    _g("Can you draw segment revenue as a pie chart?", LEDGER_DB, "fast", "viz"),
    _g("Visualize gasoline vs diesel prices by year", FUEL_CSV, "fast", "viz"),
    _g("Show me revenue by segment", HYBRID, "fast", "viz"),
    # ── keyword-free paraphrased forecasts, data present (14) ──
    _g("Where will diesel be in 2030?", FUEL_CSV, "autopilot", "forecast"),
    _g("What will gasoline cost five years from now?", FUEL_CSV, "autopilot", "forecast"),
    _g("How much revenue should we expect next quarter?", SALES_CSV, "autopilot", "forecast"),
    _g("Project units sold for the rest of the year", SALES_CSV, "autopilot", "forecast"),
    _g("Where is opex heading over the next 12 months?", LEDGER_DB, "autopilot", "forecast"),
    _g("Extrapolate segment revenue to FY2027", LEDGER_DB, "autopilot", "forecast"),
    _g("If the trend holds, what does revenue look like in 2026?", SALES_CSV, "autopilot", "forecast"),
    _g("Estimate next year's diesel price", FUEL_CSV, "autopilot", "forecast"),
    _g("What's our run-rate revenue going to be by December?", SALES_CSV, "autopilot", "forecast"),
    _g("Give me the expected Lab Services revenue for next fiscal year", HYBRID, "autopilot", "forecast"),
    _g("How high could fuel prices climb by the end of the decade?", FUEL_CSV, "autopilot", "forecast"),
    _g("Model the next 10 years of diesel prices", FUEL_CSV, "autopilot", "forecast"),
    _g("When will monthly opex cross $2M at this pace?", LEDGER_DB, "autopilot", "forecast"),
    _g("Based on history, what will Q1 revenue be in each region?", SALES_CSV, "autopilot", "forecast"),
    # ── compares needing several queries / sources (6) ──
    _g("Compare FY2025 segment revenue in the 10-K against the books", HYBRID, "autopilot", "compare"),
    _g("Compare revenue growth in the East vs West region over the last two years", SALES_CSV, "autopilot", "compare"),
    _g("Does the trial balance agree with the revenue reported in the filing?", HYBRID, "autopilot", "compare"),
    _g("How does diesel price movement correlate with gasoline?", FUEL_CSV, "autopilot", "compare"),
    _g("Compare opex by department this year versus last year and explain the gaps", LEDGER_DB, "autopilot", "compare"),
    _g("Tie out every segment in the draft 10-K to the database", HYBRID, "autopilot", "compare"),
    # ── what-ifs (8) ──
    _g("What happens to margin if CAC rises 20%?", SALES_CSV, "autopilot", "what_if"),
    _g("Suppose we cut prices by 10%, how does revenue change?", SALES_CSV, "autopilot", "what_if"),
    _g("If diesel jumps 30%, what's the impact on gasoline prices?", FUEL_CSV, "autopilot", "what_if"),
    _g("How would net income move if opex dropped 15%?", LEDGER_DB, "autopilot", "what_if"),
    _g("Assume Lab Services grows 5% faster, what's total revenue?", HYBRID, "autopilot", "what_if"),
    _g("If we stopped selling the lowest-margin product, what would revenue be?", SALES_CSV, "autopilot", "what_if"),
    _g("Run a scenario where the West region doubles its units", SALES_CSV, "autopilot", "what_if"),
    _g("Imagine headcount costs rise 8%; how much does opex grow?", LEDGER_DB, "autopilot", "what_if"),
    # ── diagnoses (6) ──
    _g("Why did revenue drop in March?", SALES_CSV, "autopilot", "diagnose"),
    _g("What drove the opex spike in Q3?", LEDGER_DB, "autopilot", "diagnose"),
    _g("Explain why Lab Services underperformed versus the filing", HYBRID, "autopilot", "diagnose"),
    _g("Which orders look unusual compared with the rest?", SALES_CSV, "autopilot", "diagnose"),
    _g("Find the root cause of the margin decline", LEDGER_DB, "autopilot", "diagnose"),
    _g("What explains the jump in diesel prices in 2022?", FUEL_CSV, "autopilot", "diagnose"),
    # ── docs-only forecasts with no series (6) ──
    _g("Predict diesel prices for the next 10 years", FUEL_DOC, "autopilot", "forecast", False),
    _g("Where will fuel prices be by 2030?", FUEL_DOC, "autopilot", "forecast", False),
    _g("Forecast travel spend next year", POLICY, "autopilot", "forecast", False),
    _g("How many vacation days will be used next year?", POLICY, "autopilot", "forecast", False),
    _g("Project contract renewals for the next three years", CONTRACT, "autopilot", "forecast", False),
    _g("What will the services fee be in 2030 if it keeps growing?", CONTRACT, "autopilot", "forecast", False),
]

SIMPLE_TASKS = ("lookup", "viz")
GATED_TASKS = ("forecast", "what_if", "diagnose")


@dataclass(frozen=True)
class Score:
    route_accuracy: float
    false_autopilot_simple: float
    data_ready_accuracy: float


def score(predict: Callable[[GoldIntent], RouteIntentResponse]) -> Score:
    preds = [(g, predict(g)) for g in GOLD_INTENTS]
    simple = [(g, p) for g, p in preds if g.task in SIMPLE_TASKS]
    gated = [(g, p) for g, p in preds if g.task in GATED_TASKS]
    return Score(
        route_accuracy=sum(g.route == p.route for g, p in preds) / len(preds),
        false_autopilot_simple=sum(p.route == "autopilot" for _, p in simple) / len(simple),
        data_ready_accuracy=sum(g.data_ready == p.data_ready for g, p in gated) / len(gated),
    )
