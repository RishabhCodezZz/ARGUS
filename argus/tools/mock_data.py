"""Mock data tools — load hand-authored JSON from data/ instead of hitting
real filings/market/news APIs (PLAN.md: "mock data is a feature, not a
shortcut"). Each gather specialist (Stage 1) gets exactly one of these.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_COMPANY_SLUGS = {
    "acme corp": "acme_corp",
    "globex": "globex",
}


def _available_companies() -> list[str]:
    return ["Acme Corp", "Globex"]


def _load(name: str, filename: str) -> dict:
    slug = _COMPANY_SLUGS.get(name.strip().lower())
    if slug is None:
        return {
            "error": f"No mock data for '{name}'.",
            "available_companies": _available_companies(),
        }
    path = _DATA_DIR / slug / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_company_snapshot(name: str) -> dict:
    """Return a basic company snapshot (sector, size, latest-year financials).

    Use this whenever the user asks about a specific company by name, e.g.
    "tell me about Acme Corp" or "what does Globex do". The lookup is
    case-insensitive. Only Acme Corp and Globex exist in this mock dataset.

    Args:
        name: The company name as mentioned by the user.

    Returns:
        A dict with company details on success, or a dict with an "error"
        key naming the companies that are actually available if not found.
    """
    filings = _load(name, "filings.json")
    if "error" in filings:
        return filings
    latest = filings["fiscal_years"][-1]
    return {
        "name": filings["company"],
        "sector": filings["sector"],
        "founded": filings["founded"],
        "employees": filings["employees"],
        "headquarters": filings["headquarters"],
        "latest_fiscal_year": latest["year"],
        "revenue_usd_millions": latest["revenue_usd_millions"],
        "net_income_usd_millions": latest["net_income_usd_millions"],
    }


def get_filings(name: str) -> dict:
    """Return structured filings data: company profile plus 5 years of
    revenue, net income, and total debt (in USD millions).

    Use this to answer questions about a company's financial history,
    profitability trend, or balance sheet. Case-insensitive lookup; only
    Acme Corp and Globex exist in this mock dataset.

    Args:
        name: The company name.

    Returns:
        A dict with "company", "sector", "founded", "employees",
        "headquarters", and "fiscal_years" (a list of yearly records), or
        an "error" dict if the company isn't in the mock dataset.
    """
    return _load(name, "filings.json")


def get_market_data(name: str) -> dict:
    """Return a monthly closing-price time series for the company's stock
    over the most recent fiscal year.

    Use this for questions about price trend, volatility, or returns.
    Case-insensitive lookup; only Acme Corp and Globex exist in this mock
    dataset.

    Args:
        name: The company name.

    Returns:
        A dict with "company", "ticker", and "prices" (a list of
        {"date", "close"} records), or an "error" dict if not found.
    """
    return _load(name, "prices.json")


def get_news(name: str) -> dict:
    """Return recent news headlines about the company, each pre-tagged with
    a sentiment label ("positive" or "negative").

    Use this for questions about recent events, catalysts, or sentiment.
    Case-insensitive lookup; only Acme Corp and Globex exist in this mock
    dataset.

    Args:
        name: The company name.

    Returns:
        A dict with "company" and "headlines" (a list of {"date",
        "headline", "sentiment"} records), or an "error" dict if not found.
    """
    return _load(name, "news.json")
