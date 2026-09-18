"""Mock data tools — load hand-authored JSON from data/ instead of hitting
real filings/market/news APIs. Deliberate: it decouples learning agent
architecture from wrangling real data-API auth and rate limits (see
DECISIONS.md). Each gather specialist (Stage 1) gets exactly one of these.
"""

import difflib
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

_COMPANY_SLUGS = {
    "acme corp": "acme_corp",
    "globex": "globex",
}


def _available_companies() -> list[str]:
    return ["Acme Corp", "Globex"]


def _suggest_closest_company(name: str) -> str | None:
    """Deterministic closest-match suggestion for a company name that
    didn't exact-match -- same difflib approach
    argus/agents/orchestrator.py's check_gather_status already uses for
    its own bounded broad-pipeline retry (decisions/matches come from
    code, not the model's own judgment, applied here to narrow
    single-specialist lookups too -- e.g. "what's Acme's price" ->
    "acme" alone doesn't exact-match "acme corp").

    cutoff=0.5, not that function's more permissive cutoff=0.0: this
    runs on EVERY unmatched lookup, including a genuinely unrelated
    name someone might type, not just as a disclosed last-resort retry
    after a real gather attempt already failed once -- a cutoff that
    low here would mean any typo silently suggests *some* company
    instead of a real "not found".

    Returns the match's real display name (e.g. "Acme Corp"), ready to
    show the user or retry with -- never the internal lowercase slug
    key -- or None when nothing is close enough to guess at.
    """
    normalized = name.strip().lower()
    matches = difflib.get_close_matches(normalized, _COMPANY_SLUGS.keys(), n=1, cutoff=0.5)
    if not matches:
        return None
    return next(c for c in _available_companies() if c.lower() == matches[0])


def _load(name: str, filename: str) -> dict:
    slug = _COMPANY_SLUGS.get(name.strip().lower())
    if slug is None:
        return {
            "error": f"No mock data for '{name}'.",
            "available_companies": _available_companies(),
            # Lets check_gather_status (argus/agents/orchestrator.py) compute
            # a retry candidate deterministically via string matching,
            # instead of leaving the model to eyeball "closest match" itself.
            "requested_name": name.strip(),
            # Deterministic best-guess for the NARROW single-specialist
            # path (a broad-pipeline request already has its own
            # separate, disclosed retry via check_gather_status).
            # orchestrator.py's instruction retries the specialist tool
            # ONCE with this name if present, then discloses the
            # substitution -- same "compute the match in code, never
            # let the model guess" rule the broad-pipeline retry
            # already follows. None when nothing is close enough.
            "suggested_name": _suggest_closest_company(name),
        }
    path = _DATA_DIR / slug / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
