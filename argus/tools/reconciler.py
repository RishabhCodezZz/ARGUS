"""Cross-source contradiction detection (spec: Reconciliation Agent, agent
#8). Pure, deterministic Python — no LLM call needed, since the whole point
is to catch cases where narrative spin ("strongest year in company
history") disagrees with the actual filed numbers. A model judging its own
plausibility can't be trusted for that; a direct lookup can.

The check: a headline discussing a specific fiscal year (mentioned by its
4-digit year, e.g. "2024 revenue") is tagged with a sentiment label in the
mock data. If that label's polarity disagrees with the sign of that year's
actual net income, that's a real, checkable contradiction — not a matter
of interpretation.
"""

import re

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def find_contradictions(filings: dict, sentiment: dict) -> list[dict]:
    """Cross-check each headline's sentiment label against the real
    fiscal-year performance for any year it explicitly mentions.

    Args:
        filings: a parsed filings dict with a "fiscal_years" list of
            {"year", "net_income_usd_millions", ...} records.
        sentiment: a parsed sentiment dict with a "headlines" list of
            {"date", "headline", "sentiment"} records.

    Returns:
        A list of contradiction records, each naming the headline, the
        year it discusses, the actual net income, and why the sentiment
        label doesn't match. Empty if nothing contradicts.
    """
    income_by_year = {
        fy["year"]: fy["net_income_usd_millions"]
        for fy in filings.get("fiscal_years", [])
    }

    contradictions = []
    for item in sentiment.get("headlines", []):
        headline = item.get("headline", "")
        label = item.get("sentiment", "")
        for match in _YEAR_RE.findall(headline):
            year = int(match)
            if year not in income_by_year:
                continue
            net_income = income_by_year[year]
            actual_positive = net_income >= 0
            claimed_positive = label == "positive"
            if actual_positive != claimed_positive:
                contradictions.append(
                    {
                        "headline": headline,
                        "headline_date": item.get("date"),
                        "claimed_sentiment": label,
                        "year_discussed": year,
                        "actual_net_income_usd_millions": net_income,
                        "issue": (
                            f"Headline sentiment is '{label}' but FY{year} net "
                            f"income was {'a loss of ' if net_income < 0 else ''}"
                            f"${net_income}M, which is "
                            f"{'not ' if not actual_positive else ''}positive."
                        ),
                    }
                )
    return contradictions
