"""Unit tests for argus/tools/reconciler.py — pure Python, no ADK/API."""

import json
from pathlib import Path

from argus.tools.reconciler import find_contradictions

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_no_contradiction_when_sentiment_matches_actual_performance():
    filings = {"fiscal_years": [{"year": 2024, "net_income_usd_millions": 50}]}
    sentiment = {"headlines": [{"date": "2025-01-01", "headline": "Great 2024 results", "sentiment": "positive"}]}
    assert find_contradictions(filings, sentiment) == []


def test_flags_positive_headline_against_a_loss_year():
    filings = {"fiscal_years": [{"year": 2024, "net_income_usd_millions": -57}]}
    sentiment = {
        "headlines": [
            {
                "date": "2025-01-22",
                "headline": "Reports record 2024 revenue, strongest year in company history",
                "sentiment": "positive",
            }
        ]
    }
    result = find_contradictions(filings, sentiment)
    assert len(result) == 1
    assert result[0]["year_discussed"] == 2024
    assert result[0]["actual_net_income_usd_millions"] == -57


def test_ignores_headline_with_no_year_mentioned():
    filings = {"fiscal_years": [{"year": 2024, "net_income_usd_millions": -57}]}
    sentiment = {"headlines": [{"date": "2025-03-11", "headline": "Announces layoffs", "sentiment": "negative"}]}
    assert find_contradictions(filings, sentiment) == []


def test_ignores_year_not_present_in_filings():
    filings = {"fiscal_years": [{"year": 2024, "net_income_usd_millions": -57}]}
    sentiment = {"headlines": [{"date": "2025-01-01", "headline": "2019 was a great year", "sentiment": "positive"}]}
    assert find_contradictions(filings, sentiment) == []


def test_negative_headline_against_a_loss_year_is_consistent():
    filings = {"fiscal_years": [{"year": 2024, "net_income_usd_millions": -57}]}
    sentiment = {"headlines": [{"date": "2025-01-01", "headline": "2024 was a rough year", "sentiment": "negative"}]}
    assert find_contradictions(filings, sentiment) == []


def test_catches_the_actual_planted_globex_contradiction():
    # this is the real mock data seeded in Stage 1 specifically for this check
    filings = json.loads((_DATA_DIR / "globex" / "filings.json").read_text())
    sentiment = json.loads((_DATA_DIR / "globex" / "news.json").read_text())
    result = find_contradictions(filings, sentiment)
    assert len(result) == 1
    assert result[0]["year_discussed"] == 2024
    assert result[0]["claimed_sentiment"] == "positive"
    assert result[0]["actual_net_income_usd_millions"] == -57


def test_acme_corp_has_no_contradictions():
    # the "clean" company - no headline should ever get flagged
    filings = json.loads((_DATA_DIR / "acme_corp" / "filings.json").read_text())
    sentiment = json.loads((_DATA_DIR / "acme_corp" / "news.json").read_text())
    assert find_contradictions(filings, sentiment) == []
