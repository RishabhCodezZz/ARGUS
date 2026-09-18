"""Unit tests for argus/tools/mock_data.py -- pure Python, no API calls.

Covers the deterministic company-name resolution (_suggest_closest_company)
that backs get_filings/get_market_data/get_news's error responses for a
partial or misspelled name (e.g. "acme" for "Acme Corp") -- found live via
the Dev UI and a scratchpad smoke test both hitting the same gap the same
day this was added.
"""

from argus.tools.mock_data import (
    _suggest_closest_company,
    get_filings,
    get_market_data,
    get_news,
)


def test_exact_match_still_works_unchanged():
    result = get_filings("Acme Corp")
    assert "error" not in result
    assert result["company"] == "Acme Corp"


def test_exact_match_is_case_insensitive():
    result = get_filings("acme corp")
    assert "error" not in result
    assert result["company"] == "Acme Corp"


def test_suggest_closest_company_resolves_a_partial_name():
    # The real gap this was added for: "acme" alone doesn't exact-match
    # "acme corp", but is an obvious partial match.
    assert _suggest_closest_company("acme") == "Acme Corp"


def test_suggest_closest_company_resolves_a_typo():
    assert _suggest_closest_company("golbex") == "Globex"


def test_suggest_closest_company_returns_none_for_unrelated_input():
    # A genuinely wrong name must NOT silently resolve to some company --
    # the cutoff exists specifically to keep this case a real "not found".
    assert _suggest_closest_company("totally unrelated corporation") is None


def test_get_market_data_error_includes_suggested_name_for_partial_input():
    result = get_market_data("acme")
    assert "error" in result
    assert result["suggested_name"] == "Acme Corp"


def test_get_filings_error_has_no_suggestion_for_unrelated_input():
    result = get_filings("totally unrelated corporation")
    assert "error" in result
    assert result["suggested_name"] is None


def test_get_news_error_still_includes_available_companies_and_requested_name():
    # Existing fields must survive unchanged alongside the new one.
    result = get_news("acme")
    assert result["available_companies"] == ["Acme Corp", "Globex"]
    assert result["requested_name"] == "acme"
    assert result["suggested_name"] == "Acme Corp"
