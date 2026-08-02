"""Unit tests for argus/eval/custom_metrics.py's pure functions — no ADK/API.

The ADK-facing metric functions (tool_names_trajectory, numeric_fidelity,
independent_groundedness, contradiction_correctness) unpack real
Invocation/EvalMetric objects and are verified live via `adk eval`
instead, same split as every other ADK-facing wrapper in this project.
"""

from google.adk.evaluation.evaluator import EvalStatus

from argus.tools.claim_matcher import compute_grounding

from argus.eval.custom_metrics import (
    _normalize_for_extraction,
    _split_digit_ranges,
    _status_for,
    _strip_currency_formatting,
    _strip_dates,
    contradiction_flagged_correctly,
    detect_company,
    has_quant_relevant_tool,
    has_reconciliation_tool,
    names_in_order_match,
    numeric_fidelity_ratio,
    recompute_quant_metrics,
)


def test_in_order_match_allows_extra_calls_between():
    assert names_in_order_match(
        ["recall_prior_findings", "full_analysis_pipeline", "check_gather_status", "evaluate_gate"],
        ["full_analysis_pipeline", "evaluate_gate"],
    )


def test_in_order_match_fails_on_wrong_order():
    assert not names_in_order_match(
        ["evaluate_gate", "full_analysis_pipeline"],
        ["full_analysis_pipeline", "evaluate_gate"],
    )


def test_in_order_match_fails_on_missing_call():
    assert not names_in_order_match(["recall_prior_findings"], ["recall_prior_findings", "save_memo"])


def test_in_order_match_empty_expected_is_trivially_true():
    assert names_in_order_match(["anything"], [])


def test_detect_company_case_insensitive():
    assert detect_company("Give me a full due-diligence analysis of ACME CORP.") == "acme corp"
    assert detect_company("what about globex?") == "globex"


def test_detect_company_returns_none_for_unknown():
    assert detect_company("Give me a full due-diligence analysis of Initech.") is None


def test_recompute_quant_metrics_matches_acme_live_observed_values():
    # Reference values from a real quant_agent run on the actual Acme Corp
    # mock data (captured live during Stage 5 dev): CAGR 5.73%, 2025 net
    # margin 7.77%, debt change +15.0, monthly price volatility 1.49%.
    filings = {
        "fiscal_years": [
            {"year": 2021, "revenue_usd_millions": 650.0, "net_income_usd_millions": 39.0, "total_debt_usd_millions": 200.0},
            {"year": 2022, "revenue_usd_millions": 700.0, "net_income_usd_millions": 45.5, "total_debt_usd_millions": 210.0},
            {"year": 2023, "revenue_usd_millions": 745.0, "net_income_usd_millions": 51.4, "total_debt_usd_millions": 220.0},
            {"year": 2024, "revenue_usd_millions": 780.0, "net_income_usd_millions": 57.0, "total_debt_usd_millions": 225.0},
            {"year": 2025, "revenue_usd_millions": 812.4, "net_income_usd_millions": 63.1, "total_debt_usd_millions": 215.0},
        ]
    }
    market = {
        "prices": [
            {"close": 78.2}, {"close": 79.5}, {"close": 81.0}, {"close": 80.1},
            {"close": 82.4}, {"close": 83.9}, {"close": 85.2}, {"close": 84.5},
            {"close": 86.8}, {"close": 88.1}, {"close": 87.3}, {"close": 89.6},
        ]
    }
    numbers = recompute_quant_metrics(filings, market)

    def has_close_to(value, tol=0.05):
        return any(abs(n - value) <= tol for n in numbers)

    assert has_close_to(5.73)  # revenue CAGR
    assert has_close_to(7.77)  # 2025 net margin
    assert has_close_to(15.0)  # debt change
    assert has_close_to(1.49, tol=0.1)  # monthly price volatility


def test_recompute_quant_metrics_includes_volatility_as_both_percent_and_fraction():
    # Regression test for the real bug this eval harness found live:
    # quant_agent's own narration sometimes states volatility as a raw
    # fraction ("0.014929") instead of a percentage ("1.49%") in the
    # exact same response -- the first version of this function only
    # included the percentage-scale value, so the fraction form (a
    # legitimate, correct restatement) scored as unfounded.
    filings = {"fiscal_years": []}
    market = {"prices": [{"close": p} for p in [78.2, 79.5, 81.0, 80.1, 82.4, 83.9, 85.2, 84.5, 86.8, 88.1, 87.3, 89.6]]}
    numbers = recompute_quant_metrics(filings, market)

    def has_close_to(value, tol=0.001):
        return any(abs(n - value) <= tol for n in numbers)

    assert has_close_to(1.49, tol=0.05)  # percentage form
    assert has_close_to(0.0149, tol=0.001)  # fraction form


def test_numeric_fidelity_ratio_all_matched():
    assert numeric_fidelity_ratio({650.0, 5.73}, {650.0, 5.73, 812.4}) == 1.0


def test_numeric_fidelity_ratio_partial_match():
    assert numeric_fidelity_ratio({650.0, 9999.0}, {650.0}) == 0.5


def test_numeric_fidelity_ratio_empty_claims_is_vacuously_perfect():
    assert numeric_fidelity_ratio(set(), {650.0}) == 1.0


def test_numeric_fidelity_ratio_rejects_fabricated_adjacent_year():
    # Same identifier-adjacency case claim_matcher.py's tests cover — a
    # fabricated 2026 must NOT pass as "close enough" to a real 2025.
    assert numeric_fidelity_ratio({2026.0}, {2025.0}) == 0.0


def test_contradiction_flagged_correctly_when_present_and_mentioned():
    assert contradiction_flagged_correctly("...a notable contradiction occurred...", 1)


def test_contradiction_flagged_correctly_fails_when_present_but_unmentioned():
    assert not contradiction_flagged_correctly("Everything looks great.", 1)


def test_contradiction_flagged_correctly_recognizes_explicit_negation():
    # "No contradictions" contains the word "contradiction" -- a naive
    # substring check would wrongly treat this as an affirmative flag.
    # This is real phrasing synthesis_agent actually produces for a clean
    # case ("There are no reported contradictions...").
    assert contradiction_flagged_correctly("There are no reported contradictions.", 0) is True


def test_contradiction_flagged_correctly_clean_response_with_no_mention():
    assert contradiction_flagged_correctly("Acme Corp shows steady growth.", 0) is True


def test_contradiction_flagged_correctly_fails_on_false_positive():
    # A response that falsely claims a contradiction when none was planted.
    assert contradiction_flagged_correctly("A contradiction was found in the headlines.", 0) is False


def test_contradiction_flagged_correctly_recognizes_hyphenated_negation():
    # Regression test for the real bug this eval harness found live: the
    # negation regex's [\w\s]{0,15} character class doesn't include "-",
    # so real synthesis_agent phrasing like "No headline-to-filing
    # contradictions were identified" failed to match (the hyphens in
    # "headline-to-filing" broke the sequence), scoring a correct clean
    # response as a false positive.
    assert contradiction_flagged_correctly(
        "No headline-to-filing contradictions were identified in the reported data.", 0
    ) is True


def test_strip_dates_removes_iso_dates():
    assert _strip_dates("On 2025-02-10 the price closed at 78.2.") == "On   the price closed at 78.2."


def test_strip_dates_prevents_negative_number_misparse():
    # Regression test for the real bug this eval harness found live: an
    # un-stripped date is parsed as three separate negative-number
    # claims by claim_matcher.py's extract_numbers (leading "-" reads as
    # a negative sign), which would otherwise tank a narrow specialist
    # response's numeric-fidelity score over formatting, not substance.
    from argus.tools.claim_matcher import extract_numbers

    raw = extract_numbers("Headline dated 2025-02-10 mentions steel delays.")
    stripped = extract_numbers(_strip_dates("Headline dated 2025-02-10 mentions steel delays."))
    assert -2.0 in raw or -10.0 in raw  # confirms the bug exists in the raw path
    assert -2.0 not in stripped and -10.0 not in stripped


def test_strip_dates_handles_date_embedded_in_filename():
    # Regression test for the real bug this eval harness found live: every
    # broad-scenario response ends with "Saved to memo:
    # memo_acme_corp_2026-08-01.md" -- a leading \b never matches between
    # the underscore before "2026" and the digit (both are \w, no
    # boundary), so the date silently survived the old regex and leaked a
    # phantom 2026 into every broad scenario's groundedness/numeric-
    # fidelity claims.
    from argus.tools.claim_matcher import extract_numbers

    text = "Saved to memo: memo_acme_corp_2026-08-01.md"
    stripped = extract_numbers(_strip_dates(text))
    assert 2026.0 not in stripped
    assert -8.0 not in stripped
    assert -1.0 not in stripped


def test_recompute_quant_metrics_includes_raw_prices():
    # Regression test: the first version only included computed
    # derivatives (CAGR, margins, volatility), so a narrow market_agent
    # response relaying raw monthly prices verbatim scored as almost
    # entirely "unfounded" even though every price it cited was correct.
    filings = {"fiscal_years": []}
    market = {"prices": [{"close": 78.2}, {"close": 79.5}, {"close": 81.0}]}
    numbers = recompute_quant_metrics(filings, market)
    assert 78.2 in numbers
    assert 79.5 in numbers
    assert 81.0 in numbers


def test_status_for_below_threshold_is_failed_not_not_evaluated():
    # Regression test for the real bug this eval harness found live: the
    # first version used NOT_EVALUATED for a genuinely below-threshold
    # score, which the outer harness's case-level aggregation treats as
    # non-blocking -- silently letting a real failure pass the gate.
    assert _status_for(0.5, threshold=0.9) == EvalStatus.FAILED


def test_status_for_above_threshold_is_passed():
    assert _status_for(0.95, threshold=0.9) == EvalStatus.PASSED


def test_status_for_no_score_is_not_evaluated():
    assert _status_for(None, threshold=0.9) == EvalStatus.NOT_EVALUATED


def test_strip_currency_formatting_removes_dollar_sign():
    assert _strip_currency_formatting("$57 million") == "57 million"


def test_strip_currency_formatting_removes_thousands_comma():
    assert _strip_currency_formatting("2,850") == "2850"


def test_strip_currency_formatting_fixes_sign_dropped_by_intervening_dollar():
    # Regression test for the real bug this eval harness found live: a
    # "$" sitting between a minus sign and its digits ("-$57 million")
    # breaks extract_numbers' adjacency check for the sign, so a real
    # net loss came back as a positive 57 instead of -57.
    from argus.tools.claim_matcher import extract_numbers

    raw = extract_numbers("Net Income: -$57 million (a net loss)")
    fixed = extract_numbers(_strip_currency_formatting("Net Income: -$57 million (a net loss)"))
    assert 57.0 in raw and -57.0 not in raw  # confirms the sign-drop bug exists in the raw path
    assert -57.0 in fixed


def test_normalize_for_extraction_recovers_comma_formatted_revenue():
    from argus.tools.claim_matcher import extract_numbers

    text = "Globex reported: Revenue: $2,850 million. Net Income: -$57 million (a net loss)."
    numbers = extract_numbers(_normalize_for_extraction(text))
    assert 2850.0 in numbers
    assert -57.0 in numbers


def test_has_quant_relevant_tool_true_for_market_or_filings():
    assert has_quant_relevant_tool(["market_agent"])
    assert has_quant_relevant_tool(["filings_agent"])
    assert has_quant_relevant_tool(["full_analysis_pipeline"])


def test_has_quant_relevant_tool_false_for_sentiment_only():
    # Regression test for the real scoping bug this eval harness found
    # live: numeric_fidelity was being graded against a pure
    # sentiment_agent response, whose numbers (e.g. a buyback amount
    # from a headline) were never meant to appear in a quant recompute
    # of filings/market data -- scoring an accurate news citation as
    # "unfounded".
    assert not has_quant_relevant_tool(["sentiment_agent"])
    assert not has_quant_relevant_tool(["retrieval_agent"])


def test_split_digit_ranges_fixes_year_range_sign_misparse():
    # Regression test for the real bug this eval harness found live: a
    # bare year range in prose ("the 2021-2025 window") is parsed by
    # extract_numbers as 2021 followed by -2025 (the dash reads as a
    # minus sign), flipping a real year into a fabricated-looking
    # negative number and dragging down groundedness for no reason.
    from argus.tools.claim_matcher import extract_numbers

    raw = extract_numbers("over the 2021-2025 window")
    fixed = extract_numbers(_split_digit_ranges("over the 2021-2025 window"))
    assert -2025.0 in raw  # confirms the bug exists in the raw path
    assert 2021.0 in fixed and 2025.0 in fixed and -2025.0 not in fixed


def test_split_digit_ranges_does_not_affect_negative_currency():
    from argus.tools.claim_matcher import extract_numbers

    fixed = extract_numbers(_split_digit_ranges("Net Income: -57 million"))
    assert -57.0 in fixed


def test_independent_groundedness_recognizes_computed_quant_claims():
    # Regression test for the real bug this eval harness found live: a
    # broad synthesis response legitimately narrates computed figures
    # (CAGR, margins, debt change) that never appear literally in the raw
    # filings JSON, only in derived form -- checking only the 3 raw
    # evidence sources (the first version of independent_groundedness)
    # scored every one of these as "unfounded" even in a clean run.
    filings = {
        "fiscal_years": [
            {"year": 2021, "revenue_usd_millions": 650.0, "net_income_usd_millions": 39.0, "total_debt_usd_millions": 200.0},
            {"year": 2025, "revenue_usd_millions": 812.4, "net_income_usd_millions": 63.1, "total_debt_usd_millions": 215.0},
        ]
    }
    market = {"prices": []}
    computed = recompute_quant_metrics(filings, market)
    # CAGR over a single 4-year gap between these two points:
    cagr = round(((812.4 / 650.0) ** (1 / 1) - 1) * 100, 2)
    result_without_quant_source = compute_grounding([cagr], [str(filings), str(market), "{}"])
    result_with_quant_source = compute_grounding([cagr], [str(filings), str(market), "{}", str(sorted(computed))])
    assert result_without_quant_source["groundedness_ratio"] == 0.0
    assert result_with_quant_source["groundedness_ratio"] == 1.0


def test_has_reconciliation_tool_true_only_for_full_pipeline():
    # Regression test for the real scoping bug this eval harness found
    # live: contradiction_correctness was being graded against a narrow
    # filings_agent-only response for Globex, which structurally cannot
    # flag a contradiction -- reconciling headline sentiment against real
    # filings is the Reconciliation Agent's job, and it only runs inside
    # full_analysis_pipeline.
    assert has_reconciliation_tool(["full_analysis_pipeline"])
    assert not has_reconciliation_tool(["filings_agent"])
    assert not has_reconciliation_tool(["sentiment_agent", "filings_agent"])
