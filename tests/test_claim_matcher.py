"""Unit tests for the claim<->evidence matcher (argus/tools/claim_matcher.py).

Pure-Python, no ADK, no API calls — this is the deterministic core the
Groundedness Verifier's whole trustworthiness claim rests on, so it gets a
real test rather than just live-demoing it and hoping.
"""

from argus.tools.claim_matcher import (
    compute_grounding,
    extract_numbers,
    is_grounded,
    normalize_for_extraction,
)


def test_extract_numbers_from_json_like_text():
    text = '{"revenue_usd_millions": 812.4, "year": 2025, "net_income_usd_millions": -118.2}'
    assert extract_numbers(text) == {812.4, 2025.0, -118.2}


def test_extract_numbers_ignores_bare_minus_sign():
    assert extract_numbers("a - b") == set()


def test_is_grounded_exact_match():
    assert is_grounded(650.0, {650.0, 2021.0}) is True


def test_is_grounded_within_rounding_tolerance():
    # narration often rounds slightly - 7.5 quoted for a source value of 7.52
    assert is_grounded(7.5, {7.52, 100.0}) is True


def test_is_grounded_rejects_fabricated_value():
    assert is_grounded(2000.0, {650.0, 700.0, 7.52}) is False


def test_is_grounded_rejects_adjacent_year():
    # regression: a magnitude-scaled relative tolerance let a fabricated
    # "2026" pass as "close enough" to a real "2025" (found live via an
    # adversarial prompt in Stage 3 dev) - years must match exactly, not
    # within a percentage of their own size.
    assert is_grounded(2026.0, {2021.0, 2022.0, 2023.0, 2024.0, 2025.0}) is False


def test_compute_grounding_all_claims_grounded():
    source = ['{"revenue_usd_millions": 812.4, "year": 2025}', '{"cagr": 5.73}']
    result = compute_grounding([812.4, 2025.0, 5.73], source)
    assert result["ungrounded"] == []
    assert set(result["grounded"]) == {812.4, 2025.0, 5.73}
    assert result["groundedness_ratio"] == 1.0


def test_compute_grounding_flags_fabricated_number():
    # this is the actual Stage 3 DoD: feed an unsupported claim, watch it get flagged
    source = ['{"revenue_usd_millions": 812.4, "net_income_usd_millions": 63.1}']
    result = compute_grounding([812.4, 2000.0], source)
    assert result["grounded"] == [812.4]
    assert result["ungrounded"] == [2000.0]
    assert result["groundedness_ratio"] == 0.5


def test_compute_grounding_no_claims_is_vacuously_fully_grounded():
    result = compute_grounding([], ["irrelevant source text"])
    assert result == {"grounded": [], "ungrounded": [], "groundedness_ratio": 1.0}


# Regression tests for check_grounding's move to deterministic extraction
# (an external code review found this was previously an LLM enumeration
# step, letting a real number silently go unchecked in either direction).
# These are the same real bugs the eval harness (argus/eval/custom_metrics.py)
# found live, ported into production now that check_grounding reads the
# draft's own prose text directly instead of trusting a supplied list.


def test_normalize_strips_thousands_commas_so_number_stays_whole():
    assert extract_numbers(normalize_for_extraction("$2,850 million")) == {2850.0}


def test_normalize_preserves_sign_when_dollar_sign_sits_before_digits():
    assert extract_numbers(normalize_for_extraction("-$57 million")) == {-57.0}


def test_normalize_strips_iso_dates_entirely():
    assert extract_numbers(normalize_for_extraction("Saved to memo_2026-08-01.md")) == set()


def test_normalize_splits_a_bare_year_range():
    assert extract_numbers(normalize_for_extraction("the 2021-2025 window")) == {2021.0, 2025.0}


# Prose calendar dates: a narrated "January 22, 2025" contributes a bare
# "22" that is not a financial claim but gets extracted as one and then
# flagged ungrounded (nothing in the evidence is a bare 22). Found live:
# two clean Globex runs dropped below the 0.98 auto-approval bar with a
# narrated calendar day as the only (or dominant) unmatched value.


def test_normalize_strips_month_day_year_date():
    assert extract_numbers(normalize_for_extraction("a headline on January 22, 2025 said")) == set()


def test_normalize_strips_month_day_without_year():
    assert extract_numbers(normalize_for_extraction("closed at 22.9 on December 31, in line")) == {22.9}


def test_normalize_strips_abbreviated_month_date():
    assert extract_numbers(normalize_for_extraction("filed Dec 31 and again Mar 5")) == set()


def test_normalize_strips_month_year():
    assert extract_numbers(normalize_for_extraction("layoffs began in March 2025 broadly")) == set()


def test_normalize_strips_quarter_references():
    assert extract_numbers(normalize_for_extraction("a wider Q2 2025 loss, worse than Q4")) == set()


def test_normalize_keeps_real_numbers_next_to_a_prose_date():
    # the date goes, the actual figures stay
    got = extract_numbers(normalize_for_extraction(
        "On January 22, 2025 revenue was 812.4 with a 7.52% CAGR"
    ))
    assert got == {812.4, 7.52}


def test_normalize_does_not_eat_may_as_a_verb():
    # "may" is a month name but also a common modal verb - only strip it
    # when it is followed by a day/year like a real date
    assert extract_numbers(normalize_for_extraction("revenue may reach 900 next year")) == {900.0}


# "loss of $N": financial English states a loss as a positive magnitude
# ("a net loss of $57 million"), but the filings store net income signed
# (-57). Found live as the only reason two clean Globex contradiction runs
# scored 0.9583 and routed to human review.


def test_normalize_signs_a_dollar_loss_amount():
    assert extract_numbers(normalize_for_extraction("reported a net loss of $57 million")) == {-57.0}


def test_normalize_signs_a_loss_with_thousands_separator():
    assert extract_numbers(normalize_for_extraction("posted losses of $1,550 million")) == {-1550.0}


def test_normalize_leaves_a_percent_loss_alone():
    # no "$" -> this is not a currency figure; evidence has a real +8 here
    assert extract_numbers(normalize_for_extraction("layoffs, a loss of 8% of the workforce")) == {8.0}


def test_normalize_does_not_touch_a_profit_amount():
    assert extract_numbers(normalize_for_extraction("a profit of $88 million, up sharply")) == {88.0}


def test_loss_restatement_now_grounds_against_signed_filing():
    # end to end: "$57 million loss" in the draft vs -57 in the evidence
    draft_numbers = sorted(extract_numbers(normalize_for_extraction(
        "Globex recorded a net loss of $57 million in fiscal year 2024."
    )))
    source = ['{"net_income_usd_millions": -57, "year": 2024}']
    result = compute_grounding(draft_numbers, source)
    assert result["ungrounded"] == []
    assert result["groundedness_ratio"] == 1.0


def test_compute_grounding_normalizes_source_text_too_not_just_claims():
    # Regression test for a real, reproduced bug (found by an external
    # code review): compute_grounding normalized the draft's own claims
    # but extracted "known" numbers from raw, un-normalized source text.
    # A date's day component looks exactly like a small real number --
    # "2025-01-01" contains "-01" -- so a fabricated "-$1M loss" (which
    # normalizes to -1.0) matched the "-01" inside an unrelated source
    # date and scored a false, perfect groundedness_ratio of 1.0.
    source = ['{"date": "2025-01-01"}']
    result = compute_grounding([-1.0], source)
    assert result["ungrounded"] == [-1.0]
    assert result["groundedness_ratio"] == 0.0
