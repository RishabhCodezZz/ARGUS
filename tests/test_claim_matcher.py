"""Unit tests for the claim<->evidence matcher (argus/tools/claim_matcher.py).

Pure-Python, no ADK, no API calls — this is the deterministic core the
Groundedness Verifier's whole trustworthiness claim rests on, so it gets a
real test rather than just live-demoing it and hoping.
"""

from argus.tools.claim_matcher import compute_grounding, extract_numbers, is_grounded


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
