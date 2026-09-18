"""Unit tests for the pure core of argus/eval/agent_count_ablation.py --
no ADK Runner, no API key. Same split this project always uses: the
scoring/reshaping logic is pure Python and gets real coverage here; the
live agent-building/running code (Tasks 4-6) is verified live, same
treatment as argus/eval/ablation.py's own build_baseline/run_variant.
"""

import json

from argus.eval.agent_count_ablation import score_groundedness, split_combined_evidence
from argus.state_keys import EVIDENCE_FILINGS, EVIDENCE_MARKET, EVIDENCE_SENTIMENT


def test_split_combined_evidence_maps_each_source_to_its_canonical_key():
    combined = {
        "filings": {"company": "Acme Corp"},
        "market": {"ticker": "ACME"},
        "sentiment": {"headlines": []},
    }
    result = split_combined_evidence(combined)
    assert json.loads(result[EVIDENCE_FILINGS]) == {"company": "Acme Corp"}
    assert json.loads(result[EVIDENCE_MARKET]) == {"ticker": "ACME"}
    assert json.loads(result[EVIDENCE_SENTIMENT]) == {"headlines": []}


def test_split_combined_evidence_defaults_missing_sources_to_empty_dict():
    result = split_combined_evidence({"filings": {"a": 1}})
    assert json.loads(result[EVIDENCE_MARKET]) == {}
    assert json.loads(result[EVIDENCE_SENTIMENT]) == {}


def test_score_groundedness_flags_a_fabricated_number():
    # Real Acme Corp mock data (data/acme_corp/filings.json) has no
    # $99,999,999M figure anywhere -- a fabricated claim like this must
    # be flagged ungrounded by the same deterministic matcher production
    # trusts (argus/tools/claim_matcher.py's compute_grounding).
    result = score_groundedness(
        "Acme Corp reported net income of $99999999 million in 2025.", "Acme Corp"
    )
    assert 99999999.0 in result["ungrounded"]


def test_score_groundedness_grounds_a_real_filed_year():
    result = score_groundedness(
        "Acme Corp's most recent fiscal year on record is 2025.", "Acme Corp"
    )
    assert 2025.0 in result["grounded"]


def test_score_groundedness_vacuous_perfect_score_when_no_numbers_stated():
    # Same documented scope limit as check_grounding itself (claim_matcher.py):
    # a prose-only response with no numeric claims scores 1.0 by design.
    result = score_groundedness("Acme Corp is a company in the mock dataset.", "Acme Corp")
    assert result["groundedness_ratio"] == 1.0
