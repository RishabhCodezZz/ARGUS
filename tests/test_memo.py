"""Unit tests for argus/tools/memo.py's pure markdown-assembly logic — no
ADK/API.

save_memo is a thin async wrapper around ToolContext.save_artifact and is
verified live in adk web instead, same split as check_gather_status and
remember_finding.
"""

from datetime import datetime, timezone

from argus.tools.memo import build_memo, memo_filename

_WHEN = datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc)


def test_memo_filename_slugifies_and_date_stamps():
    assert memo_filename("Acme Corp", _WHEN) == "memo_acme_corp_2026-08-01.md"


def test_memo_filename_handles_punctuation_and_case():
    assert memo_filename("Globex, Inc.", _WHEN) == "memo_globex_inc_2026-08-01.md"


def test_thesis_is_copied_verbatim():
    thesis = "**Analytical Thesis: Acme Corp**\n\nSome specific wording that must not change."
    memo = build_memo("Acme Corp", thesis, 1.0, {"decision": "auto_approved"}, 0, _WHEN)
    assert thesis in memo


def test_groundedness_score_is_rendered():
    memo = build_memo("Acme Corp", "thesis text", 0.9474, {"decision": "auto_approved"}, 0, _WHEN)
    assert "0.9474" in memo


def test_auto_approved_decision_is_rendered():
    memo = build_memo("Acme Corp", "t", 1.0, {"decision": "auto_approved", "reason": "clean run"}, 0, _WHEN)
    assert "auto_approved" in memo
    assert "clean run" in memo


def test_human_approved_decision_with_reason_is_rendered():
    decision = {"decision": "approve", "reason": "Human approved despite minor delta."}
    memo = build_memo("Acme Corp", "t", 0.95, decision, 0, _WHEN)
    assert "approve" in memo
    assert "Human approved despite minor delta." in memo


def test_contradiction_count_surfaces():
    memo = build_memo("Globex", "t", 1.0, {"decision": "auto_approved"}, 2, _WHEN)
    assert "2" in memo


def test_missing_thesis_degrades_gracefully():
    memo = build_memo("Acme Corp", "", 1.0, {"decision": "auto_approved"}, 0, _WHEN)
    assert "no thesis available" in memo


def test_missing_groundedness_renders_unknown():
    memo = build_memo("Acme Corp", "t", None, {"decision": "auto_approved"}, 0, _WHEN)
    assert "unknown" in memo


def test_missing_human_decision_renders_unknown():
    memo = build_memo("Acme Corp", "t", 1.0, {}, 0, _WHEN)
    assert "unknown" in memo
