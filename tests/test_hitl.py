"""Unit tests for argus/tools/hitl.py's pure gate logic — no ADK/API.

evaluate_gate/request_human_approval/record_human_decision are thin ADK-
facing wrappers (ToolContext reads/writes, the long-running suspend) and
are verified live in adk web instead, same split as check_gather_status
and recall_prior_findings/remember_finding.
"""

from argus.tools.hitl import decide_gate

_THRESHOLD = 0.98


def test_clean_run_auto_approves():
    result = decide_gate(1.0, "PASS", _THRESHOLD)
    assert result["auto_approved"] is True
    assert result["critic_passed"] is True


def test_below_threshold_escalates():
    result = decide_gate(0.9545, "PASS", _THRESHOLD)
    assert result["auto_approved"] is False
    assert "below the" in result["reason"]


def test_missing_groundedness_escalates():
    result = decide_gate(None, "PASS", _THRESHOLD)
    assert result["auto_approved"] is False
    assert "missing" in result["reason"]


def test_non_pass_critique_escalates_even_at_perfect_groundedness():
    result = decide_gate(1.0, "1. The revenue figure is unsupported.", _THRESHOLD)
    assert result["auto_approved"] is False
    assert result["critic_passed"] is False


def test_hedged_pass_does_not_count_as_a_clean_pass():
    result = decide_gate(1.0, "PASS, but double-check the debt figure.", _THRESHOLD)
    assert result["auto_approved"] is False
    assert result["critic_passed"] is False


def test_pass_tolerates_trailing_period_case_and_whitespace():
    result = decide_gate(1.0, "  pass.\n", _THRESHOLD)
    assert result["auto_approved"] is True
    assert result["critic_passed"] is True


def test_exactly_at_threshold_auto_approves():
    result = decide_gate(0.98, "PASS", _THRESHOLD)
    assert result["auto_approved"] is True
