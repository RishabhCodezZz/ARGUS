"""Unit tests for argus/tools/hitl.py's pure gate logic — no ADK/API.

request_human_approval (the long-running suspend) is verified live in adk
web. decide_gate is pure. evaluate_gate's state write is deterministic
Python over a trivial ToolContext stand-in — after an external code
review found it could leave a prior run's approval in meta.human when a
later run failed the gate, that write gets real regression coverage here
(same asyncio.run() approach as test_memo.py for the two persistence
tools it must keep fail-closed).
"""

import asyncio

from argus.state_keys import (
    EVIDENCE_FILINGS,
    META_GROUNDEDNESS,
    META_HUMAN,
    REVIEW_CRITIQUE,
)
from argus.tools.hitl import decide_gate, evaluate_gate
from argus.tools.memo import save_memo
from argus.tools.memory import remember_finding

_THRESHOLD = 0.98


class _FakeToolContext:
    def __init__(self, state):
        self.state = state
        self.saved = None
        self.remembered = None

    async def save_artifact(self, filename, part):
        self.saved = (filename, part)
        return 0

    async def add_events_to_memory(self, events):
        self.remembered = events
        return None


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


def test_auto_approval_records_meta_human():
    ctx = _FakeToolContext({META_GROUNDEDNESS: 1.0, REVIEW_CRITIQUE: "PASS"})
    result = evaluate_gate(ctx)
    assert result["auto_approved"] is True
    assert ctx.state[META_HUMAN]["decision"] == "auto_approved"


def test_failed_gate_overwrites_a_stale_prior_approval_and_keeps_persistence_closed():
    # Regression test for a real bug (found by an external code review):
    # evaluate_gate used to write meta.human ONLY on auto-approval. If an
    # earlier run in the same session auto-approved, then a later run
    # failed the gate, the stale "auto_approved" survived -- and a
    # mistaken later remember_finding / save_memo would pass their
    # code-level approval checks and persist the unreviewed result.
    ctx = _FakeToolContext(
        {
            META_HUMAN: {"decision": "auto_approved", "reason": "an earlier clean run"},
            EVIDENCE_FILINGS: {"company": "Acme Corp"},
            META_GROUNDEDNESS: 0.80,  # this run is well below threshold
            REVIEW_CRITIQUE: "PASS",
        }
    )

    result = evaluate_gate(ctx)
    assert result["auto_approved"] is False
    # The stale approval must be gone, replaced with a disallowed state.
    assert ctx.state[META_HUMAN]["decision"] == "pending_review"

    # Both persistence tools must now refuse -- by code, not by instruction.
    assert asyncio.run(remember_finding(ctx)) == {"remembered": False}
    assert asyncio.run(save_memo(ctx)) == {"saved": False}
    assert ctx.remembered is None
    assert ctx.saved is None
