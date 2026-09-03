"""Unit tests for argus/tools/memo.py's pure markdown-assembly logic — no
ADK/API.

The artifact-saving mechanics (ToolContext.save_artifact itself) are
still verified live in adk web, same split as check_gather_status. But
save_memo's rejection gate (added after an external code review found it
would save a memo even after a real human rejection) is deterministic
Python with no ADK dependency beyond a trivial ToolContext stand-in, so
it gets real regression coverage here instead of living as a live-only
claim -- these tests use asyncio.run() directly rather than adding a
pytest-asyncio dependency for four tests.
"""

import asyncio
from datetime import datetime, timezone

from argus.state_keys import EVIDENCE_FILINGS, META_HUMAN
from argus.tools.memo import build_memo, memo_filename, save_memo

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
    assert "Narrative-profitability tensions found | 2" in memo


def test_missing_thesis_degrades_gracefully():
    memo = build_memo("Acme Corp", "", 1.0, {"decision": "auto_approved"}, 0, _WHEN)
    assert "no thesis available" in memo


def test_missing_groundedness_renders_unknown():
    memo = build_memo("Acme Corp", "t", None, {"decision": "auto_approved"}, 0, _WHEN)
    assert "unknown" in memo


def test_missing_human_decision_renders_unknown():
    memo = build_memo("Acme Corp", "t", 1.0, {}, 0, _WHEN)
    assert "unknown" in memo


class _FakeToolContext:
    def __init__(self, state):
        self.state = state
        self.saved = None

    async def save_artifact(self, filename, part):
        self.saved = (filename, part)
        return 0


def test_save_memo_refuses_after_a_real_rejection():
    # Regression test for the exact gap an external code review found and
    # a live call confirmed: this used to save a memo unconditionally,
    # with the ONLY thing stopping it after a rejection being the calling
    # agent's own instructions, never enforced in code.
    ctx = _FakeToolContext({META_HUMAN: {"decision": "reject", "reason": "x"}})
    result = asyncio.run(save_memo(ctx))
    assert result == {"saved": False}
    assert ctx.saved is None


def test_save_memo_refuses_when_no_decision_was_ever_recorded():
    # Fail closed, not just fail on an explicit reject.
    ctx = _FakeToolContext({})
    result = asyncio.run(save_memo(ctx))
    assert result == {"saved": False}
    assert ctx.saved is None


def test_save_memo_proceeds_after_auto_approval():
    ctx = _FakeToolContext(
        {
            META_HUMAN: {"decision": "auto_approved", "reason": "clean run"},
            EVIDENCE_FILINGS: {"company": "Acme Corp"},
        }
    )
    result = asyncio.run(save_memo(ctx))
    assert result["saved"] is True
    assert ctx.saved is not None


def test_save_memo_proceeds_after_human_approval():
    ctx = _FakeToolContext(
        {
            META_HUMAN: {"decision": "approve", "reason": "manual review"},
            EVIDENCE_FILINGS: {"company": "Acme Corp"},
        }
    )
    result = asyncio.run(save_memo(ctx))
    assert result["saved"] is True
