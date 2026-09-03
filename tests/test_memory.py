"""Unit tests for argus/tools/memory.py's pure functions — no ADK/API.

recall_prior_findings is a thin async wrapper around a live
ToolContext/MemoryService and is verified live in adk web instead (same
split as check_gather_status in orchestrator.py). remember_finding's own
MemoryService call is too, but its rejection gate (added after an
external code review found it would persist a finding even after a real
human rejection) is deterministic and gets real coverage here instead --
same reasoning and same asyncio.run() approach as test_memo.py.
"""

import asyncio

from argus.state_keys import EVIDENCE_FILINGS, META_HUMAN
from argus.tools.memory import build_distilled_memory, extract_entity_name, remember_finding


def test_extract_entity_name_reads_company_key():
    assert extract_entity_name({"company": "Globex", "sector": "Tech"}) == "Globex"


def test_extract_entity_name_returns_none_when_missing():
    assert extract_entity_name({}) is None


def test_extract_entity_name_returns_none_for_error_dict():
    assert extract_entity_name({"error": "No mock data for 'Initech'."}) is None


def test_extract_entity_name_returns_none_for_blank_company():
    assert extract_entity_name({"company": "   "}) is None


def test_build_distilled_memory_uses_first_line_of_thesis():
    thesis = "Acme Corp shows steady growth.\nMore detail on the next line."
    result = build_distilled_memory("Acme Corp", thesis, 0.95)
    assert result == "Prior ARGUS finding on Acme Corp (groundedness 0.95): Acme Corp shows steady growth."


def test_build_distilled_memory_handles_missing_groundedness():
    result = build_distilled_memory("Globex", "Globex shows a net loss.", None)
    assert "groundedness unknown" in result


def test_build_distilled_memory_handles_empty_thesis():
    result = build_distilled_memory("Globex", "", 0.8)
    assert "(no thesis produced)" in result


class _FakeToolContext:
    def __init__(self, state):
        self.state = state
        self.remembered = None

    async def add_events_to_memory(self, events):
        self.remembered = events
        return None


def test_remember_finding_refuses_after_a_real_rejection():
    ctx = _FakeToolContext({META_HUMAN: {"decision": "reject", "reason": "x"}})
    result = asyncio.run(remember_finding(ctx))
    assert result == {"remembered": False}
    assert ctx.remembered is None


def test_remember_finding_refuses_when_no_decision_was_ever_recorded():
    ctx = _FakeToolContext({})
    result = asyncio.run(remember_finding(ctx))
    assert result == {"remembered": False}
    assert ctx.remembered is None


def test_remember_finding_proceeds_after_auto_approval():
    ctx = _FakeToolContext(
        {
            META_HUMAN: {"decision": "auto_approved", "reason": "clean run"},
            EVIDENCE_FILINGS: {"company": "Acme Corp"},
        }
    )
    result = asyncio.run(remember_finding(ctx))
    assert result["remembered"] is True
    assert ctx.remembered is not None
