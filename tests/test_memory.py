"""Unit tests for argus/tools/memory.py's pure functions — no ADK/API.

recall_prior_findings and persist_finding_to_memory are thin async wrappers
around a live ToolContext/MemoryService and are verified live in adk web
instead (same split as check_gather_status in orchestrator.py).
"""

from argus.tools.memory import build_distilled_memory, extract_entity_name


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
