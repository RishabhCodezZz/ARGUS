"""Unit tests for argus/tools/knowledge_base.py — pure Python, no ADK/API."""

from argus.tools.knowledge_base import search_documents


def test_returns_error_for_unknown_company():
    result = search_documents("Nonexistent Inc", "anything")
    assert "error" in result
    assert result["available_companies"] == ["Acme Corp", "Globex"]


def test_finds_relevant_document_by_keyword_overlap():
    result = search_documents("Globex", "CFO resignation leadership")
    assert len(result["results"]) >= 1
    assert result["results"][0]["title"] == "Leadership and Governance"


def test_finds_different_document_for_different_query():
    result = search_documents("Acme Corp", "competitors market share automation")
    assert len(result["results"]) >= 1
    assert result["results"][0]["title"] == "Competitive Landscape"


def test_returns_empty_results_when_nothing_matches():
    result = search_documents("Acme Corp", "xylophone quokka nonsense")
    assert result["results"] == []


def test_top_k_limits_result_count():
    result = search_documents("Globex", "the company risk business", top_k=1)
    assert len(result["results"]) <= 1


def test_company_lookup_is_case_insensitive():
    result = search_documents("GLOBEX", "restructuring")
    assert "error" not in result
    assert len(result["results"]) >= 1
