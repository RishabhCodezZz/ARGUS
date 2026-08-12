"""Mock internal-corpus retrieval for the Retrieval Agent — the
optional RAG path. Explicitly a stub: real RAG needs embeddings and vector
similarity search; this uses plain keyword overlap instead, which is
honest about what it actually is rather than pretending to be semantic
search running at toy scale. That's a fine stand-in for the actual design
goal here — RAG exists inside ARGUS, but only as one
subordinate tool among many — proving the architecture
(an optional AgentTool the Orchestrator may or may not call) without
building real retrieval infrastructure for a component that's deliberately
optional in the first place.
"""

import json
import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"

_COMPANY_SLUGS = {
    "acme corp": "acme_corp",
    "globex": "globex",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def search_documents(company: str, query: str, top_k: int = 2) -> dict:
    """Pure keyword-overlap search over a company's mock knowledge base.

    Ranks documents by how many distinct query words they share — not
    real semantic search, but sufficient to demonstrate retrieval as an
    optional subordinate tool.

    Returns:
        dict with "results" (list of {"title", "text"}, most relevant
        first, possibly empty if nothing matched) on success, or "error" +
        "available_companies" if the company isn't in the knowledge base.
    """
    slug = _COMPANY_SLUGS.get(company.strip().lower())
    if slug is None:
        return {
            "error": f"No knowledge base for '{company}'.",
            "available_companies": ["Acme Corp", "Globex"],
        }

    path = _DATA_DIR / f"{slug}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    query_words = _tokenize(query)
    scored = []
    for doc in data.get("documents", []):
        doc_words = _tokenize(doc["title"] + " " + doc["text"])
        overlap = len(query_words & doc_words)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return {"results": [doc for _, doc in scored[:top_k]]}


def retrieve_knowledge(company: str, query: str) -> dict:
    """Search the company's internal knowledge base — analyst notes on
    competitive landscape, strategic outlook, and leadership/governance —
    for qualitative background context. This is NOT financial data; for
    numbers, use the filings or market tools instead.

    Args:
        company: the company name.
        query: what you're trying to learn, e.g. "competitive position",
            "leadership changes", "strategic risks".

    Returns:
        dict with "results" (a list of {"title", "text"} documents, most
        relevant first — possibly empty if nothing matched) on success, or
        "error" if the company isn't in the knowledge base.
    """
    return search_documents(company, query)
