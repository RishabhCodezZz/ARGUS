"""The claim<->evidence matcher (spec: Groundedness Verifier's tool, agent
#12). This is the actual grounding check — a pure, deterministic function,
never an LLM guessing whether a number "looks right". The Verifier agent's
job is only to decide WHICH numbers to check; this module decides whether
they're real.

Split into a pure core (compute_grounding, ADK-free, unit-tested directly
in tests/test_claim_matcher.py) and a thin ADK-facing wrapper
(check_grounding, the actual FunctionTool) that pulls source text from
session state.
"""

import re

from google.adk.tools.tool_context import ToolContext

from argus.state_keys import (
    ANALYSIS_QUANT,
    EVIDENCE_FILINGS,
    EVIDENCE_MARKET,
    EVIDENCE_SENTIMENT,
    META_GROUNDEDNESS,
)

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")

# Fixed, not scaled by magnitude. A relative tolerance (e.g. 1% of the known
# value) sounds reasonable but silently breaks for identifier-like numbers
# such as years: 1% of 2025 is ~20, so a fabricated "2026" would pass as
# "close enough" to the real 2025 — found live in Stage 3 dev via an
# adversarial prompt injecting a fake future-year figure. Every rounding
# difference we've actually observed from narration (e.g. "$812 million"
# for 812.4, "7.5%" for 7.52) is sub-1, so a small fixed tolerance covers
# real narration slack without opening the year-adjacency hole.
_MATCH_TOLERANCE = 0.5


def extract_numbers(text: str) -> set[float]:
    """Pull every numeric literal out of a text blob."""
    numbers = set()
    for match in _NUMBER_RE.findall(text):
        if match and match != "-":
            numbers.add(float(match))
    return numbers


def is_grounded(claim: float, known_numbers: set[float]) -> bool:
    """A claim is grounded if some known number matches within
    _MATCH_TOLERANCE. Some slack is needed since narration rounds slightly
    (e.g. "$812 million" for 812.4) and exact float equality would
    over-reject — PLAN.md's named pitfall for this exact verifier — but the
    tolerance must stay small and fixed; see _MATCH_TOLERANCE."""
    return any(abs(claim - known) <= _MATCH_TOLERANCE for known in known_numbers)


def compute_grounding(claimed_numbers: list[float], source_texts: list[str]) -> dict:
    """Pure grounding-check logic: is each claimed number backed by
    something in source_texts, within rounding tolerance? No ADK
    dependency — this is what the unit tests exercise directly.
    """
    known_numbers = extract_numbers(" ".join(source_texts))
    grounded = [n for n in claimed_numbers if is_grounded(n, known_numbers)]
    ungrounded = [n for n in claimed_numbers if n not in grounded]
    ratio = len(grounded) / len(claimed_numbers) if claimed_numbers else 1.0
    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "groundedness_ratio": round(ratio, 4),
    }


def check_grounding(claimed_numbers: list[float], tool_context: ToolContext) -> dict:
    """Check whether each claimed number is actually backed by the gathered
    evidence or computed metrics.

    Call this with every distinct number the draft thesis states as fact
    (e.g. pass 7.52 for "7.52%", 650.0 for "$650 million", 22.90 for
    "$22.90"). You do not need to supply the source data yourself — this
    tool reads the gathered evidence and computed metrics directly from
    session state.

    Args:
        claimed_numbers: every distinct number the draft cites as a fact.

    Returns:
        dict with "grounded" (claimed numbers that matched a source value),
        "ungrounded" (claimed numbers that matched nothing), and
        "groundedness_ratio" (grounded count / total, 1.0 if none given).
    """
    sources = [
        str(tool_context.state.get(EVIDENCE_FILINGS, "")),
        str(tool_context.state.get(EVIDENCE_MARKET, "")),
        str(tool_context.state.get(EVIDENCE_SENTIMENT, "")),
        str(tool_context.state.get(ANALYSIS_QUANT, "")),
    ]
    result = compute_grounding(claimed_numbers, sources)
    # Persisted directly here, not via the agent's output_key, since this
    # agent's output_key is bound to draft.thesis (the flagged rewrite) —
    # ToolContext.state is writable, so the tool owns writing its own
    # dedicated state key rather than overloading a single text output.
    tool_context.state[META_GROUNDEDNESS] = result["groundedness_ratio"]
    return result
