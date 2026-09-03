"""The claim<->evidence matcher (spec: Groundedness Verifier's tool, agent
#12). This is the actual grounding check — a pure, deterministic function,
never an LLM guessing whether a number "looks right". check_grounding now
decides WHICH numbers to check too, deterministically — see its own
docstring for why that changed from the Verifier agent doing it.

Split into a pure core (compute_grounding, ADK-free, unit-tested directly
in tests/test_claim_matcher.py) and a thin ADK-facing wrapper
(check_grounding, the actual FunctionTool) that pulls source text from
session state.
"""

import re

from google.adk.tools.tool_context import ToolContext

from argus.state_keys import (
    ANALYSIS_QUANT,
    DRAFT_THESIS,
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


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\b")


def _strip_dates(text: str) -> str:
    """Removes ISO dates (2025-02-10) before number extraction. Without
    this, a date is misread as three separate claims (2025, -2, -10),
    since _NUMBER_RE treats a leading "-" as a negative sign. Ported into
    production from the eval harness (argus/eval/custom_metrics.py),
    which found this bug live but originally only fixed it for grading —
    the real draft text check_grounding now reads directly has the exact
    same dates in it (e.g. headline dates narrated in the thesis)."""
    return _ISO_DATE_RE.sub(" ", text)


_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _strip_currency_formatting(text: str) -> str:
    """Removes "$" and thousands-separator commas before number
    extraction. "$2,850 million" would otherwise parse as two claims (2
    and 850, never 2850); a "$" sitting between a minus sign and its
    digits ("-$57 million") breaks the sign adjacency extract_numbers
    relies on, silently flipping a real loss into a positive number."""
    return _THOUSANDS_COMMA_RE.sub("", text.replace("$", ""))


_DIGIT_RANGE_DASH_RE = re.compile(r"(?<=\d)-(?=\d)")


def _split_digit_ranges(text: str) -> str:
    """Splits a dash directly between two digit groups (e.g. a year range
    like "2021-2025") into two separate numbers, instead of letting
    extract_numbers misread the dash as a negative sign on the second."""
    return _DIGIT_RANGE_DASH_RE.sub(" ", text)


def normalize_for_extraction(text: str) -> str:
    """Composes all cleanup steps before extract_numbers. Run this on any
    human-written prose (never on raw JSON, which doesn't need it) before
    trusting its numbers to be read correctly."""
    return _split_digit_ranges(_strip_currency_formatting(_strip_dates(text)))


def is_grounded(claim: float, known_numbers: set[float]) -> bool:
    """A claim is grounded if some known number matches within
    _MATCH_TOLERANCE. Some slack is needed since narration rounds slightly
    (e.g. "$812 million" for 812.4) and exact float equality would
    over-reject on this exact kind of claim — but the
    tolerance must stay small and fixed; see _MATCH_TOLERANCE."""
    return any(abs(claim - known) <= _MATCH_TOLERANCE for known in known_numbers)


def compute_grounding(claimed_numbers: list[float], source_texts: list[str]) -> dict:
    """Pure grounding-check logic: is each claimed number backed by
    something in source_texts, within rounding tolerance? No ADK
    dependency — this is what the unit tests exercise directly.

    source_texts is normalized the same way normalize_for_extraction
    handles the draft — real evidence (market price dates, headline
    dates) genuinely contains ISO dates, and an un-normalized date's day
    component (e.g. the "-01" in "2025-01-01") is itself a valid-looking
    small number. Found by an external code review with a real repro: a
    fabricated "-$1M loss" claim (normalizes to -1.0) matched the "-01"
    inside an unrelated source date and scored a false 1.0. Without this,
    the fix that made check_grounding read the draft deterministically
    was only half done — the same un-normalized-text risk was still
    sitting on the other side of the comparison.
    """
    known_numbers = extract_numbers(normalize_for_extraction(" ".join(source_texts)))
    grounded = [n for n in claimed_numbers if is_grounded(n, known_numbers)]
    ungrounded = [n for n in claimed_numbers if n not in grounded]
    ratio = len(grounded) / len(claimed_numbers) if claimed_numbers else 1.0
    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "groundedness_ratio": round(ratio, 4),
    }


def check_grounding(tool_context: ToolContext) -> dict:
    """Check whether every number the draft thesis states as fact is
    actually backed by the gathered evidence or computed metrics. Takes
    no arguments — it reads the draft directly from session state and
    extracts its numbers itself, deterministically.

    This used to take a claimed_numbers list the calling model supplied
    (by reading the draft and manually enumerating what looked like
    facts). That was a real, live-confirmed gap: the enumeration was a
    judgment call, not a mechanical one, and a number the draft literally
    stated could silently go unchecked if the model's own listing missed
    it — in either direction, since an unlucky short/empty list also
    means an empty claimed_numbers, which vacuously scores a perfect
    groundedness_ratio of 1.0 (see compute_grounding). Extracting the
    numbers here instead means an empty list can now only happen when
    the draft genuinely states no numbers at all — the only case that
    actually deserves a perfect score.

    Returns:
        dict with "grounded" (claimed numbers that matched a source value),
        "ungrounded" (claimed numbers that matched nothing), and
        "groundedness_ratio" (grounded count / total, 1.0 if the draft
        states no numbers at all).
    """
    draft = str(tool_context.state.get(DRAFT_THESIS, ""))
    claimed_numbers = sorted(extract_numbers(normalize_for_extraction(draft)))
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
