"""Custom deterministic eval metrics for `adk eval`.

All four metrics here are pure Python re-derivations from the same mock
data and matching logic the agents themselves use — never an LLM judge.
This is this project's own core rule ("numbers/decisions come from code,
not vibes") applied to the eval harness itself: an eval whose metrics are
LLM guesses would just be trusting the system under test to grade
itself, the exact thing the Reconciliation Agent (a custom BaseAgent,
zero LLM calls) was built to avoid doing for headline spin.

Each ADK-facing metric function matches ADK's custom-metric contract
(confirmed against google/adk/evaluation/custom_metric_evaluator.py):
    def metric_fn(eval_metric: EvalMetric,
                   actual_invocations: list[Invocation],
                   expected_invocations: Optional[list[Invocation]],
                   conversation_scenario) -> EvaluationResult
`_CustomMetricEvaluator` clears `eval_metric.threshold` on the copy it
passes in before calling — pass/fail against the real threshold in
test_config.json is applied by the outer harness, not by this function;
each metric here still sets a best-effort local EvalStatus for when
results are inspected directly (e.g. by our own pytest coverage).

tool_names_trajectory exists because ADK's own built-in
tool_trajectory_avg_score compares FunctionCall.args for exact equality
too (confirmed in google/adk/evaluation/trajectory_evaluator.py:
`actual.name == expected.name and actual.args == expected.args`) — and
this project's AgentTool-wrapped calls (full_analysis_pipeline,
retrieval_agent, ...) take a free-text `request` argument whose exact
phrasing varies run to run even when the tool CHOICE was correct
(observed directly across Stage 5's live testing). The builtin metric
would fail a genuinely correct trajectory over wording, not substance.
This one only compares tool names, in order — the same IN_ORDER
semantics ADK's own ToolTrajectoryCriterion documents, just without the
args comparison that doesn't fit free-text tool arguments.
"""

import re
from typing import Optional

from google.adk.evaluation.eval_case import Invocation, get_all_tool_calls
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus, EvaluationResult, PerInvocationResult

from argus.tools.claim_matcher import compute_grounding, extract_numbers
from argus.tools.reconciler import find_contradictions

_COMPANY_FILES = {
    "acme corp": "acme_corp",
    "globex": "globex",
}


# ---------------------------------------------------------------------------
# Pure core: no ADK objects, directly unit-testable.
# ---------------------------------------------------------------------------


def names_in_order_match(actual_names: list[str], expected_names: list[str]) -> bool:
    """IN_ORDER semantics: every expected name must appear in actual, in
    order, with extra names allowed in between. Empty expected trivially
    matches, matching ADK's own ToolTrajectoryCriterion.IN_ORDER contract.
    """
    if not expected_names:
        return True
    it = iter(actual_names)
    return all(name in it for name in expected_names)


def detect_company(text: str) -> Optional[str]:
    """Finds which of the two mock companies a query/response is about, by
    the same case-insensitive name lookup argus/tools/mock_data.py uses.
    Returns None if neither name appears (e.g. a query about a company
    that isn't in the mock dataset at all).
    """
    lowered = text.lower()
    for name in _COMPANY_FILES:
        if name in lowered:
            return name
    return None


def recompute_quant_metrics(filings: dict, market: dict) -> set[float]:
    """Independently re-derives BOTH the raw data values AND the same
    computed metrics quant_agent produces, straight from the raw mock
    data — the actual "numeric fidelity" check: re-run the
    computation, don't trust the agent's own claimed numbers. Uses
    pandas' default sample-std convention (ddof=1) to match what
    quant_agent's own pandas code produces.

    Includes raw filing/price values, not just computed derivatives —
    a narrow specialist response (e.g. market_agent relaying monthly
    closing prices verbatim) cites raw data, not CAGR/margin/volatility,
    and the first version of this function only covered the latter,
    scoring narrow-scenario responses as almost entirely "unfounded"
    even though every number they cited was correct. Caught live during
    Stage 6 dev, before the full evalset run.
    """
    numbers: set[float] = set()
    years = filings.get("fiscal_years", [])
    for i, fy in enumerate(years):
        revenue = fy["revenue_usd_millions"]
        numbers.add(round(revenue, 2))
        numbers.add(round(fy["net_income_usd_millions"], 2))
        numbers.add(round(fy["total_debt_usd_millions"], 2))
        numbers.add(fy["year"])
        numbers.add(round(fy["net_income_usd_millions"] / revenue * 100, 2))
        if i > 0:
            prev_revenue = years[i - 1]["revenue_usd_millions"]
            numbers.add(round((revenue / prev_revenue - 1) * 100, 2))

    if len(years) >= 2:
        first, last = years[0], years[-1]
        n = len(years) - 1
        cagr = ((last["revenue_usd_millions"] / first["revenue_usd_millions"]) ** (1 / n) - 1) * 100
        numbers.add(round(cagr, 2))
        numbers.add(round(last["total_debt_usd_millions"] - first["total_debt_usd_millions"], 2))

    prices = [p["close"] for p in market.get("prices", [])]
    for price in prices:
        numbers.add(round(price, 2))
    if len(prices) >= 2:
        returns = [(prices[i] / prices[i - 1] - 1) * 100 for i in range(1, len(prices))]
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1) if n > 1 else 0.0
        volatility_pct = variance**0.5
        numbers.add(round(volatility_pct, 2))
        # quant_agent's own pandas code sometimes narrates this as a raw
        # fraction (e.g. "0.014929") instead of a percentage ("1.49%") --
        # both are the same real number, just different units, observed
        # live in the exact same response. Include both so a legitimate
        # narration in either unit is recognized, not just whichever one
        # this function happened to compute first.
        numbers.add(round(volatility_pct / 100, 6))

    return numbers


def _status_for(overall: Optional[float], threshold: float) -> EvalStatus:
    """NOT_EVALUATED means "no score was computed" (e.g. no scenario in
    this eval case matched a known company) -- genuinely below-threshold
    IS a failure and must be reported as FAILED, not NOT_EVALUATED. The
    first version of these metrics used NOT_EVALUATED for both cases,
    and the outer harness's case-level aggregation treats NOT_EVALUATED
    as non-blocking -- so a real failing score was silently not gating
    the eval case's overall PASS/FAIL. Caught live during Stage 6 dev.
    """
    if overall is None:
        return EvalStatus.NOT_EVALUATED
    return EvalStatus.PASSED if overall >= threshold else EvalStatus.FAILED


def numeric_fidelity_ratio(claimed_numbers: set[float], correct_numbers: set[float], tolerance: float = 0.5) -> float:
    """Fraction of claimed numbers that match an independently
    recomputed, real value within a small fixed tolerance (same
    tolerance and same reasoning as claim_matcher.py's own
    _MATCH_TOLERANCE — narration rounding is sub-1, a fabricated
    identifier like a year is not)."""
    if not claimed_numbers:
        return 1.0
    matched = sum(1 for n in claimed_numbers if any(abs(n - c) <= tolerance for c in correct_numbers))
    return matched / len(claimed_numbers)


_NEGATED_CONTRADICTION_RE = re.compile(
    r"\b(no|zero|without)\b[\w\s-]{0,30}\bcontradictions?\b", re.IGNORECASE
)


def contradiction_flagged_correctly(response_text: str, expected_contradiction_count: int) -> bool:
    """Checks whether the response's contradiction-disclosure behavior
    matches ground truth: if contradictions were planted, the response
    must AFFIRMATIVELY flag one; if none were planted, it must not.

    A naive substring check on "contradiction" is wrong here: synthesis
    agent's own instruction produces phrasing like "There are no reported
    contradictions..." for the clean case, which contains the word
    "contradiction" while correctly reporting there isn't one. Caught by
    this module's own pytest coverage before it ever shipped.
    """
    mentions_word = "contradiction" in response_text.lower()
    negated = bool(_NEGATED_CONTRADICTION_RE.search(response_text))
    affirmatively_flagged = mentions_word and not negated
    if expected_contradiction_count > 0:
        return affirmatively_flagged
    return not affirmatively_flagged


# ---------------------------------------------------------------------------
# ADK-facing adapters: unpack real Invocation/EvalMetric objects and call
# the pure functions above. Verified live via `adk eval`, not unit-tested
# directly — same split as check_gather_status and the other ADK-facing
# tool wrappers throughout this project.
# ---------------------------------------------------------------------------


def _response_text(invocation: Invocation) -> str:
    if not invocation.final_response or not invocation.final_response.parts:
        return ""
    return " ".join(p.text for p in invocation.final_response.parts if p.text)


def _query_text(invocation: Invocation) -> str:
    if not invocation.user_content or not invocation.user_content.parts:
        return ""
    return " ".join(p.text for p in invocation.user_content.parts if p.text)


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\b")


def _strip_dates(text: str) -> str:
    """Removes ISO dates (2025-02-10) before number extraction.

    claim_matcher.py's extract_numbers uses a regex that treats a
    leading "-" as a negative sign, so an un-stripped date like
    "2025-02-10" is parsed as three separate claims: 2025, -2, -10.
    Narrow specialist responses that relay dated headlines/prices
    verbatim are full of these, which would otherwise tank their
    numeric-fidelity/groundedness scores over formatting, not
    substance. claim_matcher.py itself is left untouched — this
    preprocessing is scoped to the eval harness only, since the real
    Verifier's own drafts are prose, not date-heavy raw listings, and
    this project doesn't touch a verified production path to fix an
    eval-only edge case.

    Deliberately no leading \\b: every broad-scenario response ends
    with something like "Saved to memo: memo_acme_corp_2026-08-01.md",
    and a leading \\b never matches between the underscore before
    "2026" and the digit itself (both are \\w, so there's no boundary
    there) -- the date silently survived stripping and leaked a
    phantom 2026 (plus -8, -1) into groundedness/numeric-fidelity
    claims. Caught live investigating an unexpectedly low
    broad_clean_acme score, before running the other 3 (more
    expensive) broad scenarios that all share this same memo-filename
    suffix. Dropping the leading \\b is safe here: the dd-dd-dddd
    literal shape doesn't otherwise occur in this project's mock data
    or agent phrasing, and a standalone injected year (e.g. "2026"
    alone, as in the adversarial-injection scenario) has no dashes, so
    it's never matched by this pattern regardless of boundary.
    """
    return _ISO_DATE_RE.sub(" ", text)


_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _strip_currency_formatting(text: str) -> str:
    """Removes "$" and thousands-separator commas before number
    extraction — the same class of bug as _strip_dates, found live
    while investigating why narrow_filings scored low. claim_matcher.py's
    extract_numbers has no comma handling, so "$2,850 million" is parsed
    as two separate claims (2 and 850, never 2850). Worse, a "$" sitting
    between a minus sign and its digits ("-$57 million") breaks the
    adjacency extract_numbers relies on to read the sign, so the claim
    comes back as a positive 57 instead of -57 — silently flipping the
    sign of a real net loss. Stripping "$" first (so "-$57" -> "-57")
    fixes both at once. Scoped to the eval harness only, same reasoning
    as _strip_dates.
    """
    return _THOUSANDS_COMMA_RE.sub("", text.replace("$", ""))


_DIGIT_RANGE_DASH_RE = re.compile(r"(?<=\d)-(?=\d)")


def _split_digit_ranges(text: str) -> str:
    """Splits a dash directly between two digit groups (e.g. a year range
    like "2021-2025") into two separate numbers, instead of letting
    extract_numbers misread the dash as a negative sign on the second
    number (2021, then -2025 -- found live via broad_clean_acme's own
    "over the 2021-2025 window" phrasing). Runs after _strip_dates, which
    already removes full YYYY-MM-DD dates outright (day/month aren't
    meaningful claims there); this handles what's left over, like a bare
    year range in prose. A "-$57" negative-currency claim is unaffected:
    the dash there is preceded by whitespace, not a digit.
    """
    return _DIGIT_RANGE_DASH_RE.sub(" ", text)


def _normalize_for_extraction(text: str) -> str:
    """Composes all eval-only preprocessing steps before extract_numbers."""
    return _split_digit_ranges(_strip_currency_formatting(_strip_dates(text)))


_QUANT_RELEVANT_TOOLS = {"market_agent", "filings_agent", "quant_agent", "full_analysis_pipeline"}


def has_quant_relevant_tool(tool_names: list[str]) -> bool:
    """True if the trajectory touched a tool whose numbers are checkable
    against recompute_quant_metrics (filings/market data). A pure
    sentiment_agent or retrieval_agent response cites numbers from news
    text or knowledge-base prose that were never meant to appear in a
    quant recompute — e.g. a "$50M share buyback" mentioned in a headline
    is real and grounded (compute_grounding checks it against the full
    evidence set, including news), but it isn't a quant claim, so
    numeric_fidelity has nothing meaningful to check for that invocation.
    """
    return any(name in _QUANT_RELEVANT_TOOLS for name in tool_names)


def has_reconciliation_tool(tool_names: list[str]) -> bool:
    """True only if full_analysis_pipeline ran — reconciliation between
    headline sentiment and real filings is the Reconciliation Agent's
    job, and it only runs inside the full pipeline. A narrow filings_agent
    or sentiment_agent call never reconciles across sources, so it can
    neither correctly flag nor correctly stay silent about a
    contradiction that requires evidence it never gathered.
    """
    return "full_analysis_pipeline" in tool_names


def _load_mock_data(company_key: str) -> tuple[dict, dict, dict]:
    import json
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / _COMPANY_FILES[company_key]
    filings = json.loads((data_dir / "filings.json").read_text(encoding="utf-8"))
    market = json.loads((data_dir / "prices.json").read_text(encoding="utf-8"))
    sentiment = json.loads((data_dir / "news.json").read_text(encoding="utf-8"))
    return filings, market, sentiment


def tool_names_trajectory(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario=None,
) -> EvaluationResult:
    """Custom trajectory metric: tool NAMES only, in order — see module
    docstring for why the builtin tool_trajectory_avg_score doesn't fit
    this architecture's free-text AgentTool call arguments."""
    del conversation_scenario
    if not expected_invocations:
        return EvaluationResult()

    per_invocation_results = []
    total = 0.0
    for actual, expected in zip(actual_invocations, expected_invocations):
        actual_names = [c.name for c in get_all_tool_calls(actual.intermediate_data)]
        expected_names = [c.name for c in get_all_tool_calls(expected.intermediate_data)]
        matched = names_in_order_match(actual_names, expected_names)
        score = 1.0 if matched else 0.0
        total += score
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=EvalStatus.PASSED if matched else EvalStatus.FAILED,
            )
        )
    overall = total / len(per_invocation_results) if per_invocation_results else None
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=_status_for(overall, 1.0),
        per_invocation_results=per_invocation_results,
    )


def numeric_fidelity(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario=None,
) -> EvaluationResult:
    """Re-runs the same quant computation independently from the raw mock
    data and checks the response's claimed numbers against it —
    "re-run the code, assert equal", not a trust-the-agent check."""
    del expected_invocations, conversation_scenario
    per_invocation_results = []
    scores = []
    for actual in actual_invocations:
        tool_names = [c.name for c in get_all_tool_calls(actual.intermediate_data)]
        if not has_quant_relevant_tool(tool_names):
            continue
        text = _response_text(actual)
        company = detect_company(_query_text(actual)) or detect_company(text)
        if company is None:
            continue
        filings, market, _sentiment = _load_mock_data(company)
        correct = recompute_quant_metrics(filings, market)
        claimed = extract_numbers(_normalize_for_extraction(text))
        score = numeric_fidelity_ratio(claimed, correct)
        scores.append(score)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=score,
                eval_status=EvalStatus.PASSED if score >= 0.8 else EvalStatus.FAILED,
            )
        )
    overall = sum(scores) / len(scores) if scores else None
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=_status_for(overall, 0.8),
        per_invocation_results=per_invocation_results,
    )


def independent_groundedness(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario=None,
) -> EvaluationResult:
    """Re-derives groundedness independently using claim_matcher.py's own
    compute_grounding, against the real mock data as source — deliberately
    NOT reading meta.groundedness from the run (that number comes from the
    system under test grading itself; this recomputes it from scratch).

    Includes recompute_quant_metrics' output as a 4th source, alongside
    the 3 raw evidence dicts — mirroring production's own check_grounding
    (argus/tools/claim_matcher.py), which checks against ANALYSIS_QUANT
    (quant_agent's computed output) in addition to the 3 raw evidence
    keys. A broad synthesis response legitimately narrates computed
    figures (CAGR, margins, debt change) that never appear literally in
    the raw filings JSON — only in a derived form. Checking only the 3
    raw sources (the first version of this function) meant every
    legitimate computed claim in a broad response scored as "unfounded".
    This still isn't circular: recompute_quant_metrics is an independent
    re-derivation from the same raw data, not a read of the agent's own
    quant_agent output.
    """
    del expected_invocations, conversation_scenario
    per_invocation_results = []
    scores = []
    for actual in actual_invocations:
        text = _response_text(actual)
        company = detect_company(_query_text(actual)) or detect_company(text)
        if company is None:
            continue
        filings, market, sentiment = _load_mock_data(company)
        computed = recompute_quant_metrics(filings, market)
        claimed = list(extract_numbers(_normalize_for_extraction(text)))
        result = compute_grounding(claimed, [str(filings), str(market), str(sentiment), str(sorted(computed))])
        score = result["groundedness_ratio"]
        scores.append(score)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=score,
                eval_status=EvalStatus.PASSED if score >= 0.9 else EvalStatus.FAILED,
            )
        )
    overall = sum(scores) / len(scores) if scores else None
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=_status_for(overall, 0.9),
        per_invocation_results=per_invocation_results,
    )


def contradiction_correctness(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario=None,
) -> EvaluationResult:
    """Checks the response's contradiction-disclosure behavior against
    find_contradictions' ground truth for whichever company the scenario
    concerns — not whether the agent's own reconciliation agent said so,
    but whether the final text actually reflects reality."""
    del expected_invocations, conversation_scenario
    per_invocation_results = []
    scores = []
    for actual in actual_invocations:
        tool_names = [c.name for c in get_all_tool_calls(actual.intermediate_data)]
        if not has_reconciliation_tool(tool_names):
            continue
        text = _response_text(actual)
        company = detect_company(_query_text(actual)) or detect_company(text)
        if company is None:
            continue
        filings, _market, sentiment = _load_mock_data(company)
        expected_count = len(find_contradictions(filings, sentiment))
        correct = contradiction_flagged_correctly(text, expected_count)
        score = 1.0 if correct else 0.0
        scores.append(score)
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=score,
                eval_status=EvalStatus.PASSED if correct else EvalStatus.FAILED,
            )
        )
    overall = sum(scores) / len(scores) if scores else None
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=_status_for(overall, 1.0),
        per_invocation_results=per_invocation_results,
    )
