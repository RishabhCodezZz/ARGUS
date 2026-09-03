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
from argus.tools.claim_matcher import normalize_for_extraction as _normalize_for_extraction
from argus.tools.reconciler import find_contradictions

# Number normalization comes from production claim_matcher.py, not a copy
# here. It used to be a private copy in this file (scoped "eval-only" at
# the time); those exact steps were later ported into production, and then
# production grew two more (prose calendar dates like "January 22, 2025",
# and "loss of $57M" -> -57) that this copy never got — so a clean broad
# run scored 0.8929 independent_groundedness here while the real Verifier
# scored the same draft 1.0. claim_matcher is now the single source of
# truth. The metric stays independent where it actually matters — the
# SOURCE of ground truth (recompute_quant_metrics + raw mock data), not
# the number-parsing, which should be byte-identical to what the system
# under test uses.

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
        # Elapsed fiscal years, not row count -- len(years)-1 only equals
        # the real elapsed span when the years are consecutive. Current
        # mock data always is, so this was silently correct today, but
        # sparse data (e.g. only 2021 and 2025 present) would give n=1
        # instead of the real 4, wildly inflating the computed CAGR.
        # Found by an external code review.
        n = last["year"] - first["year"]
        cagr = ((last["revenue_usd_millions"] / first["revenue_usd_millions"]) ** (1 / n) - 1) * 100 if n else 0.0
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


_NEGATED_DISCLOSURE_RE = re.compile(
    r"\b(no|zero|not any|without|absence of|there (?:is|are|was|were) no)\b"
    r"[\w\s,'’-]{0,40}?\b(contradictions?|tensions?|mismatch(?:es)?|"
    r"discrepanc(?:y|ies)|inconsistenc(?:y|ies)|disconnects?)\b",
    re.IGNORECASE,
)

# synthesis.py now instructs the agent to describe a reconciliation
# finding as a "tension between the narrative framing and the real
# profitability" and to RESERVE the word "contradiction" for the stronger
# "the company stated something false" case. So a correct disclosure often
# never contains the literal "contradiction" — it says "stark tension
# between Globex's narrative framing and its reported profitability", or
# "this headline tone does not match actual reported profitability". This
# recognises that phrasing: a tension/mismatch term within ~160 chars of a
# narrative-framing term, either order. A generic "competitive tension in
# core markets" has no narrative-side term nearby and does not match.
_TENSION_TERM = (
    r"(?:tension|contradict\w*|mismatch|discrepanc\w+|inconsistenc\w+|at odds|"
    r"does\s*n['’]?t\s+match|does\s+not\s+match|do\s+not\s+match|did\s+not\s+match|"
    r"fails?\s+to\s+match|diverge\w*|disconnect)"
)
_NARRATIVE_TERM = (
    r"(?:narrative|headline|framing|messaging|marketing|optimistic|rosy|upbeat|"
    r"(?:corporate|management)\s+tone|positive\s+(?:tone|framing|spin|sentiment))"
)
_TENSION_DISCLOSURE_RE = re.compile(
    rf"{_TENSION_TERM}[\s\S]{{0,160}}{_NARRATIVE_TERM}"
    rf"|{_NARRATIVE_TERM}[\s\S]{{0,160}}{_TENSION_TERM}",
    re.IGNORECASE,
)

_NO_ANALYSIS_RE = re.compile(
    r"could\s*n['’]?t\s+find|could\s+not\s+find|unable\s+to\s+(?:find|retrieve|locate|produce|analy[sz]e)|"
    r"no\s+(?:mock\s+)?data\s+(?:is\s+)?(?:was\s+)?(?:available|found)|not\s+in\s+the\s+dataset",
    re.IGNORECASE,
)
_HAS_THESIS_RE = re.compile(r"confidence\s+level|analytical\s+thesis|overall\s+take", re.IGNORECASE)


def declined_without_analysis(response_text: str) -> bool:
    """True when the run reported it could not analyze the requested entity
    and produced NO thesis of its own — a graceful decline. It has no
    contradiction-disclosure obligation, so contradiction_correctness must
    skip it (not score it against a company name it merely listed as
    available). A bounded retry that substitutes another company and DOES
    deliver an analysis ("...here's an analysis of Globex instead: <thesis>")
    still carries the thesis markers, so it is not treated as a decline —
    it gets scored normally against the substituted company.
    """
    return bool(_NO_ANALYSIS_RE.search(response_text)) and not _HAS_THESIS_RE.search(response_text)


def contradiction_flagged_correctly(response_text: str, expected_contradiction_count: int) -> bool:
    """Checks whether the response's contradiction-disclosure behavior
    matches ground truth: if a reconciliation finding was planted, the
    response must AFFIRMATIVELY disclose it (as a "contradiction" OR, per
    synthesis.py's current instruction, as a narrative-vs-profitability
    "tension"); if none was planted, it must not.

    A naive substring check on "contradiction" is wrong both ways: the
    clean case produces "There are no reported contradictions...", which
    contains the word while correctly reporting none — and the flagged
    case now usually says "tension" and never says "contradiction" at all.
    Both are covered by this module's own pytest.
    """
    mentions_contradiction_word = "contradiction" in response_text.lower()
    describes_tension = bool(_TENSION_DISCLOSURE_RE.search(response_text))
    negated = bool(_NEGATED_DISCLOSURE_RE.search(response_text))
    affirmatively_flagged = (mentions_contradiction_word or describes_tension) and not negated
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
        actual_names = [c.name for c in get_all_tool_calls(actual.intermediate_data) if c.name]
        expected_names = [c.name for c in get_all_tool_calls(expected.intermediate_data) if c.name]
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
        tool_names = [c.name for c in get_all_tool_calls(actual.intermediate_data) if c.name]
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
    but whether the final text actually reflects reality.

    Skipped (NOT_EVALUATED) for an invocation that gracefully declined to
    analyze the requested entity: a "couldn't find data for Initech — Acme
    Corp and Globex are available" response has no disclosure obligation,
    and scoring it against a company it merely LISTED (detect_company would
    latch onto "Globex" and expect its planted contradiction to have been
    flagged) is a false failure. A bounded retry that substitutes a
    company and delivers a real thesis is NOT a decline and is still
    scored, against the substituted company.
    """
    del expected_invocations, conversation_scenario
    per_invocation_results = []
    scores = []
    for actual in actual_invocations:
        tool_names = [c.name for c in get_all_tool_calls(actual.intermediate_data) if c.name]
        if not has_reconciliation_tool(tool_names):
            continue
        text = _response_text(actual)
        if declined_without_analysis(text):
            continue
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
