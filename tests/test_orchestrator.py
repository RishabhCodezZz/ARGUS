"""Unit tests for argus/agents/orchestrator.py's deterministic control-flow
logic — no live LLM/API calls.

check_gather_status is plain dict-and-counter logic wrapped in a
ToolContext-typed signature; a minimal stand-in object whose .state is a
real dict satisfies exactly what the function touches (tool_context.state
.get(...) / tool_context.state[...] = ...), without needing a full ADK
Runner/session/InvocationContext. Same approach as test_guardrails.py's
enforce_tool_budget tests.
"""

from argus.agents.orchestrator import _REPLAN_BUDGET, _REPLAN_STATE_KEY, check_gather_status
from argus.state_keys import EVIDENCE_FILINGS, EVIDENCE_MARKET, EVIDENCE_SENTIMENT


class _FakeToolContext:
    def __init__(self, state=None):
        self.state = state or {}


def _clean_state():
    return {
        EVIDENCE_FILINGS: {"company": "Acme Corp"},
        EVIDENCE_MARKET: {"company": "Acme Corp"},
        EVIDENCE_SENTIMENT: {"company": "Acme Corp"},
    }


def test_gather_ok_when_all_three_sources_are_clean():
    ctx = _FakeToolContext(_clean_state())
    assert check_gather_status(ctx) == {"gather_ok": True}


def test_gather_not_ok_when_any_source_has_an_error():
    state = _clean_state()
    state[EVIDENCE_FILINGS] = {"error": "No mock data for 'Initech'.", "available_companies": ["Acme Corp", "Globex"]}
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False
    assert result["available_companies"] == ["Acme Corp", "Globex"]


def test_first_failure_may_retry_within_budget():
    state = _clean_state()
    state[EVIDENCE_FILINGS] = {"error": "x", "available_companies": []}
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["may_retry"] is True
    assert ctx.state[_REPLAN_STATE_KEY] == 1


def test_second_consecutive_failure_exhausts_the_budget():
    # Simulates the orchestrator calling this twice in one session on a
    # genuinely repeated failure -- the counter must persist and eventually
    # cap the retry rather than allow unbounded attempts. _REPLAN_BUDGET is
    # 1, so a second failure must NOT be allowed to retry again.
    state = _clean_state()
    state[EVIDENCE_FILINGS] = {"error": "x", "available_companies": []}
    ctx = _FakeToolContext(state)
    check_gather_status(ctx)  # first failure: count becomes 1
    result = check_gather_status(ctx)  # second failure: count becomes 2
    assert result["may_retry"] is False
    assert ctx.state[_REPLAN_STATE_KEY] == 1 + _REPLAN_BUDGET


def test_uses_the_first_erroring_source_in_filings_market_sentiment_order():
    # Sources are checked in a fixed order. If only market has the error,
    # its available_companies -- not some other source's -- is what's used.
    state = _clean_state()
    state[EVIDENCE_MARKET] = {"error": "x", "available_companies": ["Globex"]}
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["available_companies"] == ["Globex"]


def test_state_never_populated_reads_as_gather_ok():
    # Documents a real precondition, not a bug: check_gather_status only
    # looks for an explicit "error" key. If evidence.* was never written at
    # all (state completely empty), _parse_state_json(None) returns {} for
    # every source -- no "error" key found, so this reads as success. The
    # function's docstring says to call it "immediately after every
    # full_analysis_pipeline call", which always populates evidence.* one
    # way or another -- this test exists to make that assumption explicit
    # and pin the actual behavior if it's ever called before that.
    ctx = _FakeToolContext({})
    assert check_gather_status(ctx) == {"gather_ok": True}
