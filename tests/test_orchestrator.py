"""Unit tests for argus/agents/orchestrator.py's deterministic control-flow
logic — no live LLM/API calls.

check_gather_status is plain dict-and-counter logic wrapped in a
ToolContext-typed signature; a minimal stand-in object whose .state is a
real dict satisfies exactly what the function touches (tool_context.state
.get(...) / tool_context.state[...] = ...), without needing a full ADK
Runner/session/InvocationContext. Same approach as test_guardrails.py's
enforce_tool_budget tests.
"""

from argus.agents.orchestrator import (
    _REPLAN_BUDGET,
    _REPLAN_STATE_KEY,
    _stamp_request_id,
    check_gather_status,
)
from argus.state_keys import (
    EVIDENCE_FILINGS,
    EVIDENCE_FILINGS_RUN_ID,
    EVIDENCE_MARKET,
    EVIDENCE_MARKET_RUN_ID,
    EVIDENCE_SENTIMENT,
    EVIDENCE_SENTIMENT_RUN_ID,
    META_REQUEST_ID,
)

_REQ = "req-current"


class _FakeToolContext:
    def __init__(self, state=None):
        self.state = state or {}


def _clean_state():
    # A genuine, fresh gather: evidence present, error-free, and every
    # gatherer stamp matches the current request id.
    return {
        META_REQUEST_ID: _REQ,
        EVIDENCE_FILINGS: {"company": "Acme Corp"},
        EVIDENCE_MARKET: {"company": "Acme Corp"},
        EVIDENCE_SENTIMENT: {"company": "Acme Corp"},
        EVIDENCE_FILINGS_RUN_ID: _REQ,
        EVIDENCE_MARKET_RUN_ID: _REQ,
        EVIDENCE_SENTIMENT_RUN_ID: _REQ,
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


def test_state_never_populated_reads_as_failure_not_success():
    # Regression test for a real bug (found by an external code review):
    # this used to read empty/missing evidence.* state as a *successful*
    # gather, since _parse_state_json(None) returns {}, which has no
    # "error" key. That meant a model that skipped calling
    # full_analysis_pipeline entirely (or stale/cleared state) would sail
    # through as gather_ok=True. Empty state must now count as failure.
    ctx = _FakeToolContext({})
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False


def test_stale_evidence_from_a_prior_turn_is_rejected():
    # Regression test for a real bug (found by an external code review):
    # in a multi-turn session, evidence from a prior successful request
    # stays populated and error-free. If the model skips
    # full_analysis_pipeline on the next broad request but still calls
    # check_gather_status, the old evidence must NOT read as gather_ok.
    # Here _stamp_request_id has advanced meta.request_id for the new
    # turn, but the three gatherer stamps still carry the prior id.
    state = _clean_state()
    state[META_REQUEST_ID] = "req-a-new-turn"  # gatherer stamps still "req-current"
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False
    assert result["stale"] is True
    # A staleness failure is NOT a gather error, so it must not consume a
    # re-plan attempt or hand back a closest-match retry name.
    assert "suggested_retry_name" not in result
    assert _REPLAN_STATE_KEY not in ctx.state


def test_partially_stale_evidence_is_rejected():
    # One gatherer re-ran this turn, the other two are left over. All
    # three stamps must match the current request, not just one.
    state = _clean_state()
    state[EVIDENCE_MARKET_RUN_ID] = "req-stale"
    state[EVIDENCE_SENTIMENT_RUN_ID] = "req-stale"
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False
    assert result["stale"] is True


def test_missing_request_id_reads_as_stale():
    # If _stamp_request_id never ran (no meta.request_id at all), fail
    # closed rather than trusting whatever evidence happens to be around.
    state = _clean_state()
    del state[META_REQUEST_ID]
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False
    assert result["stale"] is True


def test_same_turn_gather_error_is_not_reported_as_stale():
    # A genuine "no mock data" error THIS turn (stamps match the current
    # request) must still flow to the retry path, not the stale path.
    state = _clean_state()
    state[EVIDENCE_FILINGS] = {
        "error": "No mock data for 'Initech'.",
        "available_companies": ["Acme Corp", "Globex"],
        "requested_name": "Initech",
    }
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["gather_ok"] is False
    assert result.get("stale") is not True
    assert result["suggested_retry_name"] == "Globex"


def test_stamp_request_id_writes_a_fresh_id_each_turn():
    class _CbCtx:
        def __init__(self):
            self.state = {}

    ctx = _CbCtx()
    _stamp_request_id(ctx)
    first = ctx.state[META_REQUEST_ID]
    _stamp_request_id(ctx)
    second = ctx.state[META_REQUEST_ID]
    assert first and second and first != second


def test_suggested_retry_name_matches_the_closest_available_company():
    # Regression test for a real gap (found by an external code review):
    # the retry target used to be a judgment call left entirely to the
    # model's own reading of available_companies -- nothing guaranteed it
    # picked the same match a deterministic comparison would.
    state = _clean_state()
    state[EVIDENCE_FILINGS] = {
        "error": "No mock data for 'Globexx'.",
        "available_companies": ["Acme Corp", "Globex"],
        "requested_name": "Globexx",
    }
    ctx = _FakeToolContext(state)
    result = check_gather_status(ctx)
    assert result["suggested_retry_name"] == "Globex"
