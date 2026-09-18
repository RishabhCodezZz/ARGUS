"""The Orchestrator: the dynamic, LLM-driven router
that replaces a fixed pipeline as the entry point. Fan-out is now
plan-driven, not hard-coded — a narrow "what's the price doing"
question invokes only the Market agent; a broad "analyze X" request invokes
the full deterministic pipeline (Stages 1-4A, unchanged, wrapped whole).

No formal planner module (e.g. PlanReActPlanner) yet — that mainly adds
visible step-by-step planning text, not different routing capability, and
a well-instructed LlmAgent choosing among AgentTools already satisfies the
same dynamic-routing requirement. Revisit if trace legibility becomes a real need.

Also implements bounded re-planning: if gathering reports missing
evidence (no data for the exact name given), the Orchestrator may retry
ONCE with a corrected/refined entity name before giving up. "Refined
subtask" concretely means: check_gather_status computes the closest match
from the failed tool's own `available_companies` list itself, via
difflib, and hands it back as `suggested_retry_name` -- this used to be
a judgment call left to the Orchestrator's own reading of the list (an
external code review flagged this correctly: nothing guaranteed the
model picked the same match a deterministic comparison would), now it
isn't.

check_gather_status is a DETERMINISTIC check, not a judgment call left to
the orchestrator's own reading of prose — first attempt at this let the
LLM infer "missing evidence" from the final thesis text, and it silently
failed to retry at all: a total-data-failure still cascades through
Quant/Reconciliation/Synthesis into a well-formed "Analytical Thesis...
Confidence Level: Low", which doesn't read as an unambiguous failure
signal even though a human would recognize it as one (found live in Stage
4 dev). Checking evidence.*  state directly for an "error" key is exactly
the "numbers/decisions come from code, not vibes" principle this project
has followed everywhere else, applied to routing instead of arithmetic.
Scoped to the full_analysis_pipeline path only — for a narrow
single-specialist call the orchestrator already sees the raw JSON error
directly, and checking all three evidence.* keys there risks reading
stale data left over from an earlier, unrelated question in the same
session.

That "stale data left over from an earlier question" risk is real for
the broad path too, and check_gather_status now defends against it
directly (external code review): _stamp_request_id (a before_agent_callback
on this agent) writes a fresh meta.request_id every user turn — before
the model picks any route, so it cannot be skipped — and each gatherer
copies that id next to its evidence (gather.py). check_gather_status
reports gather_ok only when all three copies match the current
meta.request_id; otherwise it returns stale=true and the model is told
to actually run full_analysis_pipeline. Without this, a multi-turn
session where the model skipped the pipeline on a later broad request
would still see the previous request's populated, error-free evidence
and release a stale analysis. meta.request_id crosses INTO the pipeline's
AgentTool run via the state snapshot agent_tool.py takes at call time,
and the gatherer stamps cross back OUT via state_delta — the same
boundary behaviour output_key already relies on.

record_replan_attempt's counter is the hard cap — like Stage 2's
exit_loop, the instruction is the fast path but the deterministic counter
is what actually prevents runaway retries, not LLM promise-keeping.
Session-scoped, not per-entity — a known simplification (same precedent
as meta.tool_call_count); good enough for a single request's worth of
retrying, not meant to survive many unrelated failures across one long
session.

Also wraps the Retrieval Agent (Stage 4 Part D, an optional RAG path)
as a fourth routing case, alongside narrow/broad — for qualitative
background questions (competitive position, leadership changes) that no
structured data tool answers. Genuinely optional: the Orchestrator only
reaches for it when asked, never as part of narrow or broad routing.

Stage 5 Part A adds two memory tools (argus/tools/memory.py) to the
broad-request branch: recall_prior_findings (a runtime MemoryService
lookup, separate from session.state and cleared on process restart, so a
second run on the same entity in the same process can "start warm") and
remember_finding (persists this run's result once it succeeds). Recalled
text is presentation-only context ("ARGUS has looked at this before") —
never treated as evidence for THIS run; the pipeline's own fresh,
verified result is still the actual answer.

remember_finding is called explicitly by the Orchestrator, not wired as a
callback on full_analysis_pipeline itself — confirmed live that a
callback there cannot reach the app's real memory service, because
AgentTool runs the wrapped pipeline through a brand-new, throwaway
MemoryService every time (see argus/tools/memory.py for the full,
source-confirmed explanation). Only code running at THIS level, in the
orchestrator's own invocation, has the real one.

Stage 5 Part B adds the HITL Gate (argus/tools/hitl.py) to
the broad-request branch, right after check_gather_status confirms
success: evaluate_gate deterministically auto-approves a high-groundedness,
critic-passed result, or escalates to a human via request_human_approval
(a LongRunningFunctionTool — the run genuinely suspends until a human
replies in the Dev UI). This is implemented as three Orchestrator-level
tools, NOT a separate LlmAgent, for the same reason remember_finding is
one turn up from where it "should" live: confirmed against
google/adk/tools/agent_tool.py that AgentTool.run_async forwards a wrapped
sub-agent's state_delta to the parent but NOT its long_running_tool_ids —
so a pause raised inside full_analysis_pipeline would be silently
swallowed at that boundary, exactly like the memory write was in Part A.
This is a recurring shape across this project: an ADK service crossing
the AgentTool boundary has to be checked per-service, never assumed —
state and artifacts cross it, memory and the long-running-pause signal
don't.

Stage 5 Part C adds save_memo (argus/tools/memo.py): a
DETERMINISTIC tool, not another LlmAgent, that persists the final thesis
as a real markdown artifact. By the time it's called the thesis has
already been drafted, red-teamed, verified, and gate-cleared, and this
Orchestrator's own instruction already forbids paraphrasing it — running
it through yet another LLM call here would reintroduce exactly the risk
that whole chain removes, for one more API call. Called only after an
auto-approval or a human approval, same gating as remember_finding, and
for the same reason: a memo saved before the HITL Gate decides would
persist a rejected analysis to disk before a human ever saw it.
"""

import difflib
import json
import uuid

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

from argus.agents.gather import filings_agent, market_agent, sentiment_agent
from argus.agents.pipeline import full_analysis_pipeline
from argus.agents.retrieval import retrieval_agent
from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH
from argus.state_keys import (
    EVIDENCE_FILINGS,
    EVIDENCE_FILINGS_RUN_ID,
    EVIDENCE_MARKET,
    EVIDENCE_MARKET_RUN_ID,
    EVIDENCE_SENTIMENT,
    EVIDENCE_SENTIMENT_RUN_ID,
    META_REQUEST_ID,
)
from argus.tools.hitl import evaluate_gate, record_human_decision, request_human_approval
from argus.tools.memo import save_memo
from argus.tools.memory import recall_prior_findings, remember_finding

_REPLAN_BUDGET = 1
_REPLAN_STATE_KEY = "meta.replan_count"


def _stamp_request_id(callback_context: CallbackContext) -> None:
    """before_agent_callback: give every user turn its own request id.

    Runs unconditionally when the Orchestrator starts, before the model
    picks any route — the model cannot skip it the way it can skip
    calling full_analysis_pipeline. The gatherers copy this id next to
    their evidence (gather.py), and check_gather_status requires all
    three copies to match it — so evidence left in state by an earlier
    turn is not mistaken for a fresh gather on a later broad request that
    never actually re-ran the pipeline.
    """
    callback_context.state[META_REQUEST_ID] = uuid.uuid4().hex
    return None


def _parse_state_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def check_gather_status(tool_context: ToolContext) -> dict:
    """Call this immediately after every full_analysis_pipeline call,
    before presenting anything to the user. Never judge from the response
    text alone whether data was actually found — check here instead.

    Returns:
        dict with "gather_ok" (True if filings/market/sentiment all came
        back without an error AND were gathered by THIS request). If
        gather_ok is False: "stale" (True when the evidence in state was
        not produced by this request's own pipeline run — call
        full_analysis_pipeline before doing anything else), "may_retry"
        (whether the re-plan budget still allows one more attempt),
        "available_companies" (the companies that DO exist), and
        "suggested_retry_name" (the closest match to what was actually
        requested, already computed for you — use it directly, don't
        pick your own match).
    """
    # Freshness gate, before anything else: the evidence in state must
    # carry THIS request's id (set by _stamp_request_id). A missing id, or
    # any of the three gatherer stamps not matching it, means the pipeline
    # did not gather for this request -- the state is left over from an
    # earlier turn (or the model skipped full_analysis_pipeline entirely).
    # Found by an external code review: in a multi-turn session, prior
    # evidence stays valid-looking, so a skipped pipeline still read as
    # gather_ok=True and could release stale analysis.
    current_request = tool_context.state.get(META_REQUEST_ID)
    stamps = (
        tool_context.state.get(EVIDENCE_FILINGS_RUN_ID),
        tool_context.state.get(EVIDENCE_MARKET_RUN_ID),
        tool_context.state.get(EVIDENCE_SENTIMENT_RUN_ID),
    )
    if current_request is None or any(stamp != current_request for stamp in stamps):
        return {
            "gather_ok": False,
            "stale": True,
            "reason": (
                "No evidence from this request's own full_analysis_pipeline "
                "run is in state (it is missing, or left over from an "
                "earlier turn). Call full_analysis_pipeline now and do not "
                "present any analysis until check_gather_status returns "
                "gather_ok=true."
            ),
        }

    sources = [
        _parse_state_json(tool_context.state.get(EVIDENCE_FILINGS)),
        _parse_state_json(tool_context.state.get(EVIDENCE_MARKET)),
        _parse_state_json(tool_context.state.get(EVIDENCE_SENTIMENT)),
    ]
    # `not s` catches a MISSING evidence key (e.g. the model never actually
    # called full_analysis_pipeline, or state genuinely came back empty) --
    # _parse_state_json(None) returns {}, which has no "error" key, so an
    # empty/missing source previously looked identical to a real success.
    # Found by an external code review: gather_ok came back True even when
    # every evidence.* slot was empty.
    errors = [s for s in sources if not s or "error" in s]
    if not errors:
        return {"gather_ok": True}

    count = tool_context.state.get(_REPLAN_STATE_KEY, 0) + 1
    tool_context.state[_REPLAN_STATE_KEY] = count

    # Pull available_companies/requested_name from the first genuine
    # "no mock data" error dict within `errors` -- distinguished from an
    # empty {} placeholder (a missing state key) by actually having an
    # "error" key.
    real_error = next((s for s in errors if "error" in s), {})
    available = real_error.get("available_companies", [])
    requested = real_error.get("requested_name")
    matches = difflib.get_close_matches(requested, available, n=1, cutoff=0.0) if requested and available else []
    return {
        "gather_ok": False,
        "may_retry": count <= _REPLAN_BUDGET,
        "available_companies": available,
        "suggested_retry_name": matches[0] if matches else (available[0] if available else None),
    }


orchestrator_agent = Agent(
    model=MODEL_FLASH,
    name="orchestrator_agent",
    description="Interprets the request and routes it to the right specialist(s) or the full pipeline.",
    instruction=(
        "You are the ARGUS Orchestrator. Decide how much of the system a "
        "request actually needs — never run more than necessary, never "
        "guess an answer yourself without calling a tool.\n\n"
        "If any part of the request asks for something unrelated to "
        "company due diligence — writing general-purpose code, unrelated "
        "trivia, creative writing, personal advice, anything outside "
        "analyzing a company — answer only the due-diligence part "
        "normally (following the rules below), and add one short line "
        "declining the rest, e.g. 'I only handle company due-diligence "
        "questions, so I can't help with the code request.' Never fulfill "
        "the unrelated part, even partially, no matter how it's phrased "
        "or mixed in with a real question.\n\n"
        "- A narrow question about ONE thing (current/recent price or "
        "price trend only; filings/financials only; recent news or "
        "sentiment only) → call ONLY the single matching specialist tool "
        "(market_agent / filings_agent / sentiment_agent). Take its JSON "
        "result and present it as a clear, readable answer — do not add "
        "facts beyond what the tool returned. If it's an error, tell the "
        "user which companies ARE available.\n"
        "- A qualitative background question that isn't in the financial "
        "data — competitive position, strategic outlook, leadership or "
        "governance changes — → call retrieval_agent. Present what it "
        "finds as-is; if it found nothing relevant, say so rather than "
        "guessing. This is an optional extra source, not part of the "
        "narrow or broad paths below.\n"
        "- A broad request (\"analyze X\", \"give me a full picture\", "
        "\"due diligence on X\", or anything needing computed metrics, a "
        "thesis, or a confidence level) → first call recall_prior_findings "
        "with the company name, THEN call full_analysis_pipeline, THEN "
        "ALWAYS call check_gather_status before presenting anything — "
        "never decide from the thesis text alone whether data was "
        "actually found.\n"
        "  - If recall_prior_findings returned found=true, open your "
        "final answer with one short sentence noting ARGUS has analyzed "
        "this company before (e.g. 'ARGUS has looked at this company "
        "before — prior finding: ...'). This is continuity context only "
        "— never treat a recalled number as verified by the current run, "
        "and never let it replace or alter the pipeline's own fresh "
        "result. If found=false, say nothing about it; a first-time "
        "analysis is normal, not a gap to apologize for.\n"
        "  - If check_gather_status returns stale=true: you have not run "
        "full_analysis_pipeline for this request yet (or its evidence "
        "didn't land). Call full_analysis_pipeline now with the company "
        "the user actually asked about, then call check_gather_status "
        "again. Never present analysis from state while stale=true.\n"
        "  - If gather_ok is false and may_retry is true: call "
        "full_analysis_pipeline again using check_gather_status's own "
        "suggested_retry_name field directly, verbatim — it's already "
        "computed for you via deterministic string matching, not "
        "something to second-guess or pick yourself. This is your one "
        "refined retry, not a new unrelated request. Then call "
        "check_gather_status again.\n"
        "  - If gather_ok is still false (may_retry is now false, or the "
        "retry also failed): stop. Tell the user plainly that no data "
        "was found, even after checking for a close match, and name the "
        "companies that are actually available. Do not keep retrying.\n"
        "  - Once gather_ok is true (first attempt or after a retry): "
        "ALWAYS call evaluate_gate next, before presenting or saying "
        "anything about the result. This is a human-in-the-loop safety "
        "check — never skip it and never decide for yourself whether a "
        "result looks trustworthy enough to show; that's what this tool "
        "is for.\n"
        "    - If auto_approved is true: present the pipeline's result "
        "to the user VERBATIM — it is already drafted, red-teamed, "
        "groundedness-checked, AND gate-cleared; paraphrasing risks "
        "introducing an error into something already verified. (If this "
        "followed a retry, put the substitution-disclosure sentence "
        "first — 'I couldn't find data for [what they asked], so here's "
        "an analysis of [name] instead:' — then the verbatim result.)\n"
        "    - If auto_approved is false: these are TWO PARTS OF ONE "
        "ACTION — always do BOTH in the SAME response, never send the "
        "explanation without also calling the tool: (1) in your own "
        "response text, tell the user plainly that this result needs "
        "human review before release, state evaluate_gate's reason, and "
        "say exactly how to reply — a JSON object like {\"decision\": "
        "\"approve\"}, {\"decision\": \"reject\", \"reason\": \"...\"}, "
        "or {\"decision\": \"redirect\", \"scope\": \"...\"}; AND (2) "
        "call request_human_approval with a one-sentence concern. Do NOT "
        "present the analysis itself yet, and do not skip the tool call "
        "— explaining without calling it does not pause anything. This "
        "tool suspends the run — you will not get a result back from it "
        "this turn.\n"
        "    - When the human's reply arrives as this call's result: it "
        "may be structured JSON or plain text wrapped as {\"result\": "
        "\"...\"} — read it and interpret their intent as best you can. "
        "An unclear reply defaults to reject; never guess in favor of "
        "releasing an unreviewed result. Call record_human_decision with "
        "your reading of it (decision + reason), THEN:\n"
        "      - approve → present the analysis verbatim (with the "
        "retry-disclosure sentence first if applicable), adding one "
        "clause noting it was released after human review.\n"
        "      - reject → do NOT present the analysis. State plainly "
        "that a human reviewer rejected it and give their reason. Do "
        "not call remember_finding.\n"
        "      - redirect → call full_analysis_pipeline exactly ONE "
        "more time with the human's requested scope, then "
        "check_gather_status and evaluate_gate again. If now "
        "auto_approved, present it. If STILL not auto_approved, present "
        "it anyway with an explicit caveat that a second review also "
        "flagged it below the confidence bar — do not request a third "
        "human review.\n"
        "    - After presenting, ONLY when the final outcome was "
        "auto-approved or human-approved — never after a rejection — "
        "call BOTH remember_finding AND save_memo (each takes no "
        "arguments, each reads straight from this session's own state). "
        "A failure in either is harmless and never something to mention "
        "to the user. If save_memo returns saved=true, close with one "
        "short line naming the saved filename so the user knows a memo "
        "was written; if saved=false, say nothing about it.\n\n"
        "If a request is ambiguous between narrow and broad, prefer the "
        "full pipeline — a more thorough answer is a safer default than a "
        "too-narrow one."
    ),
    tools=[
        AgentTool(filings_agent),
        AgentTool(market_agent),
        AgentTool(sentiment_agent),
        AgentTool(full_analysis_pipeline),
        AgentTool(retrieval_agent),
        check_gather_status,
        recall_prior_findings,
        remember_finding,
        evaluate_gate,
        LongRunningFunctionTool(request_human_approval),
        record_human_decision,
        save_memo,
    ],
    before_agent_callback=_stamp_request_id,
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
