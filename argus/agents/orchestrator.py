"""The Orchestrator (spec P5, agent #1): the dynamic, LLM-driven router
that replaces a fixed pipeline as the entry point. Fan-out is now
plan-driven, not hard-coded (spec OR5) — a narrow "what's the price doing"
question invokes only the Market agent; a broad "analyze X" request invokes
the full deterministic pipeline (Stages 1-4A, unchanged, wrapped whole).

No formal planner module (e.g. PlanReActPlanner) yet — that mainly adds
visible step-by-step planning text, not different routing capability, and
a well-instructed LlmAgent choosing among AgentTools already satisfies
OR5's actual requirement. Revisit if trace legibility becomes a real need.

Also implements OR4 — bounded re-planning: if gathering reports missing
evidence (no data for the exact name given), the Orchestrator may retry
ONCE with a corrected/refined entity name before giving up. "Refined
subtask" concretely means: pick the closest match from the failed tool's
own `available_companies` list and retry with that.

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

record_replan_attempt's counter is the hard cap — like Stage 2's
exit_loop, the instruction is the fast path but the deterministic counter
is what actually prevents runaway retries, not LLM promise-keeping.
Session-scoped, not per-entity — a known simplification (same precedent
as meta.tool_call_count); good enough for a single request's worth of
retrying, not meant to survive many unrelated failures across one long
session.

Also wraps the Retrieval Agent (Stage 4 Part D, spec's optional RAG path)
as a fourth routing case, alongside narrow/broad — for qualitative
background questions (competitive position, leadership changes) that no
structured data tool answers. Genuinely optional: the Orchestrator only
reaches for it when asked, never as part of narrow or broad routing.
"""

import json

from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext

from argus.agents.gather import filings_agent, market_agent, sentiment_agent
from argus.agents.pipeline import full_analysis_pipeline
from argus.agents.retrieval import retrieval_agent
from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH
from argus.state_keys import EVIDENCE_FILINGS, EVIDENCE_MARKET, EVIDENCE_SENTIMENT

_REPLAN_BUDGET = 1
_REPLAN_STATE_KEY = "meta.replan_count"


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
        back without an error). If gather_ok is False: "may_retry"
        (whether the re-plan budget still allows one more attempt) and
        "available_companies" (the companies that DO exist — pick the
        closest match to what the user asked for if you retry).
    """
    sources = [
        _parse_state_json(tool_context.state.get(EVIDENCE_FILINGS)),
        _parse_state_json(tool_context.state.get(EVIDENCE_MARKET)),
        _parse_state_json(tool_context.state.get(EVIDENCE_SENTIMENT)),
    ]
    errors = [s for s in sources if "error" in s]
    if not errors:
        return {"gather_ok": True}

    count = tool_context.state.get(_REPLAN_STATE_KEY, 0) + 1
    tool_context.state[_REPLAN_STATE_KEY] = count
    return {
        "gather_ok": False,
        "may_retry": count <= _REPLAN_BUDGET,
        "available_companies": errors[0].get("available_companies", []),
    }


orchestrator_agent = Agent(
    model=MODEL_FLASH,
    name="orchestrator_agent",
    description="Interprets the request and routes it to the right specialist(s) or the full pipeline.",
    instruction=(
        "You are the ARGUS Orchestrator. Decide how much of the system a "
        "request actually needs — never run more than necessary, never "
        "guess an answer yourself without calling a tool.\n\n"
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
        "thesis, or a confidence level) → call full_analysis_pipeline, "
        "THEN ALWAYS call check_gather_status before presenting anything "
        "— never decide from the thesis text alone whether data was "
        "actually found.\n"
        "  - If gather_ok is true AND this was your first attempt (the "
        "name you called it with matches what the user asked for): "
        "present the pipeline's result to the user VERBATIM — it is "
        "already drafted, red-teamed, and groundedness-checked; "
        "paraphrasing risks introducing an error into something already "
        "verified.\n"
        "  - If gather_ok is false and may_retry is true: pick whichever "
        "name in available_companies is the closest match to what the "
        "user actually asked for, and call full_analysis_pipeline again "
        "with that corrected name — this is your one refined retry, not "
        "a new unrelated request. Then call check_gather_status again.\n"
        "  - If gather_ok is true but ONLY after a retry (you had to "
        "substitute a different company name than the user asked for): "
        "you MUST say so before the analysis — one sentence stating "
        "plainly that their exact request wasn't found and this is the "
        "closest match instead (e.g. 'I couldn't find data for [what "
        "they asked], so here's an analysis of [name] instead:'). Never "
        "present a substituted company's analysis as if it directly "
        "answered the original request.\n"
        "  - If gather_ok is still false (may_retry is now false, or the "
        "retry also failed): stop. Tell the user plainly that no data "
        "was found, even after checking for a close match, and name the "
        "companies that are actually available. Do not keep retrying.\n\n"
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
    ],
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
