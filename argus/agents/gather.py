"""The parallel evidence-gathering phase: three independent
specialists, each with its own mock-data tool, fanned out via ParallelAgent.

Each sub-agent must write to a UNIQUE session.state key — see
argus/state_keys.py. Parallel branches share no state or history with each
other while running, so each one independently re-reads the user's request
to figure out which company ("entity") to look up.

Each gatherer's final response must be the tool's RAW JSON, verbatim, with
no prose wrapper. output_key persists whatever text the LLM produces as its
final turn — if that text were a natural-language paraphrase instead of the
raw payload, a transcription slip there would silently corrupt a number
before the Quant agent ever sees it, undermining "numbers come from code,
words come from the LLM" (the project's non-negotiable #1 rule). Forcing
verbatim JSON out of the gatherers keeps that boundary exact.

Each gatherer also carries an after_agent_callback that stamps its
evidence with the current meta.request_id (a per-turn id set by the
Orchestrator). check_gather_status uses those three stamps to tell fresh
evidence apart from evidence left in state by an earlier turn — see
orchestrator.py. Each stamp is its own unique key, same anti-collision
rule as the evidence keys.
"""

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent

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
from argus.tools.mock_data import get_filings, get_market_data, get_news

_VERBATIM_JSON = (
    "Respond with ONLY the raw JSON object the tool returned — no prose, "
    "no markdown, no code fences, no commentary before or after. Your "
    "entire reply must be exactly that JSON and nothing else."
)


def _stamp_freshness(stamp_key: str):
    """Builds an after_agent_callback that tags this gatherer's evidence
    with the current meta.request_id (set once per turn by the
    Orchestrator's _stamp_request_id). check_gather_status later requires
    all three tags to match the current request before it reports
    gather_ok — so evidence from an earlier turn can't be mistaken for a
    fresh gather if the model skips full_analysis_pipeline on a later
    request. Each gatherer gets its OWN stamp key (parallel branches must
    not share a state key — same rule as the evidence keys)."""

    def _cb(callback_context: CallbackContext) -> None:
        callback_context.state[stamp_key] = callback_context.state.get(META_REQUEST_ID)
        return None

    return _cb

filings_agent = Agent(
    model=MODEL_FLASH,
    name="filings_agent",
    description="Pulls structured filings data: revenue, net income, debt by year.",
    instruction=(
        "Identify the company the user is asking about and call get_filings "
        f"with that name. {_VERBATIM_JSON} Never invent numbers — only "
        "report what the tool returns."
    ),
    tools=[get_filings],
    output_key=EVIDENCE_FILINGS,
    after_agent_callback=_stamp_freshness(EVIDENCE_FILINGS_RUN_ID),
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)

market_agent = Agent(
    model=MODEL_FLASH,
    name="market_agent",
    description="Pulls the company's stock price time series.",
    instruction=(
        "Identify the company the user is asking about and call "
        f"get_market_data with that name. {_VERBATIM_JSON} Never invent "
        "prices — only report what the tool returns."
    ),
    tools=[get_market_data],
    output_key=EVIDENCE_MARKET,
    after_agent_callback=_stamp_freshness(EVIDENCE_MARKET_RUN_ID),
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)

sentiment_agent = Agent(
    model=MODEL_FLASH,
    name="sentiment_agent",
    description="Pulls recent news headlines and their sentiment.",
    instruction=(
        "Identify the company the user is asking about and call get_news "
        f"with that name. {_VERBATIM_JSON} Never invent headlines — only "
        "report what the tool returns."
    ),
    tools=[get_news],
    output_key=EVIDENCE_SENTIMENT,
    after_agent_callback=_stamp_freshness(EVIDENCE_SENTIMENT_RUN_ID),
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)

evidence_gather = ParallelAgent(
    name="evidence_gather",
    description="Runs the Filings, Market, and Sentiment agents concurrently.",
    sub_agents=[filings_agent, market_agent, sentiment_agent],
)
