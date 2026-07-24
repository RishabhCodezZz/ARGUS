"""The parallel evidence-gathering phase (spec P1): three independent
specialists, each with its own mock-data tool, fanned out via ParallelAgent.

Each sub-agent must write to a UNIQUE session.state key (spec OR2) — see
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
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent

from argus.config import MODEL_FLASH
from argus.state_keys import EVIDENCE_FILINGS, EVIDENCE_MARKET, EVIDENCE_SENTIMENT
from argus.tools.mock_data import get_filings, get_market_data, get_news

_VERBATIM_JSON = (
    "Respond with ONLY the raw JSON object the tool returned — no prose, "
    "no markdown, no code fences, no commentary before or after. Your "
    "entire reply must be exactly that JSON and nothing else."
)

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
)

evidence_gather = ParallelAgent(
    name="evidence_gather",
    description="Runs the Filings, Market, and Sentiment agents concurrently.",
    sub_agents=[filings_agent, market_agent, sentiment_agent],
)
