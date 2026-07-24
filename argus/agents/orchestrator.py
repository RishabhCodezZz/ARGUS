"""The Orchestrator (spec P5, agent #1): the dynamic, LLM-driven router
that replaces a fixed pipeline as the entry point. Fan-out is now
plan-driven, not hard-coded (spec OR5) — a narrow "what's the price doing"
question invokes only the Market agent; a broad "analyze X" request invokes
the full deterministic pipeline (Stages 1-4A, unchanged, wrapped whole).

No formal planner module (e.g. PlanReActPlanner) yet — that mainly adds
visible step-by-step planning text, not different routing capability, and
a well-instructed LlmAgent choosing among AgentTools already satisfies
OR5's actual requirement. Revisit if trace legibility becomes a real need.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool

from argus.agents.gather import filings_agent, market_agent, sentiment_agent
from argus.agents.pipeline import full_analysis_pipeline
from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH

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
        "facts beyond what the tool returned.\n"
        "- A broad request (\"analyze X\", \"give me a full picture\", "
        "\"due diligence on X\", or anything needing computed metrics, a "
        "thesis, or a confidence level) → call full_analysis_pipeline. "
        "Present its result to the user VERBATIM — it is already a "
        "drafted, red-teamed, groundedness-checked analysis; paraphrasing "
        "it risks introducing an error into something already verified.\n\n"
        "If a request is ambiguous between narrow and broad, prefer the "
        "full pipeline — a more thorough answer is a safer default than a "
        "too-narrow one."
    ),
    tools=[
        AgentTool(filings_agent),
        AgentTool(market_agent),
        AgentTool(sentiment_agent),
        AgentTool(full_analysis_pipeline),
    ],
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
