"""The Refiner (spec P3, agent #11): rewrites the draft to address the
Critic's critique. Runs only when the Critic did NOT pass (if it had, the
Critic's exit_loop call already stopped the loop before Refiner runs again).

Still bound by the project's core rule: revising the thesis must not
invent numbers. The Refiner gets the same computed-metrics/evidence access
as the Critic so it can fix an unsupported claim by either grounding it
properly or removing it — never by making up a replacement figure.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from argus.config import MODEL_FLASH
from argus.state_keys import (
    ANALYSIS_QUANT,
    DRAFT_THESIS,
    EVIDENCE_FILINGS,
    EVIDENCE_MARKET,
    EVIDENCE_SENTIMENT,
    REVIEW_CRITIQUE,
)


def _refiner_instruction(context: ReadonlyContext) -> str:
    draft = context.state.get(DRAFT_THESIS, "(no draft)")
    critique = context.state.get(REVIEW_CRITIQUE, "(no critique)")
    quant = context.state.get(ANALYSIS_QUANT, "(no computed metrics)")
    filings = context.state.get(EVIDENCE_FILINGS, "(no filings data)")
    market = context.state.get(EVIDENCE_MARKET, "(no market data)")
    sentiment = context.state.get(EVIDENCE_SENTIMENT, "(no sentiment data)")
    return (
        "You are the ARGUS Refiner. Rewrite the draft thesis below to fix "
        "every problem the Critic raised. For each critique point: either "
        "ground the claim properly using the evidence/metrics given, or "
        "remove it if it can't be grounded. Never invent a number or fact "
        "to patch a gap — an honest 'the data doesn't support this' is "
        "correct, a fabricated fix is not.\n\n"
        "Output the FULL revised thesis as your response — it completely "
        "replaces the previous draft. Keep the 'not investment advice' "
        "closing line and the confidence level.\n\n"
        f"Critique to address:\n{critique}\n\n"
        f"Current draft:\n{draft}\n\n"
        f"Computed metrics:\n{quant}\n\n"
        f"Filings evidence:\n{filings}\n\n"
        f"Market evidence:\n{market}\n\n"
        f"Sentiment evidence:\n{sentiment}\n"
    )


refiner_agent = Agent(
    model=MODEL_FLASH,
    name="refiner_agent",
    description="Rewrites the draft thesis to address the Critic's critique.",
    instruction=_refiner_instruction,
    output_key=DRAFT_THESIS,
)
