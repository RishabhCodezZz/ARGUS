"""The Red-Team Critic (spec P3, agent #10): attacks the drafted thesis
before a human ever sees it. Checks the draft against the actual evidence
and computed metrics (not just internal coherence) for unsupported leaps,
cherry-picking, or claims that don't trace back to a source.

Only the Critic carries exit_loop — per spec OR3, the CRITIC is what
signals PASS, not the Refiner. Attaching it here means a clean pass stops
the loop immediately without wasting a redundant Refiner call.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext

from argus.config import MODEL_FLASH
from argus.state_keys import (
    ANALYSIS_QUANT,
    DRAFT_THESIS,
    EVIDENCE_FILINGS,
    EVIDENCE_MARKET,
    EVIDENCE_SENTIMENT,
    REVIEW_CRITIQUE,
)


def exit_loop(tool_context: ToolContext) -> dict:
    """Call this ONLY when the draft thesis passes review with no issues —
    every claim traces to the evidence or computed metrics, no unsupported
    leaps, no cherry-picking. This ends the critique/refine loop early.
    """
    tool_context.actions.escalate = True
    # No skip_summarization: we want the model's follow-up "PASS" text to
    # still land in review.critique via output_key. With it skipped, the
    # turn ends at the empty tool return and output_key captures that
    # instead — verified empirically (Stage 2 dev): state showed "{}", not
    # "PASS". One extra LLM call is a fine trade for a legible state trail.
    return {}


def _critic_instruction(context: ReadonlyContext) -> str:
    draft = context.state.get(DRAFT_THESIS, "(no draft)")
    quant = context.state.get(ANALYSIS_QUANT, "(no computed metrics)")
    filings = context.state.get(EVIDENCE_FILINGS, "(no filings data)")
    market = context.state.get(EVIDENCE_MARKET, "(no market data)")
    sentiment = context.state.get(EVIDENCE_SENTIMENT, "(no sentiment data)")
    return (
        "You are the ARGUS Red-Team Critic. Your job is to attack the "
        "draft thesis below, not to be agreeable. Check every claim in it "
        "against the evidence and computed metrics provided:\n"
        "- Does every number in the draft actually appear in 'Computed "
        "metrics' or the raw evidence? Flag any that don't.\n"
        "- Are there unsupported leaps — conclusions that don't follow "
        "from the data?\n"
        "- Is there cherry-picking — positive or negative evidence "
        "mentioned selectively while contradicting evidence is ignored?\n"
        "- Are qualitative claims (sentiment, catalysts) actually backed "
        "by the headlines given, or overstated?\n\n"
        "If the draft has NO issues: these are two parts of ONE action, "
        "always do both, never just one — (1) call the exit_loop tool, "
        "AND (2) respond with exactly: PASS\n\n"
        "If it has issues, do NOT call exit_loop. Respond with a "
        "numbered list of concrete, specific problems the Refiner must "
        "fix — not vague feedback like 'improve clarity'.\n\n"
        f"Draft thesis:\n{draft}\n\n"
        f"Computed metrics:\n{quant}\n\n"
        f"Filings evidence:\n{filings}\n\n"
        f"Market evidence:\n{market}\n\n"
        f"Sentiment evidence:\n{sentiment}\n"
    )


critic_agent = Agent(
    model=MODEL_FLASH,
    name="critic_agent",
    description="Red-teams the draft thesis against the evidence; PASS or a structured critique.",
    instruction=_critic_instruction,
    tools=[exit_loop],
    output_key=REVIEW_CRITIQUE,
)
