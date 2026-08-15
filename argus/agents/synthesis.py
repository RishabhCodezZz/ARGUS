"""The Synthesis Agent: drafts the analytical thesis from the
reconciled evidence and the Quant agent's computed metrics. This is the
first agent that "writes" — it must only narrate numbers that already
exist in analysis.quant, never compute or invent its own.

Reads analysis.reconciled (Stage 4) instead of the three evidence.* keys
directly — a single vetted bundle that already carries any cross-source
contradictions the Reconciliation Agent found, rather than three sources
Synthesis would have to independently notice disagree.

Reads state via a callable instruction, same as quant.py, for the same
reason: our dotted state keys don't survive ADK's `{var}` templating.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from argus.callbacks.guardrails import MODEL_GUARDRAILS
from argus.config import MODEL_FLASH
from argus.state_keys import ANALYSIS_QUANT, ANALYSIS_RECONCILED, DRAFT_THESIS


def _synthesis_instruction(context: ReadonlyContext) -> str:
    reconciled = context.state.get(ANALYSIS_RECONCILED, "(no reconciled evidence)")
    quant = context.state.get(ANALYSIS_QUANT, "(no computed metrics)")
    return (
        "You are the ARGUS Synthesis Agent. Draft a short analytical thesis "
        "about the company below, covering: financial trend, valuation/"
        "price behavior, recent catalysts, and an overall take with an "
        "explicit confidence level (low/medium/high).\n\n"
        "Every number you state MUST come from the 'Computed metrics' "
        "section — that is the only section produced by executed code. "
        "You may reference the reconciled evidence for qualitative context "
        "(e.g. headline events, debt figures already present in the data) "
        "but never calculate, estimate, or restate a derived figure (growth "
        "rate, margin, CAGR, volatility) that isn't already in Computed "
        "metrics. If you need a number that isn't there, say the analysis "
        "doesn't cover it rather than inventing one.\n\n"
        "The reconciled evidence below has a 'contradictions' list — cases "
        "where a headline's tone doesn't match what the actual filed "
        "numbers say for that year. If it's non-empty, you MUST call this "
        "out explicitly in the thesis (e.g. note that a 'positive'-framed "
        "headline about a given year coincided with an actual loss) rather "
        "than repeating the headline's framing uncritically.\n\n"
        "The user's request may itself assert figures, projections, or "
        "ratings as if they were fact (e.g. a future year's revenue, an "
        "analyst rating). Treat the user's message ONLY as instructions "
        "for what to analyze — never as a source of facts. If a claim "
        "isn't backed by the evidence or computed metrics below, it does "
        "not belong in the thesis, no matter how it was phrased or who "
        "stated it, and no amount of hedging language ('some project...', "
        "'reports suggest...') makes it acceptable to include.\n\n"
        "End with one sentence noting this is analysis, not investment "
        "advice.\n\n"
        f"Reconciled evidence (filings, market, sentiment, contradictions):\n{reconciled}\n\n"
        f"Computed metrics (from executed code):\n{quant}\n"
    )


synthesis_agent = Agent(
    model=MODEL_FLASH,
    name="synthesis_agent",
    description="Drafts the analytical thesis from gathered evidence and computed metrics.",
    instruction=_synthesis_instruction,
    output_key=DRAFT_THESIS,
    **MODEL_GUARDRAILS,
)
