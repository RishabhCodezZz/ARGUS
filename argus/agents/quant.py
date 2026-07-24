"""The Quantitative Agent (spec P2): the deterministic truth layer. Reads
gathered evidence from state and computes real numbers via Gemini's
built-in code execution (real pandas/numpy) — never estimates (spec G3/TR3,
CLAUDE.md's non-negotiable principle #1).

BuiltInCodeExecutor may only be used alone on an agent — it cannot be
combined with other tools (verified against current ADK docs). So instead
of a FunctionTool, this agent reads prior state through a callable
instruction (an InstructionProvider) that indexes context.state directly.
That also sidesteps ADK's `{var}` instruction-templating, which only
matches valid Python identifiers and silently fails to substitute our
dotted state keys like "evidence.filings" (verified against ADK source).
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.code_executors import BuiltInCodeExecutor

from argus.config import MODEL_FLASH
from argus.state_keys import ANALYSIS_QUANT, EVIDENCE_FILINGS, EVIDENCE_MARKET


def _quant_instruction(context: ReadonlyContext) -> str:
    filings = context.state.get(EVIDENCE_FILINGS, "(no filings data gathered)")
    market = context.state.get(EVIDENCE_MARKET, "(no market data gathered)")
    return (
        "You are the ARGUS Quantitative Agent, the project's deterministic "
        "truth layer. You MUST use the code executor to compute every "
        "number below with real Python (pandas/numpy) — never state a "
        "number you did not compute in executed code. If the data below is "
        "missing or incomplete, say so instead of guessing.\n\n"
        f"Filings data (JSON):\n{filings}\n\n"
        f"Market data (JSON):\n{market}\n\n"
        "Using code, compute and report:\n"
        "1. Year-over-year revenue growth rate for each year after the first.\n"
        "2. Net income margin (net_income / revenue) for every year.\n"
        "3. Revenue CAGR across the full fiscal_years window.\n"
        "4. Total debt change from the first year to the last.\n"
        "5. Monthly price volatility: the standard deviation of "
        "month-over-month percentage returns from the market data.\n\n"
        "Present the results as a clear, labeled list of computed values."
    )


quant_agent = Agent(
    model=MODEL_FLASH,
    name="quant_agent",
    description=(
        "Computes real financial metrics via executed Python code — the "
        "deterministic truth layer."
    ),
    instruction=_quant_instruction,
    code_executor=BuiltInCodeExecutor(),
    output_key=ANALYSIS_QUANT,
)
