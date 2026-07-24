"""The Groundedness Verifier (spec P4, agent #12): the measurable-trust
layer. Where the Critic (Stage 2) subjectively red-teams the draft's
reasoning, this agent does something narrower and deterministic — checks
every number the draft states as fact against a real matcher tool, and
produces an auditable meta.groundedness score. Flags ungrounded claims in
place rather than silently deleting them (PLAN.md's named pitfall for this
agent: over-rejecting is worse than a visible flag a human can review).
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH
from argus.state_keys import DRAFT_THESIS
from argus.tools.claim_matcher import check_grounding


def _verifier_instruction(context: ReadonlyContext) -> str:
    draft = context.state.get(DRAFT_THESIS, "(no draft)")
    return (
        "You are the ARGUS Groundedness Verifier. Your job is narrow and "
        "mechanical, not a subjective quality review — that already "
        "happened in the critique loop.\n\n"
        "1. Read the draft thesis below and list every distinct number it "
        "states as fact (percentages, dollar amounts, prices, years — "
        "convert each to a plain float, e.g. 7.52 for '7.52%').\n"
        "2. Call check_grounding with that full list of numbers.\n"
        "3. Rewrite the draft: for every number the tool reports as "
        "ungrounded, wrap that specific claim with '[UNVERIFIED: ...]' — "
        "do not delete it and do not change the number, just flag it so a "
        "human reviewer can see exactly what wasn't confirmed. Every "
        "grounded claim stays exactly as it was.\n"
        "4. Output the full thesis text with those flags applied — this "
        "replaces the draft. No commentary outside the thesis itself.\n\n"
        f"Draft thesis:\n{draft}\n"
    )


verifier_agent = Agent(
    model=MODEL_FLASH,
    name="verifier_agent",
    description="Checks every numeric claim in the draft against the evidence via a deterministic matcher tool.",
    instruction=_verifier_instruction,
    tools=[check_grounding],
    output_key=DRAFT_THESIS,
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
