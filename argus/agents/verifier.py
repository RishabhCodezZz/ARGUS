"""The Groundedness Verifier: the measurable-trust
layer. Where the Critic (Stage 2) subjectively red-teams the draft's
reasoning, this agent does something narrower and deterministic — checks
every number the draft states as fact against a real matcher tool, and
produces an auditable meta.groundedness score. Flags ungrounded claims in
place rather than silently deleting them: over-rejecting is worse than a
visible flag a human can review.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_REVIEW
from argus.state_keys import DRAFT_THESIS
from argus.tools.claim_matcher import check_grounding


def _verifier_instruction(context: ReadonlyContext) -> str:
    draft = context.state.get(DRAFT_THESIS, "(no draft)")
    return (
        "You are the ARGUS Groundedness Verifier. Your job is narrow and "
        "mechanical, not a subjective quality review — that already "
        "happened in the critique loop.\n\n"
        "1. Call check_grounding — it takes no arguments; it reads the "
        "draft thesis directly from session state and extracts every "
        "number in it itself, deterministically. You do not need to (and "
        "should not try to) list the numbers yourself.\n"
        "2. Rewrite the draft below: for every number the tool's "
        "'ungrounded' list reports, find that exact claim in the draft "
        "and wrap it with '[UNVERIFIED: ...]' — do not delete it and do "
        "not change the number, just flag it so a human reviewer can see "
        "exactly what wasn't confirmed. Every grounded claim stays "
        "exactly as it was.\n"
        "3. Output the full thesis text with those flags applied — this "
        "replaces the draft. No commentary outside the thesis itself.\n\n"
        f"Draft thesis:\n{draft}\n"
    )


verifier_agent = Agent(
    model=MODEL_REVIEW,
    name="verifier_agent",
    description="Checks every numeric claim in the draft against the evidence via a deterministic matcher tool.",
    instruction=_verifier_instruction,
    tools=[check_grounding],
    output_key=DRAFT_THESIS,
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
