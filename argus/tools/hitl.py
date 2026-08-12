"""The Human-in-the-Loop Gate: auto-approve a high-confidence
run, or suspend it for a human decision on a low-confidence one.

evaluate_gate is a DETERMINISTIC check — the same "decisions come from
code, not vibes" principle already applied to arithmetic (Quant),
contradiction detection (Reconciliation), and retry routing
(check_gather_status in orchestrator.py). The model never eyeballs a
groundedness score against a threshold and decides for itself whether
it's good enough — that comparison is arithmetic, so code does it.

request_human_approval is a LongRunningFunctionTool. Its
Python body does nothing and MUST return None — confirmed against
google/adk/flows/llm_flows/functions.py: `if (tool.is_long_running ...)
and not function_response: return None` skips auto-building a
function_response event for a falsy return. That's what leaves the call
genuinely unanswered, which is what makes the Dev UI render it as
pending. Returning a truthy "pending" dict here would auto-generate a
function_response event for this call id — and the Dev UI's own
resume-detection logic (any event with a matching functionResponse id
counts as "already answered", confirmed in the compiled frontend's
restorePendingLongRunningCalls()) would then treat the call as already
resolved and never show the approve/reject/redirect box at all. So the
orchestrator's own instruction — not this tool's return value, which the
model never actually sees — is what has to explain the ask and the
expected reply format to the user, in the model's own response text.

The human's eventual reply (typed into the Dev UI's pending-call input)
becomes this call's function_response content directly: the Dev UI
JSON-parses it if it can, otherwise wraps it as {"result": "<text>"}.
Either shape then just shows up as this tool's "result" in the model's
context on the next turn — the model reads it and calls
record_human_decision with its own best reading of what was decided.
"""

from google.adk.tools.tool_context import ToolContext

from argus.config import GROUNDEDNESS_THRESHOLD
from argus.state_keys import META_GROUNDEDNESS, META_HUMAN, REVIEW_CRITIQUE

_VALID_DECISIONS = {"approve", "reject", "redirect"}


def _is_pass(critique: str) -> bool:
    """The Critic's own instruction (critic.py) mandates responding with
    exactly the word PASS on a clean review. Tolerates a trailing period/
    whitespace/case, since that's the only slack real model output has
    shown — anything else (e.g. "PASS, but...") is correctly NOT a pass."""
    return critique.strip().rstrip(".").upper() == "PASS"


def decide_gate(groundedness: float | None, critique: str, threshold: float) -> dict:
    """Pure gate logic: does this run clear the bar for automatic release?

    Fails closed — anything not positively confirmed escalates to a human:
    a missing groundedness score, a score below threshold, or a critique
    that isn't a clean PASS, all route to escalation. Only a fully clean
    run auto-approves.

    Args:
        groundedness: meta.groundedness, or None if never computed.
        critique: review.critique, the Critic's final verdict text.
        threshold: the minimum groundedness ratio to auto-approve.

    Returns:
        dict with "auto_approved" (bool), "groundedness", "critic_passed",
        and "reason" (why it was or wasn't auto-approved).
    """
    critic_passed = _is_pass(critique)
    reasons = []
    if groundedness is None:
        reasons.append("groundedness score is missing")
    elif groundedness < threshold:
        reasons.append(f"groundedness {groundedness:.4f} is below the {threshold} threshold")
    if not critic_passed:
        reasons.append("the critic did not record a clean PASS")

    if reasons:
        return {
            "auto_approved": False,
            "groundedness": groundedness,
            "critic_passed": critic_passed,
            "reason": "; ".join(reasons),
        }
    return {
        "auto_approved": True,
        "groundedness": groundedness,
        "critic_passed": True,
        "reason": "groundedness and critic both cleared the bar",
    }


def evaluate_gate(tool_context: ToolContext) -> dict:
    """Call this immediately after check_gather_status confirms gather_ok,
    before presenting or saying anything about the result. Deterministically
    decides whether this run's groundedness and critic verdict clear the
    bar for automatic release, or need a human's review first. Never skip
    this and never decide for yourself whether a result looks trustworthy
    enough to show — this check owns that decision.

    Returns:
        dict with "auto_approved" (bool) and "reason". If auto_approved is
        true, meta.human is already recorded for you — go ahead and
        present the result. If false, do NOT present anything yet; call
        request_human_approval next.
    """
    groundedness = tool_context.state.get(META_GROUNDEDNESS)
    critique = tool_context.state.get(REVIEW_CRITIQUE, "")
    result = decide_gate(groundedness, critique, GROUNDEDNESS_THRESHOLD)
    if result["auto_approved"]:
        tool_context.state[META_HUMAN] = {
            "decision": "auto_approved",
            "reason": result["reason"],
        }
    return result


def request_human_approval(concern: str, tool_context: ToolContext) -> None:
    """Pause the run and ask a human to review a result evaluate_gate did
    NOT auto-approve. This tool never returns a value to you — the run
    genuinely suspends here until a human responds.

    Explaining the situation in your response text is NOT a substitute for
    calling this tool — they are two parts of one action. Text alone does
    not pause anything; always call this IN THE SAME TURN as your
    explanation, never on its own in a later turn.

    In that same response text, you MUST tell the user: why review is
    needed (evaluate_gate's reason), and exactly how to reply — a JSON
    object like {"decision": "approve"}, {"decision": "reject", "reason":
    "..."}, or {"decision": "redirect", "scope": "..."}. The run only
    continues once a human supplies that.

    Args:
        concern: one sentence naming why this run needs review.
    """
    return None


def record_human_decision(decision: str, reason: str, tool_context: ToolContext) -> dict:
    """Call this right after a human's reply to request_human_approval
    arrives, to record what they decided into ARGUS's state. Their raw
    reply may be structured JSON or free text wrapped as {"result": ...}
    — read it and pass your own best interpretation here.

    Args:
        decision: "approve", "reject", or "redirect" — your best reading
            of the human's intent. An unclear reply should be treated as
            "reject" (never guess in favor of releasing an unreviewed
            result).
        reason: the human's stated reason or requested scope, if any
            (empty string if none given).

    Returns:
        dict with the normalized "decision" that was recorded.
    """
    normalized = decision.strip().lower()
    if normalized not in _VALID_DECISIONS:
        normalized = "reject"  # fail closed on anything unrecognized
    tool_context.state[META_HUMAN] = {"decision": normalized, "reason": reason}
    return {"decision": normalized, "reason": reason}
