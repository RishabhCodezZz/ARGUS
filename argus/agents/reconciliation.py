"""The Reconciliation Agent: cross-checks facts across
sources and flags contradictions before Synthesis ever drafts a word. A
custom BaseAgent, not an LlmAgent — the whole point is a check no model
should grade itself on. See argus/tools/reconciler.py for the actual
detection logic; this file is just the ADK wiring around it.

Writes the complete evidence bundle (filings + market + sentiment, plus
any contradictions found) to analysis.reconciled, which Synthesis now
reads instead of the three evidence.* keys separately — a single, vetted
source of truth rather than three that might disagree.
"""

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from argus.state_keys import (
    ANALYSIS_RECONCILED,
    EVIDENCE_FILINGS,
    EVIDENCE_MARKET,
    EVIDENCE_SENTIMENT,
)
from argus.tools.reconciler import find_contradictions


def _parse_state_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class ReconciliationAgent(BaseAgent):
    """Deterministic cross-check of evidence.filings against
    evidence.sentiment for contradictions; passes through all gathered
    evidence (plus any flags) as a single analysis.reconciled bundle.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        filings = _parse_state_json(ctx.session.state.get(EVIDENCE_FILINGS))
        market = _parse_state_json(ctx.session.state.get(EVIDENCE_MARKET))
        sentiment = _parse_state_json(ctx.session.state.get(EVIDENCE_SENTIMENT))

        contradictions = find_contradictions(filings, sentiment)

        reconciled = {
            "filings": filings,
            "market": market,
            "sentiment": sentiment,
            "contradictions": contradictions,
        }

        summary = (
            f"Reconciliation complete: {len(contradictions)} contradiction(s) found."
            if contradictions
            else "Reconciliation complete: no contradictions found."
        )

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=summary)]),
            actions=EventActions(
                state_delta={ANALYSIS_RECONCILED: json.dumps(reconciled)}
            ),
        )


reconciliation_agent = ReconciliationAgent(
    name="reconciliation_agent",
    description=(
        "Cross-checks evidence for contradictions (e.g. a 'positive' "
        "headline about a year that was actually a loss) before Synthesis drafts."
    ),
)
