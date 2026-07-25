"""Long-term memory (spec SM2) — distilled per-entity findings persisted via
ADK's MemoryService, so a later run on the same entity "starts warm". A
genuinely different store from session.state: state.* is short-term,
per-run working memory (SM1) that a session's own agents read/write via
state_delta/output_key; memory persists ACROSS sessions and is written/read
through Context.add_events_to_memory / Context.search_memory instead —
there is no state_keys.py entry for it because it never touches
session.state at all.

Writes ONE distilled sentence per run (entity + groundedness + thesis
headline), not the whole noisy multi-agent trace via
add_session_to_memory() — dumping every tool call and evidence JSON blob in
would make search return noise instead of the one line that actually
matters. InMemoryMemoryService (like Part D's retrieval stub) does keyword
matching, not real semantic search — an honest stub, not a gap to hide.

BaseMemoryService.add_memory() looks like the obvious write API (it takes
a plain MemoryEntry) but InMemoryMemoryService never overrides it, so it
inherits the abstract base's unconditional `raise NotImplementedError`.
add_events_to_memory() IS concretely implemented, but it takes a real ADK
Event, not a MemoryEntry — the same Event type reconciliation.py already
yields from its custom BaseAgent.

remember_finding is a tool on the ORCHESTRATOR, not an after_agent_callback
on full_analysis_pipeline — that was the first design, and it silently
persisted into a throwaway store. Confirmed against
google/adk/tools/agent_tool.py: AgentTool.run_async runs the wrapped
sub-agent through a brand-new `Runner(..., memory_service=
InMemoryMemoryService(), ...)` — a fresh, empty instance, NOT the app's
real one from `adk web --memory_service_uri`. artifact_service is
explicitly forwarded to the parent via ForwardingArtifactService(
tool_context) in that same constructor call; memory_service is not. So a
callback anywhere inside the pipeline can never reach the real memory
store — only code running in the OUTER (orchestrator-level) invocation
does. State, by contrast, DOES cross this boundary: AgentTool.run_async
copies each inner event's state_delta onto the parent's tool_context.state
as it streams out, which is why remember_finding can just read
EVIDENCE_FILINGS/DRAFT_THESIS/META_GROUNDEDNESS from the orchestrator's
own state after full_analysis_pipeline returns, same as
check_gather_status already does.
"""

import json
import uuid

from google.adk.events import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from argus.state_keys import DRAFT_THESIS, EVIDENCE_FILINGS, META_GROUNDEDNESS


def _parse_state_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_entity_name(filings: dict) -> str | None:
    """Pulls the company display name out of a parsed evidence.filings dict.

    Returns None if filings is empty/errored (e.g. gather failed) — nothing
    worth remembering in that case.
    """
    name = filings.get("company")
    return name if isinstance(name, str) and name.strip() else None


def build_distilled_memory(entity: str, thesis: str, groundedness: float | None) -> str:
    """Formats the one distilled sentence persisted to long-term memory per run.

    Just the headline (first line) of the thesis, not the full memo — this
    is meant to be skimmed by a future recall, not re-read in full.
    """
    score = f"{groundedness:.2f}" if isinstance(groundedness, (int, float)) else "unknown"
    headline = thesis.strip().splitlines()[0] if thesis and thesis.strip() else "(no thesis produced)"
    return f"Prior ARGUS finding on {entity} (groundedness {score}): {headline}"


async def recall_prior_findings(entity: str, tool_context: ToolContext) -> dict:
    """Check ARGUS's long-term memory for a prior finding about this company
    from an earlier session, before running a fresh analysis.

    This is NOT financial data and never overrides what the current run's
    own freshly-gathered evidence says — it only lets the answer note
    continuity with past work (e.g. "ARGUS looked at this company before").
    If nothing is on file, that's a normal result, not an error.

    Args:
        entity: the company name to check for prior findings.

    Returns:
        dict with "found" (bool) and "findings" (list of prior distilled
        finding strings, empty if nothing on file for this entity).
    """
    response = await tool_context.search_memory(entity)
    findings = [
        part.text
        for entry in response.memories
        for part in (entry.content.parts or [])
        if part.text
    ]
    return {"found": bool(findings), "findings": findings}


async def remember_finding(tool_context: ToolContext) -> dict:
    """Call this once, right after a broad analysis has completed
    successfully (check_gather_status returned gather_ok=true) — distills
    the just-finished run into one sentence and saves it to ARGUS's
    long-term memory, so a future request about the same company starts
    warm. Reads the result straight from this session's own state; takes
    no arguments.

    Returns:
        dict with "remembered" (bool) — false only if there's no usable
        entity in state (e.g. called at the wrong time), which is safe to
        ignore.
    """
    filings = _parse_state_json(tool_context.state.get(EVIDENCE_FILINGS))
    entity = extract_entity_name(filings)
    if entity is None:
        return {"remembered": False}

    thesis = tool_context.state.get(DRAFT_THESIS, "")
    groundedness = tool_context.state.get(META_GROUNDEDNESS)
    distilled = build_distilled_memory(entity, thesis, groundedness)

    # A synthetic Event, not a real one from the run's own history — this is
    # what add_events_to_memory actually accepts (see module docstring).
    # Needs an explicit unique id: InMemoryMemoryService dedupes incoming
    # events by id within a session, and every Event defaults to id='' — a
    # second call in the same session would silently no-op without this.
    memory_event = Event(
        id=uuid.uuid4().hex,
        author="argus_memory",
        content=types.Content(role="model", parts=[types.Part(text=distilled)]),
    )
    await tool_context.add_events_to_memory(events=[memory_event])
    return {"remembered": True, "entity": entity}
