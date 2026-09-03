"""Runtime memory — distilled per-entity findings persisted via ADK's
MemoryService, so a later run on the same entity "starts warm". A
genuinely different store from session.state: state.* is per-run working
memory that a session's own agents read/write via state_delta/output_key;
this store persists ACROSS sessions but only for the life of the running
process. It is ADK's InMemoryMemoryService — an in-process store that is
CLEARED ON RESTART, not a durable database. A cross-restart store would
need a custom SQLite-backed MemoryService (deliberately deferred — the
only non-in-memory implementations ADK ships are Vertex-AI-backed, which
conflicts with this project's fully-local decision; see DECISIONS.md).
It is written/read through Context.add_events_to_memory /
Context.search_memory, and there is no state_keys.py entry for it because
it never touches session.state at all.

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

from argus.state_keys import DRAFT_THESIS, EVIDENCE_FILINGS, META_GROUNDEDNESS, META_HUMAN


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
    """Formats the one distilled sentence persisted to runtime memory per run.

    Just the headline (first line) of the thesis, not the full memo — this
    is meant to be skimmed by a future recall, not re-read in full.
    """
    score = f"{groundedness:.2f}" if isinstance(groundedness, (int, float)) else "unknown"
    headline = thesis.strip().splitlines()[0] if thesis and thesis.strip() else "(no thesis produced)"
    return f"Prior ARGUS finding on {entity} (groundedness {score}): {headline}"


async def recall_prior_findings(entity: str, tool_context: ToolContext) -> dict:
    """Check ARGUS's runtime memory for a prior finding about this company
    from an earlier session in the same running process, before running a
    fresh analysis.

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
    runtime memory, so a future request about the same company starts
    warm. Reads the result straight from this session's own state; takes
    no arguments.

    Returns:
        dict with "remembered" (bool) — false if there's no usable entity
        in state, or if the result was never approved (see below), both
        safe to ignore.
    """
    # Same code-level enforcement as save_memo (argus/tools/memo.py) --
    # a rejected result must never be persisted, and the ONLY thing that
    # previously stopped it was the calling agent's own instructions.
    # Fail closed: anything other than an explicit approve/auto_approve
    # blocks the write, including a missing decision entirely.
    human_decision = tool_context.state.get(META_HUMAN) or {}
    if human_decision.get("decision") not in ("approve", "auto_approved"):
        return {"remembered": False}

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
