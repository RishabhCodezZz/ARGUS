"""The Retrieval Agent (spec P5, agent #3): the optional RAG path. Wrapped
as an AgentTool the Orchestrator MAY call for qualitative/background
questions a structured data tool can't answer — competitive position,
strategic outlook, leadership changes. Never for numbers; the filings and
market specialists own those.

Deliberately the smallest, most stub-like agent in the project — the spec
itself frames RAG as "one subordinate tool among many," not a required
path (see argus/tools/knowledge_base.py for what "stub" means here).
"""

from google.adk.agents.llm_agent import Agent

from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH
from argus.state_keys import RETRIEVAL_CTX
from argus.tools.knowledge_base import retrieve_knowledge

retrieval_agent = Agent(
    model=MODEL_FLASH,
    name="retrieval_agent",
    description=(
        "Searches an internal knowledge base of analyst notes for "
        "qualitative background — competitive landscape, strategic "
        "outlook, leadership/governance. Not for financial data or numbers."
    ),
    instruction=(
        "Identify the company and what background context is wanted, "
        "then call retrieve_knowledge with that company and a short query "
        "describing what you're looking for. Report back the retrieved "
        "documents' content directly — do not add claims or commentary "
        "beyond what they say. If nothing matched, say so plainly rather "
        "than guessing or inventing background information."
    ),
    tools=[retrieve_knowledge],
    output_key=RETRIEVAL_CTX,
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)
