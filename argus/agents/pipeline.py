"""The full deterministic due-diligence pipeline (Stages 1-3 + Stage 4
Part A), composed as a single unit: parallel gather → code-execution
Quant → Reconciliation → Synthesis → adversarial review loop → Groundedness
Verifier.

Lives in its own file (not argus/agent.py) so the Orchestrator (Stage 4
Part B) can import and wrap it as an AgentTool without a circular import —
argus/agent.py now just points root_agent at the Orchestrator, which reaches
for this whole pipeline as one of its tools for full-analysis requests.
"""

from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent

from argus.agents.critic import critic_agent
from argus.agents.gather import evidence_gather
from argus.agents.quant import quant_agent
from argus.agents.reconciliation import reconciliation_agent
from argus.agents.refiner import refiner_agent
from argus.agents.synthesis import synthesis_agent
from argus.agents.verifier import verifier_agent

review_loop = LoopAgent(
    name="adversarial_review",
    description="Red-teams the draft thesis and revises it until it passes or 4 iterations run out.",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=4,
)

full_analysis_pipeline = SequentialAgent(
    name="full_analysis_pipeline",
    description=(
        "Runs a complete due-diligence analysis on a company: gathers "
        "filings/market/sentiment data, computes real financial metrics, "
        "reconciles sources for contradictions, drafts and red-teams a "
        "thesis, and verifies every claim is grounded. Use this for broad "
        "requests like 'analyze X' or 'give me a full picture of X' — not "
        "for a single narrow fact (e.g. just the current price), which a "
        "single specialist tool can answer faster."
    ),
    sub_agents=[
        evidence_gather,
        quant_agent,
        reconciliation_agent,
        synthesis_agent,
        review_loop,
        verifier_agent,
    ],
)
