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

root_agent = SequentialAgent(
    name="argus_pipeline",
    sub_agents=[
        evidence_gather,
        quant_agent,
        reconciliation_agent,
        synthesis_agent,
        review_loop,
        verifier_agent,
    ],
)
