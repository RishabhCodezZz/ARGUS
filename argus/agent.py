from google.adk.agents.sequential_agent import SequentialAgent

from argus.agents.gather import evidence_gather
from argus.agents.quant import quant_agent
from argus.agents.synthesis import synthesis_agent

root_agent = SequentialAgent(
    name="argus_pipeline",
    sub_agents=[evidence_gather, quant_agent, synthesis_agent],
)
