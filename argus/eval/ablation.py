"""Ablation study: measures how much the
critic/refiner loop (Stage 2) and code execution (Stage 1's Quant agent)
each actually contribute to groundedness, rather than assuming they help.

Builds 3 pipeline variants and runs each on the same 2 prompts:
  - baseline:      the real full_analysis_pipeline shape (gather -> quant ->
                    reconciliation -> synthesis -> critic/refiner loop ->
                    verifier).
  - no_critic:      same, with the adversarial review loop removed --
                    synthesis's draft goes straight to the verifier.
  - no_code_exec:  same, with quant_agent removed entirely -- synthesis
                    only has reconciled evidence, no BuiltInCodeExecutor
                    output, to narrate from.

Every variant is built from BaseAgent.clone() calls on the same singletons
argus/agents/pipeline.py uses (confirmed against
google/adk/agents/base_agent.py: a sub_agent may only have ONE parent, so
reusing quant_agent/synthesis_agent/etc. directly in a second tree would
raise; .clone() recursively clones with parent_agent reset to None, giving
each ablation variant a fully independent tree). Runs each variant through
a real ADK Runner + InMemorySessionService/InMemoryArtifactService, exactly
the pipeline's own entry point when the Orchestrator invokes it as an
AgentTool -- NOT a hand-rolled simulation.

Reads meta.groundedness straight from session state after each run rather
than re-deriving it: that score is check_grounding's own output
(argus/tools/claim_matcher.py), a deterministic Python matcher, not an LLM
judging itself -- the same "don't trust the system grading itself" concern
that shaped independent_groundedness in custom_metrics.py doesn't apply
here, since there's no LLM in the loop being asked to grade its own work.

Run directly: `python -m argus.eval.ablation` (needs GOOGLE_API_KEY in the
environment, same as any other live run in this project).
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from argus.agents.critic import critic_agent
from argus.agents.gather import evidence_gather
from argus.agents.quant import quant_agent
from argus.agents.reconciliation import reconciliation_agent
from argus.agents.refiner import refiner_agent
from argus.agents.synthesis import synthesis_agent
from argus.agents.verifier import verifier_agent
from argus.state_keys import DRAFT_THESIS, META_GROUNDEDNESS

_SCENARIOS = [
    ("Acme Corp (clean)", "Give me a full due-diligence analysis of Acme Corp."),
    ("Globex (planted contradiction)", "Give me a full due-diligence analysis of Globex."),
]


def _build_review_loop(name: str) -> LoopAgent:
    return LoopAgent(
        name=name,
        description="Red-teams the draft thesis and revises it until it passes or 4 iterations run out.",
        sub_agents=[critic_agent.clone(), refiner_agent.clone()],
        max_iterations=4,
    )


def build_baseline() -> SequentialAgent:
    return SequentialAgent(
        name="ablation_baseline",
        sub_agents=[
            evidence_gather.clone(),
            quant_agent.clone(),
            reconciliation_agent.clone(),
            synthesis_agent.clone(),
            _build_review_loop("ablation_review_baseline"),
            verifier_agent.clone(),
        ],
    )


def build_no_critic() -> SequentialAgent:
    return SequentialAgent(
        name="ablation_no_critic",
        sub_agents=[
            evidence_gather.clone(),
            quant_agent.clone(),
            reconciliation_agent.clone(),
            synthesis_agent.clone(),
            verifier_agent.clone(),
        ],
    )


def build_no_code_exec() -> SequentialAgent:
    return SequentialAgent(
        name="ablation_no_code_exec",
        sub_agents=[
            evidence_gather.clone(),
            reconciliation_agent.clone(),
            synthesis_agent.clone(),
            _build_review_loop("ablation_review_no_code_exec"),
            verifier_agent.clone(),
        ],
    )


async def _run_once(agent, prompt: str, app_name: str) -> dict:
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    user_id = "ablation_user"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for _event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
        pass
    final_session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session.id)
    return {
        "thesis": final_session.state.get(DRAFT_THESIS, ""),
        "groundedness": final_session.state.get(META_GROUNDEDNESS),
    }


async def run_variant(build_agent, prompt: str, app_name: str, max_attempts: int = 4) -> dict:
    """Runs one pipeline variant on one prompt via a real ADK Runner, and
    reads back the final thesis text + the Verifier's own groundedness
    score from session state.

    Retries on RESOURCE_EXHAUSTED (429): unlike the `adk eval` CLI (which
    has its own retry wrapping observed to ride out the free-tier 15/min
    ceiling across this whole project), a raw Runner script has no such
    protection -- a live run here hit a 429 that exhausted tenacity's own
    retry budget and killed the entire script, losing all prior progress.
    Rebuilds the agent tree fresh on each attempt (build_agent, not a
    pre-built instance) since a partially-run agent tree isn't safe to
    reuse after a mid-run failure.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await _run_once(build_agent(), prompt, app_name)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure gets one retry
            last_error = exc
            wait_s = 35 * attempt
            print(f"    attempt {attempt} failed ({exc!r}), waiting {wait_s}s before retry", file=sys.stderr)
            await asyncio.sleep(wait_s)
    raise RuntimeError(f"run_variant exhausted {max_attempts} attempts") from last_error


_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / ".ablation_checkpoint.json"


def _load_checkpoint() -> dict:
    if _CHECKPOINT_PATH.exists():
        import json

        return json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(results: dict) -> None:
    import json

    serializable = {f"{v}||{s}": r for (v, s), r in results.items()}
    _CHECKPOINT_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


async def main() -> None:
    variants = {
        "baseline": build_baseline,
        "no_critic": build_no_critic,
        "no_code_exec": build_no_code_exec,
    }
    checkpoint = _load_checkpoint()
    results = {}
    for key, value in checkpoint.items():
        variant_name, scenario_name = key.split("||")
        results[(variant_name, scenario_name)] = value

    for variant_name, builder in variants.items():
        for scenario_name, prompt in _SCENARIOS:
            if (variant_name, scenario_name) in results:
                print(f"--- Skipping {variant_name} / {scenario_name} (already checkpointed) ---", file=sys.stderr)
                continue
            print(f"--- Running {variant_name} / {scenario_name} ---", file=sys.stderr)
            result = await run_variant(builder, prompt, app_name=f"argus_ablation_{variant_name}")
            results[(variant_name, scenario_name)] = result
            _save_checkpoint(results)
            print(f"  groundedness={result['groundedness']}", file=sys.stderr)
            await asyncio.sleep(20)

    print("\n| Variant | Scenario | Groundedness | Unverified flags |")
    print("|---|---|---|---|")
    for (variant_name, scenario_name), result in results.items():
        unverified_count = result["thesis"].count("[UNVERIFIED:")
        print(f"| {variant_name} | {scenario_name} | {result['groundedness']} | {unverified_count} |")

    print("\n\n=== Full thesis text per run (for qualitative inspection) ===\n")
    for (variant_name, scenario_name), result in results.items():
        print(f"\n----- {variant_name} / {scenario_name} -----")
        print(result["thesis"])


if __name__ == "__main__":
    asyncio.run(main())
