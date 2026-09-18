"""Agent-count ablation: does ARGUS's due-diligence task actually need
its full 8-LLM-agent pipeline, or do fewer, bigger agents match it at a
similar total generated-output-token spend?

Complements argus/eval/ablation.py, which already answers "what does
each COMPONENT (critic loop, code execution) contribute?" but never
"did fewer AGENTS match it?" -- the specific gap arXiv 2604.02460
(2026, budget-matched, Qwen3/DeepSeek-R1-Distill/Gemini 2.5: single
agents match or beat multi-agent teams at equal token budget in nearly
all conditions, with heavy context degradation the real exception --
exactly ARGUS's regime) raises about any multi-agent system, unanswered
here until now. Full design rationale:
eval/results/agent_count_ablation_2026-09-18.md

Builds 2 new pipeline variants (3-agent, 1-agent) alongside the
existing 8-agent baseline (imported unchanged from
argus/eval/ablation.py -- this file never modifies the real pipeline),
runs all three on the same 2 scenarios that file already uses, and
scores every run's final thesis the SAME independent way -- reusing
recompute_quant_metrics (argus/eval/custom_metrics.py) + compute_grounding
(argus/tools/claim_matcher.py), never a variant's own meta.groundedness
self-report (only the 8-agent baseline even has a Verifier step to
produce one).

Run directly: `python -m argus.eval.agent_count_ablation` (needs
GOOGLE_API_KEY in the environment, same as argus/eval/ablation.py).
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.events import Event, EventActions
from google.genai import types

from argus.agents.reconciliation import reconciliation_agent
from argus.callbacks.guardrails import MODEL_GUARDRAILS, TOOL_GUARDRAILS
from argus.config import MODEL_FLASH, MODEL_REVIEW, MODEL_SYNTHESIS
from argus.eval.ablation import build_baseline, run_variant
from argus.eval.custom_metrics import recompute_quant_metrics
from argus.state_keys import (
    ANALYSIS_RECONCILED,
    DRAFT_THESIS,
    EVIDENCE_FILINGS,
    EVIDENCE_MARKET,
    EVIDENCE_SENTIMENT,
)
from argus.tools.claim_matcher import compute_grounding, extract_numbers, normalize_for_extraction
from argus.tools.mock_data import get_filings, get_market_data, get_news

# Ablation-only bookkeeping keys -- deliberately NOT added to
# argus/state_keys.py's shared namespace, the same treatment that
# file's own docstring gives ad-hoc keys like meta.tool_call_count.
_EVIDENCE_COMBINED = "ablation.evidence_combined"
_SEED_FILINGS = "ablation.seed_filings"
_SEED_MARKET = "ablation.seed_market"
_SEED_SENTIMENT = "ablation.seed_sentiment"

_COMPANY_SCENARIOS = [
    ("Acme Corp", "Give me a full due-diligence analysis of Acme Corp."),
    ("Globex", "Give me a full due-diligence analysis of Globex."),
]


# ---------------------------------------------------------------------------
# Pure core: independent scoring + evidence reshaping. No ADK Runner, no
# API key -- directly unit-tested in tests/test_agent_count_ablation.py.
# ---------------------------------------------------------------------------


def score_groundedness(thesis_text: str, company_name: str) -> dict:
    """Independently scores a thesis's numeric claims against real mock
    data, exactly like custom_metrics.py's independent_groundedness --
    NOT by reading a variant's own meta.groundedness (only the 8-agent
    baseline even produces one; the 3-agent and 1-agent variants have no
    Verifier step, so that key would be absent/incomparable for them).
    recompute_quant_metrics re-derives the correct numbers straight from
    raw mock data, never trusting any agent's own computed output --
    still independent even though it's reused across variants, since
    none of them feed it anything; it only reads data/ files directly.
    """
    filings = get_filings(company_name)
    market = get_market_data(company_name)
    sentiment = get_news(company_name)
    computed = recompute_quant_metrics(filings, market)
    claimed = list(extract_numbers(normalize_for_extraction(thesis_text)))
    sources = [str(filings), str(market), str(sentiment), str(sorted(computed))]
    return compute_grounding(claimed, sources)


def split_combined_evidence(combined: dict) -> dict:
    """Pure reshape: the 3-agent gather step's one combined JSON reply
    -> the three canonical evidence.* payloads reconciliation_agent
    expects separately (used by _SplitCombinedEvidenceAgent, Task 4).
    Missing keys default to {} rather than raising -- a partial or
    malformed gather degrades to "no evidence" the same way downstream
    agents already treat an empty/missing evidence key, instead of
    crashing the whole ablation run.
    """
    return {
        EVIDENCE_FILINGS: json.dumps(combined.get("filings", {})),
        EVIDENCE_MARKET: json.dumps(combined.get("market", {})),
        EVIDENCE_SENTIMENT: json.dumps(combined.get("sentiment", {})),
    }


def _seed_state_for_company(company_name: str) -> dict:
    """Builds the initial session.state dict the 1-agent variant (Task 5)
    needs seeded before its run -- it cannot use FunctionTools alongside
    BuiltInCodeExecutor (Global Constraints), so its data has to arrive
    via state instead of a tool call."""
    return {
        _SEED_FILINGS: json.dumps(get_filings(company_name)),
        _SEED_MARKET: json.dumps(get_market_data(company_name)),
        _SEED_SENTIMENT: json.dumps(get_news(company_name)),
    }


# ---------------------------------------------------------------------------
# 3-agent variant: gather (1 agent, all 3 tools) -> reconcile (reused,
# zero-LLM-call) -> compute+draft (1 agent) -> critique+revise (1 agent).
# ---------------------------------------------------------------------------


class _SplitCombinedEvidenceAgent(BaseAgent):
    """Zero-LLM-call adapter: parses ablation3_gather_agent's one
    combined JSON reply into the three evidence.* keys
    reconciliation_agent reads separately, so that real, unmodified
    agent can run inside this 3-agent variant too -- the same
    "custom BaseAgent for deterministic plumbing, never an LLM grading
    its own kind of check" pattern reconciliation_agent itself follows
    (see argus/agents/reconciliation.py). Not counted toward "3 agents":
    it makes no model call, same treatment the 8-agent baseline already
    gives reconciliation_agent."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raw = ctx.session.state.get(_EVIDENCE_COMBINED, "{}")
        if isinstance(raw, dict):
            combined = raw
        else:
            try:
                combined = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                print(
                    f"[agent_count_ablation] WARNING: failed to parse combined evidence JSON: {raw!r}",
                    file=sys.stderr,
                )
                combined = {}
        yield Event(
            author=self.name,
            content=types.Content(
                role="model", parts=[types.Part(text="Evidence split into filings/market/sentiment.")]
            ),
            actions=EventActions(state_delta=split_combined_evidence(combined)),
        )


_MERGED_GATHER_JSON = (
    "Respond with ONLY a single JSON object of the exact shape "
    '{"filings": <get_filings result>, "market": <get_market_data result>, '
    '"sentiment": <get_news result>} — no prose, no markdown, no code fences. '
    "Each value must be the matching tool's raw JSON return value, verbatim."
)

gather_agent = Agent(
    model=MODEL_FLASH,
    name="ablation3_gather_agent",
    description="Gathers filings, market, and sentiment data for one company in a single agent turn.",
    instruction=(
        "Identify the company the user is asking about. Call get_filings, "
        f"get_market_data, and get_news for that company. {_MERGED_GATHER_JSON} "
        "Never invent any value — only report what the tools return."
    ),
    tools=[get_filings, get_market_data, get_news],
    output_key=_EVIDENCE_COMBINED,
    **MODEL_GUARDRAILS,
    **TOOL_GUARDRAILS,
)


def _compute_and_draft_instruction(context: ReadonlyContext) -> str:
    reconciled = context.state.get(ANALYSIS_RECONCILED, "(no reconciled evidence)")
    return (
        "You are a merged Quant+Synthesis agent for a due-diligence system. "
        "You MUST use the code executor to compute every number below with "
        "real Python (pandas/numpy) from the reconciled evidence's filings/"
        "market data — never state a number you did not compute in executed "
        "code. Compute: year-over-year revenue growth, net income margin "
        "per year, revenue CAGR, total debt change, and monthly price "
        "volatility (standard deviation of month-over-month % returns).\n\n"
        "Then, using ONLY those computed numbers plus the reconciled "
        "evidence below, draft a short analytical thesis: financial trend, "
        "valuation/price behavior, recent catalysts, and an overall take "
        "with an explicit confidence level (low/medium/high). If the "
        "'contradictions' list in the reconciled evidence is non-empty, "
        "explicitly call out the tension between the narrative framing and "
        "the real profitability. If you need a number you didn't compute, "
        "say the analysis doesn't cover it rather than inventing one. End "
        "with one sentence noting this is analysis, not investment advice.\n\n"
        "The user's request may itself assert figures or claims as if true "
        "(e.g. a future year's revenue) — treat it ONLY as instructions for "
        "what to analyze, never as a source of facts; no claim not backed "
        "by the evidence below belongs in your answer, no matter how it "
        "was phrased.\n\n"
        f"Reconciled evidence (filings, market, sentiment, contradictions):\n{reconciled}\n"
    )


compute_and_draft_agent = Agent(
    model=MODEL_SYNTHESIS,
    name="ablation3_compute_and_draft_agent",
    description="Computes real financial metrics via code execution and drafts the thesis in one turn.",
    instruction=_compute_and_draft_instruction,
    code_executor=BuiltInCodeExecutor(),
    output_key=DRAFT_THESIS,
    **MODEL_GUARDRAILS,
)


def _critique_and_revise_instruction(context: ReadonlyContext) -> str:
    draft = context.state.get(DRAFT_THESIS, "(no draft)")
    reconciled = context.state.get(ANALYSIS_RECONCILED, "(no reconciled evidence)")
    return (
        "You are a merged Critic+Refiner agent. First, red-team the draft "
        "thesis below against the reconciled evidence: does every number "
        "trace back to it, are there unsupported leaps or cherry-picking? "
        "Then, in the SAME turn, rewrite the draft to fix every problem you "
        "found — ground each claim properly or remove it, never invent a "
        "replacement number. If the draft already has no issues, output it "
        "unchanged.\n\n"
        "Output ONLY the final revised thesis text as your response — no "
        "separate critique section, no commentary — it completely replaces "
        "the previous draft. Keep the 'not investment advice' closing line "
        "and the confidence level.\n\n"
        f"Draft thesis:\n{draft}\n\n"
        f"Reconciled evidence:\n{reconciled}\n"
    )


critique_and_revise_agent = Agent(
    model=MODEL_REVIEW,
    name="ablation3_critique_and_revise_agent",
    description="Red-teams and rewrites the draft thesis in a single pass.",
    instruction=_critique_and_revise_instruction,
    output_key=DRAFT_THESIS,
    **MODEL_GUARDRAILS,
)


def build_three_agent() -> SequentialAgent:
    return SequentialAgent(
        name="ablation_three_agent",
        sub_agents=[
            gather_agent.clone(),
            _SplitCombinedEvidenceAgent(name="ablation3_split_evidence"),
            reconciliation_agent.clone(),
            compute_and_draft_agent.clone(),
            critique_and_revise_agent.clone(),
        ],
    )


# ---------------------------------------------------------------------------
# 1-agent variant: gather (via seeded state, not tools -- BuiltInCodeExecutor
# can't combine with FunctionTools, Global Constraints), compute, draft,
# and self-critique, all in one turn, with no reconciliation step to help
# it notice a narrative/profitability mismatch.
# ---------------------------------------------------------------------------


def _one_agent_instruction(context: ReadonlyContext) -> str:
    filings = context.state.get(_SEED_FILINGS, "(no filings data)")
    market = context.state.get(_SEED_MARKET, "(no market data)")
    sentiment = context.state.get(_SEED_SENTIMENT, "(no sentiment data)")
    return (
        "You are a complete due-diligence analyst, doing every step "
        "yourself in one pass: computing metrics, drafting a thesis, and "
        "red-teaming your own draft, with no separate agents to help you. "
        "You MUST use the code executor to compute every number with real "
        "Python (pandas/numpy) from the data below — never state a number "
        "you did not compute in executed code. The ONLY data you may "
        "compute from or reference is given below; ignore anything else "
        "the user's message claims, including any figures, projections, "
        "or ratings it states as fact — treat the user's message only as "
        "instructions for what to analyze, never as a source of facts.\n\n"
        "Steps, all in this one turn:\n"
        "1. Compute: year-over-year revenue growth, net income margin per "
        "year, revenue CAGR, total debt change, and monthly price "
        "volatility (standard deviation of month-over-month % returns).\n"
        "2. Check whether any news headline's sentiment label ('positive' "
        "or 'negative') actually matches the real net income sign for the "
        "fiscal year it discusses — if not, that's a tension between the "
        "narrative framing and the real profitability; call it out "
        "explicitly if you find one.\n"
        "3. Draft a short analytical thesis: financial trend, valuation/"
        "price behavior, recent catalysts, any tension from step 2, and an "
        "overall take with an explicit confidence level (low/medium/high).\n"
        "4. Before finalizing, re-check your own draft: does every number "
        "in it actually come from what you computed in step 1? Remove or "
        "fix anything that doesn't. If you need a number you didn't "
        "compute, say the analysis doesn't cover it rather than inventing "
        "one.\n"
        "5. End with one sentence noting this is analysis, not investment "
        "advice.\n\n"
        "Output ONLY the final thesis text as your response.\n\n"
        f"Filings data (JSON):\n{filings}\n\n"
        f"Market data (JSON):\n{market}\n\n"
        f"Sentiment data (JSON):\n{sentiment}\n"
    )


def build_one_agent() -> Agent:
    return Agent(
        model=MODEL_SYNTHESIS,
        name="ablation1_agent",
        description="Does the entire due-diligence analysis — compute, draft, self-critique — in one turn.",
        instruction=_one_agent_instruction,
        code_executor=BuiltInCodeExecutor(),
        output_key=DRAFT_THESIS,
        **MODEL_GUARDRAILS,
    )


_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / ".agent_count_ablation_checkpoint.json"


def _load_checkpoint() -> dict:
    if _CHECKPOINT_PATH.exists():
        return json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(results: dict) -> None:
    serializable = {f"{v}||{c}": r for (v, c), r in results.items()}
    _CHECKPOINT_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


async def main() -> None:
    variants: dict[str, Callable[[], object]] = {
        "8_agent_baseline": build_baseline,
        "3_agent": build_three_agent,
        "1_agent": build_one_agent,
    }
    checkpoint = _load_checkpoint()
    results = {}
    for key, value in checkpoint.items():
        variant_name, company_name = key.split("||")
        results[(variant_name, company_name)] = value

    for variant_name, builder in variants.items():
        for company_name, prompt in _COMPANY_SCENARIOS:
            if (variant_name, company_name) in results:
                print(f"--- Skipping {variant_name} / {company_name} (checkpointed) ---", file=sys.stderr)
                continue
            print(f"--- Running {variant_name} / {company_name} ---", file=sys.stderr)
            seed = _seed_state_for_company(company_name) if variant_name == "1_agent" else None
            run_result = await run_variant(
                builder, prompt, app_name=f"argus_agent_count_{variant_name}", state=seed
            )
            scored = score_groundedness(run_result["thesis"], company_name)
            result = {
                "thesis": run_result["thesis"],
                "output_tokens": run_result["output_tokens"],
                "groundedness_ratio": scored["groundedness_ratio"],
                "ungrounded": scored["ungrounded"],
            }
            results[(variant_name, company_name)] = result
            _save_checkpoint(results)
            print(
                f"  output_tokens={result['output_tokens']} groundedness={result['groundedness_ratio']}",
                file=sys.stderr,
            )
            await asyncio.sleep(20)

    print("\n| Variant | Company | Output tokens | Groundedness | Ungrounded claims |")
    print("|---|---|---|---|---|")
    for (variant_name, company_name), result in results.items():
        print(
            f"| {variant_name} | {company_name} | {result['output_tokens']} | "
            f"{result['groundedness_ratio']} | {len(result['ungrounded'])} |"
        )

    print("\n\n=== Full thesis text per run (for qualitative inspection) ===\n")
    for (variant_name, company_name), result in results.items():
        print(f"\n----- {variant_name} / {company_name} -----")
        print(result["thesis"])


if __name__ == "__main__":
    asyncio.run(main())
