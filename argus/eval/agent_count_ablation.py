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
docs/superpowers/plans/2026-09-18-agent-count-budget-ablation-SPEC.md

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
