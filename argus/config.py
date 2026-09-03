"""Central place for model IDs, thresholds, and budgets.

Swap models here, not inside individual agent files.

Each MODEL_* below is a configured ``Gemini`` instance, not a bare string,
so every model call carries retry-with-backoff (see ``_RETRY`` and the note
above it). Agents still just write ``model=MODEL_FLASH`` — ADK accepts a
``BaseLlm`` instance anywhere it accepts a model-id string.
"""

from google.adk.models import Gemini
from google.genai import types

# --- model IDs: the "which free-tier pool" decision -------------------------
#
# Gemini API left the free tier for Pro models on 2026-04-01 (verified 2026-07).
#
# Pinned to Flash-Lite, not the usual "-latest" Flash alias, and this is a
# standing decision rather than a temporary one — it was deliberately
# revisited after Stage 6 (the eval harness and ablation study, which run
# mostly single-turn and non-interactive) and kept: even that work hit the
# free tier's 15-requests/minute ceiling on Flash-Lite more than once, and
# "gemini-flash-latest"'s 20-requests/*day* cap would make that meaningfully
# worse, not better. Worth re-checking again before any future stage that
# needs heavy live interactive testing.
_FLASH_ID = "gemini-3.5-flash-lite"

# Multi-pool split, take 3 — every ID below was hit-and-trial verified with
# a real, isolated generate_content call BEFORE being wired into an agent
# (see the two prior reverted attempts: gemini-3.8-flash and gemini-3.7-flash
# both 503 under real load, gemini-3.0-flash / "gemini-3-flash" are both
# 404 NOT_FOUND, and this exact "gemini-2.5-flash-lite" spelling 404s too —
# none of those are used here). Deliberately conservative: orchestrator,
# quant, and all 3 gatherers — the entire "nothing else can run without
# this" backbone — stay on the one model proven reliable across this whole
# project, rather than diversifying everything at once again.
_REVIEW_ID = "gemini-3.1-flash-lite"    # critic_agent, refiner_agent,
                                        # verifier_agent — high-frequency
                                        # loop (up to 4 calls each) needs
                                        # the large 500-req/day pool;
                                        # verifier only adds 1 call/run,
                                        # barely dents it. gemini-3.6-flash
                                        # (verifier's first assignment) is
                                        # dropped entirely: it 503'd on its
                                        # very first real pipeline call —
                                        # third full-Flash model in a row
                                        # (3.8, 3.7, 3.6) to fail under
                                        # real load despite passing an
                                        # isolated test call. Full-Flash
                                        # tier is no longer trusted on the
                                        # critical path; Lite-tier only.
_SYNTHESIS_ID = "gemini-3.5-flash"      # synthesis_agent — single call/run,
                                        # rank #4, dedicated pool.
_RETRIEVAL_ID = "gemini-2.5-flash"     # retrieval_agent — lowest priority,
                                        # rarely invoked; safest place to
                                        # run a freshly-validated model.

# --- transient-failure retry ----------------------------------------------
#
# A transient 5xx from an overloaded model — most often a 503 "This model is
# currently experiencing high demand" on ONE of the three parallel gatherers
# — used to abort the entire run: ADK's ParallelAgent runs its sub-agents in
# an asyncio.TaskGroup, so the first sub-agent's ServerError cancels its
# siblings and propagates as an ExceptionGroup all the way to the root, which
# the Dev UI shows only as "an error occurred". google-genai does NOT retry
# unless retry_options is set explicitly (a None default reaches the client
# as "no retry"). This gives every model call up to 5 attempts with
# exponential backoff (~2s, 4s, 8s, 16s, capped at 30s) over google-genai's
# default retryable set (408, 429, and 5xx), so a brief demand spike
# self-heals instead of killing the pipeline. It does not paper over a real,
# sustained quota exhaustion (RESOURCE_EXHAUSTED that persists past ~30s of
# backoff still surfaces) — see the quota notes in CLAUDE.md.
_RETRY = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2.0,
    max_delay=30.0,
    exp_base=2.0,
)


def _model(model_id: str) -> Gemini:
    return Gemini(model=model_id, retry_options=_RETRY)


MODEL_FLASH = _model(_FLASH_ID)

# Reserved for Synthesis/Critic once/if the project moves off the free tier.
# Keep equal to MODEL_FLASH until then so nothing silently starts costing money.
MODEL_PRO = MODEL_FLASH

MODEL_REVIEW = _model(_REVIEW_ID)
MODEL_SYNTHESIS = _model(_SYNTHESIS_ID)
MODEL_RETRIEVAL = _model(_RETRIEVAL_ID)

# Stage 5 Part B: minimum meta.groundedness to auto-approve a run
# without a human review. Set against real observed scores, not a round
# number picked in the abstract: a clean run scores 1.00; Stage 3's
# adversarial run (one fabricated year, flagged [UNVERIFIED: 2026]) scored
# 0.9545. 0.98 auto-approves the former and escalates the latter, so both
# the auto-approve and human-review paths are reachable with real data.
GROUNDEDNESS_THRESHOLD = 0.98
