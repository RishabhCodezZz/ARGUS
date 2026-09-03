"""Central place for model IDs, thresholds, and budgets.

Swap models here, not inside individual agent files.
"""

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
MODEL_FLASH = "gemini-3.5-flash-lite"

# Reserved for Synthesis/Critic once/if the project moves off the free tier.
# Keep equal to MODEL_FLASH until then so nothing silently starts costing money.
MODEL_PRO = MODEL_FLASH

# Multi-pool split, take 3 — every ID below was hit-and-trial verified with
# a real, isolated generate_content call BEFORE being wired into an agent
# (see the two prior reverted attempts: gemini-3.8-flash and gemini-3.7-flash
# both 503 under real load, gemini-3.0-flash / "gemini-3-flash" are both
# 404 NOT_FOUND, and this exact "gemini-2.5-flash-lite" spelling 404s too —
# none of those are used here). Deliberately conservative: orchestrator,
# quant, and all 3 gatherers — the entire "nothing else can run without
# this" backbone — stay on the one model proven reliable across this whole
# project, rather than diversifying everything at once again.
MODEL_REVIEW = "gemini-3.1-flash-lite"  # critic_agent, refiner_agent,
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
MODEL_SYNTHESIS = "gemini-3.5-flash"    # synthesis_agent — single call/run,
                                          # rank #4, dedicated pool.
MODEL_RETRIEVAL = "gemini-2.5-flash"    # retrieval_agent — lowest priority,
                                          # rarely invoked; safest place to
                                          # run a freshly-validated model.

# Stage 5 Part B: minimum meta.groundedness to auto-approve a run
# without a human review. Set against real observed scores, not a round
# number picked in the abstract: a clean run scores 1.00; Stage 3's
# adversarial run (one fabricated year, flagged [UNVERIFIED: 2026]) scored
# 0.9545. 0.98 auto-approves the former and escalates the latter, so both
# the auto-approve and human-review paths are reachable with real data.
GROUNDEDNESS_THRESHOLD = 0.98
