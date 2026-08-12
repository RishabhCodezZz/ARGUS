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

# Stage 5 Part B: minimum meta.groundedness to auto-approve a run
# without a human review. Set against real observed scores, not a round
# number picked in the abstract: a clean run scores 1.00; Stage 3's
# adversarial run (one fabricated year, flagged [UNVERIFIED: 2026]) scored
# 0.9545. 0.98 auto-approves the former and escalates the latter, so both
# the auto-approve and human-review paths are reachable with real data.
GROUNDEDNESS_THRESHOLD = 0.98
