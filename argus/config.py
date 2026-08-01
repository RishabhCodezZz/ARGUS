"""Central place for model IDs, thresholds, and budgets (spec DEP5).

Swap models here, not inside individual agent files.
"""

# Gemini API left the free tier for Pro models on 2026-04-01 (verified 2026-07).
# `gemini-3.1-pro` referenced in earlier planning docs is not a stable model ID.
#
# TEMPORARY: pinned to Flash-Lite, not the usual "-latest" Flash alias.
# gemini-flash-latest currently resolves to gemini-3.6-flash, which GA'd only
# days ago and free-tier caps at 20 requests/day — we hit that limit during
# Stage 1 dev (a single fan-out turn burns several raw calls). Flash-Lite's
# free daily quota is far higher, which is what active development needs.
# User instruction (2026-07-24): switch back to "gemini-flash-latest" once
# the full pipeline is built and iteration slows down.
MODEL_FLASH = "gemini-3.5-flash-lite"

# Reserved for Synthesis/Critic once/if the project moves off the free tier.
# Keep equal to MODEL_FLASH until then so nothing silently starts costing money.
MODEL_PRO = MODEL_FLASH

# Stage 5 Part B (HITL1): minimum meta.groundedness to auto-approve a run
# without a human review. Set against real observed scores, not a round
# number picked in the abstract: a clean run scores 1.00; Stage 3's
# adversarial run (one fabricated year, flagged [UNVERIFIED: 2026]) scored
# 0.9545. 0.98 auto-approves the former and escalates the latter, so both
# the auto-approve and human-review paths are reachable with real data.
GROUNDEDNESS_THRESHOLD = 0.98
