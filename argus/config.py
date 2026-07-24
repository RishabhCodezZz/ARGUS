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
