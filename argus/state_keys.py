"""The session.state key namespace.

The original 9 keys were defined in full on day one. Parallel specialists
(Stage 1+) must each write to a UNIQUE key from this file — sharing a key
across parallel agents is a race condition.

RETRIEVAL_CTX was added in Stage 4 Part D — it's a key the Retrieval
Agent's output is expected to use, so unlike the ad-hoc bookkeeping keys
(e.g. meta.tool_call_count, kept as local constants in their own modules),
this one belongs in the shared namespace.
"""

EVIDENCE_FILINGS = "evidence.filings"
EVIDENCE_MARKET = "evidence.market"
EVIDENCE_SENTIMENT = "evidence.sentiment"

# Per-request freshness stamps (Stage 5 follow-up, external code review).
# The Orchestrator's before_agent_callback writes META_REQUEST_ID once per
# user turn; each gatherer's after_agent_callback copies that value next to
# its own evidence key. check_gather_status then reports gather_ok only when
# all three stamps match the CURRENT request -- so evidence left in state by
# an earlier turn can't be mistaken for a fresh gather if the model skips
# full_analysis_pipeline on a later broad request. Parallel gatherers each
# write a UNIQUE stamp key, same rule as the evidence keys themselves.
META_REQUEST_ID = "meta.request_id"
EVIDENCE_FILINGS_RUN_ID = "evidence.filings_run_id"
EVIDENCE_MARKET_RUN_ID = "evidence.market_run_id"
EVIDENCE_SENTIMENT_RUN_ID = "evidence.sentiment_run_id"

ANALYSIS_QUANT = "analysis.quant"
ANALYSIS_RECONCILED = "analysis.reconciled"

DRAFT_THESIS = "draft.thesis"
REVIEW_CRITIQUE = "review.critique"

# Naming precision worth being explicit about (an external code review
# raised this): this score measures NUMERIC groundedness specifically --
# every number the draft states, checked against the source data. A
# prose-only fabricated claim with no numbers in it (e.g. an invented
# qualitative assertion) contains nothing for check_grounding to check,
# and scores a vacuous 1.0 by design (see compute_grounding's own
# docstring in argus/tools/claim_matcher.py) -- that's a real, disclosed
# scope limit, not a bug: this project's groundedness check was built to
# defend against fabricated NUMBERS specifically (its whole Stage 3
# motivation), not as a general-purpose factual/citation verifier. Kept
# as "meta.groundedness" (not renamed to "meta.numeric_groundedness")
# since it's referenced across config.py's threshold, hitl.py's gate,
# memo.py's output, and the README -- the scope is now documented
# instead of renamed.
META_GROUNDEDNESS = "meta.groundedness"
META_HUMAN = "meta.human"

RETRIEVAL_CTX = "retrieval.context"
