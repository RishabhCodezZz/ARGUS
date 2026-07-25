"""The session.state key namespace (spec SM1).

The original 9 keys were defined in full on day one. Parallel specialists
(Stage 1+) must each write to a UNIQUE key from this file — sharing a key
across parallel agents is a race condition (spec OR2).

RETRIEVAL_CTX was added in Stage 4 Part D — the spec's own agent table
names `retrieval_ctx` as the Retrieval Agent's output, so unlike the
ad-hoc bookkeeping keys (e.g. meta.tool_call_count, kept as local constants
in their own modules), this one belongs in the shared namespace.
"""

EVIDENCE_FILINGS = "evidence.filings"
EVIDENCE_MARKET = "evidence.market"
EVIDENCE_SENTIMENT = "evidence.sentiment"

ANALYSIS_QUANT = "analysis.quant"
ANALYSIS_RECONCILED = "analysis.reconciled"

DRAFT_THESIS = "draft.thesis"
REVIEW_CRITIQUE = "review.critique"

META_GROUNDEDNESS = "meta.groundedness"
META_HUMAN = "meta.human"

RETRIEVAL_CTX = "retrieval.context"
