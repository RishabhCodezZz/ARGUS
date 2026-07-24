"""The session.state key namespace (spec SM1).

Defined once, in full, on day one. Parallel specialists (Stage 1+) must each
write to a UNIQUE key from this file — sharing a key across parallel agents is
a race condition (spec OR2). Stage 0 does not use any of these yet.
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
