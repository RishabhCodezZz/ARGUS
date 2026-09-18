"""Sums real generated-output tokens off a run's ADK events -- the
"budget" axis for the agent-count ablation (argus/eval/agent_count_ablation.py).

Split into its own tiny module, imported by BOTH argus/eval/ablation.py
and argus/eval/agent_count_ablation.py, rather than living in either one:
the new ablation file needs to import run_variant/build_baseline FROM
ablation.py, so putting this helper in ablation.py itself and having the
new file import it from there would create a circular import the moment
ablation.py also needed it (which it does, for its own runs).

candidates_token_count (not prompt_token_count/total_token_count) is
generated-OUTPUT tokens specifically -- confirmed against
google.genai.types.GenerateContentResponseUsageMetadata -- matching the
"total output tokens" budget metric this ablation reports (agreed with
Rishabh over prompt tokens or raw call-count: quota here is a per-minute
REQUEST-rate limit, not a token-cost limit, so a call-count proxy would
conflate "how many round trips" with "how much was actually generated";
output tokens is the more direct budget analogue to the paper this
ablation is checking against).
"""

from typing import Iterable


def sum_output_tokens(events: Iterable[object]) -> int:
    """Sums candidates_token_count across every event that carries
    usage_metadata; events without it (e.g. events with no model call
    attached) contribute zero, not an error."""
    total = 0
    for event in events:
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            continue
        count = getattr(usage, "candidates_token_count", None)
        if count is not None:
            total += count
    return total
