"""Unit tests for argus/eval/token_accounting.py -- pure Python, no ADK
Runner, no API key. Stand-in objects with a bare .usage_metadata
attribute are enough, the same "minimal fake, not a real ADK type"
approach test_orchestrator.py's _FakeToolContext already uses.
"""

from types import SimpleNamespace

from argus.eval.token_accounting import sum_output_tokens


def _event(candidates_token_count):
    if candidates_token_count is None:
        return SimpleNamespace(usage_metadata=None)
    return SimpleNamespace(usage_metadata=SimpleNamespace(candidates_token_count=candidates_token_count))


def test_sums_output_tokens_across_events():
    events = [_event(10), _event(25), _event(7)]
    assert sum_output_tokens(events) == 42


def test_skips_events_with_no_usage_metadata():
    events = [_event(10), _event(None), _event(5)]
    assert sum_output_tokens(events) == 15


def test_empty_events_sums_to_zero():
    assert sum_output_tokens([]) == 0
