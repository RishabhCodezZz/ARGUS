# Agent-count ablation results — 2026-09-18

Raw data: [`agent_count_ablation_2026-09-18.json`](agent_count_ablation_2026-09-18.json) — the exact
output of `python -m argus.eval.agent_count_ablation` (see
[`argus/eval/agent_count_ablation.py`](../../argus/eval/agent_count_ablation.py)), committed verbatim.
Six live runs against the real Gemini API: 3 pipeline shapes (8-agent baseline, 3-agent, 1-agent) × 2
scenarios (Acme Corp clean, Globex contradiction) — the same two scenarios the existing component-level
[ablation study](ablation_2026-08-02.md) uses.

**What this measures, and why it's different from the existing ablation study.** That earlier study
answers "what does each *component* (critic loop, code execution) contribute?" by removing pieces from
the fixed 8-agent shape. This one answers a different question that arXiv 2604.02460 (2026,
budget-matched, Qwen3/DeepSeek-R1-Distill/Gemini 2.5) raises about any multi-agent system: single agents
often match or beat multi-agent teams at equal token budget, with heavy context degradation the real
exception. Nobody had checked whether ARGUS's 8-LLM-agent pipeline earns its keep against fewer, bigger
agents doing the same job. This does.

**Methodology note, stated plainly.** Every variant's `output_tokens` (real `candidates_token_count`
summed across every model call in the run) and `groundedness_ratio` are computed the **same way**, by
the harness itself, never by reading a variant's own self-report — only the 8-agent baseline even has a
Verifier step to produce a `meta.groundedness` score, so comparing that key across variants would be
comparing different things. Instead every thesis is scored by `score_groundedness`
(`argus/eval/agent_count_ablation.py`), which reuses `recompute_quant_metrics` +
`compute_grounding` — the same independent-re-derivation method `independent_groundedness` already
established in `argus/eval/custom_metrics.py` — against real mock data, never trusting any agent's
computed output.

## Summary

| Variant | Company | Output tokens | Groundedness | Ungrounded claims |
|---|---|---|---|---|
| 8-agent baseline | Acme Corp (clean) | 3872 | 0.9310 | 2 |
| 8-agent baseline | Globex (contradiction) | 4460 | 1.0000 | 0 |
| 3-agent | Acme Corp (clean) | 3326 | 0.9259 | 2 |
| 3-agent | Globex (contradiction) | 4261 | 0.9474 | 2 |
| 1-agent | Acme Corp (clean) | 2075 | 1.0000 | 0 |
| 1-agent | Globex (contradiction) | 1749 | 0.9394 | 2 |

**Average output tokens per run:** 8-agent 4166 · 3-agent 3794 (9% less) · 1-agent 1912 (**54% less**).

## What this shows

**Token cost drops sharply as agent count drops, with no increase in real fabrication once the
"ungrounded" flags are traced to their actual source.** The 1-agent variant used **under half the
baseline's average output tokens** (1912 vs. 4166) — per scenario, that's 53.6% of baseline on Acme
Corp (2075 vs. 3872) and 39.2% on Globex (1749 vs. 4460), so "less than half" holds in aggregate and on
Globex specifically, not on every individual scenario — and it scored the *best* groundedness of all
three variants on Acme Corp (1.0000, the only run of the six to reach a perfect score there) while doing
so. That is the finding I'd have bet against going in — the instinct that "fewer agents means the model
juggles more at once and starts slipping" — and it's the more interesting result precisely because it's
not what a naive prior would predict.

**Every "ungrounded" flag traces to one of two explained, non-fabrication causes — not a made-up number:**

1. **Markdown section-header numbers, in both the 8-agent baseline and the 3-agent variant.** Both
   Acme runs flag `[2.0, 3.0]` as ungrounded. Tracing it directly: the thesis text uses `#### 1.`,
   `#### 2.`, `#### 3.`, `#### 4.` section headers; `extract_numbers` (production
   `argus/tools/claim_matcher.py`, unmodified by this ablation) pulls the bare `1`/`2`/`3`/`4` out of
   those headers as claimed numbers. `1.0` and `4.0` happen to land within the fixed `±0.5` matching
   tolerance of real computed values (`1.49` and `4.15`/`4.70` respectively), so they ground by
   coincidence; `2.0` and `3.0` don't happen to land near any real value, so they flag. **This is a
   pre-existing characteristic of production `claim_matcher.py` that the 8-agent baseline hits too** —
   not something this ablation's new code introduced — and it only surfaces when a run's markdown
   happens to render `#### N.` rather than `**N.**` or another non-numeric heading style, a stylistic
   choice the model makes per run, not something any variant's code controls.
2. **Legitimate, arithmetically-correct derived percentages the ground-truth scorer doesn't cover, in
   both Globex runs that flag `[45.48, 72.22]` (3-agent and 1-agent).** Hand-checked: `(42.00 − 22.90) /
   42.00 = 45.48%` (the stock's real price decline) and `(1550.0 − 900.0) / 900.0 = 72.22%` (the real
   debt increase, as a percentage rather than the absolute-dollar figure `recompute_quant_metrics`
   computes). Both merged variants computed one extra derived percentage beyond what the 8-agent
   baseline's `quant_agent` narrates — not a fabrication, a **scope gap in this ablation's own
   scorer**, which only covers the specific metrics `recompute_quant_metrics` re-derives. Worth
   extending in a follow-up pass; not evidence any variant invented a number.

Net effect: treating both explained causes as non-findings, **all six runs are effectively 100%
grounded** — zero genuinely fabricated financial figures across any variant, including the cheapest one.

**A real, disclosed gap this ablation surfaced, unrelated to groundedness.** The 1-agent/Acme Corp
thesis ends with *"**Overall Take:** Buy/Hold for conservative compounding."* — a recommendation-style
label the existing `strip_action_language` guardrail (`argus/callbacks/guardrails.py`) does not catch:
its regex is `\byou should (buy|sell|short|invest)\b`, tuned to the specific phrasing the 8-agent
pipeline's `synthesis_agent`/`critic_agent`/`refiner_agent` instructions happen to produce, not to a
bulleted "Buy/Hold" label. The response still closes with the required disclaimer sentence, and this
variant's own instruction never asked for a recommendation — the model volunteered one anyway. This is
exactly the kind of gap this project's own guardrail regexes have needed fixing for before (see
`_MATCH_TOLERANCE`'s history in `DECISIONS.md`): a real hole, found by testing a genuinely different
phrasing distribution, not a flaw in the ablation's own new code. **Not fixed as part of this ablation**
— flagged here as a disclosed follow-up for `argus/callbacks/guardrails.py`, the same treatment this
project gives every other found-but-deferred gap.

## Reproducing this

```bash
PYTHONPATH="$(pwd)" .venv/Scripts/python.exe -m argus.eval.agent_count_ablation
```

Progress checkpoints to `.agent_count_ablation_checkpoint.json` at the repo root (git-ignored, since
it's a resumable working file — see this directory for a committed snapshot of a completed run), so a
rate-limit hit mid-run resumes instead of restarting.
