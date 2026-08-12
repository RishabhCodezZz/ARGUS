# Design decisions

Why ARGUS is built the way it is. Each one is a real trade-off made during development, not a
default. See [DEVLOG.md](DEVLOG.md) for the full build history these were made across.

## Mock data instead of real financial APIs

Real filings/market/news APIs mean auth, rate limits, schema drift, and cost — none of which teach
you anything about building a multi-agent system. Using hand-authored mock data for two companies
deliberately decouples "learn to build agents" from "wrangle a third-party data API," so every hour
spent goes into the architecture, not plumbing. The mock data is also deliberately *not* trivial: one
company (Globex) has a planted contradiction — a headline calling 2024 "the strongest year in company
history" against a real filed net loss for that year — specifically so the Reconciliation Agent has
something real to catch, not a toy case.

## The Reconciliation Agent makes zero LLM calls

It's a custom `BaseAgent`, not an `LlmAgent` — its entire job (does this headline's tone match what
the company actually filed for that period) is pure Python. An LLM asked to judge whether another
LLM's narration is spun is the wrong tool for that job: it's the same kind of pattern-matching that
produced the spin in the first place, just checking its own work. Contradiction detection here is
closer to a lookup than a judgment call, so it's implemented as one.

## Evaluation metrics are deterministic, never LLM-as-judge

All four custom `adk eval` metrics — tool trajectory, numeric fidelity, groundedness, contradiction
correctness — are pure Python re-derivations from the same mock data the agents themselves read, not
another model grading the output. An eval whose scorer is itself an LLM is trusting the system under
test to grade itself, which defeats the point of measuring it. This is the project's core rule
("numbers come from code, not the model") applied to the harness that checks the project, not just
the project itself.

## A fixed matching tolerance, not a percentage

The groundedness checker allows small rounding slack when matching a stated number against source
data (e.g. "$812 million" for a real value of 812.4). The first version scaled that slack by a
percentage of the value — reasonable-sounding, until you realize 1% of the year 2025 is about 20. A
fabricated future year ("2026") then matches a real one ("2025") and passes as grounded. Found via a
live adversarial test that injected exactly that kind of fake claim. Fixed with a small **fixed**
tolerance instead — every real rounding difference observed in actual narration is well under 1,
so a fixed tolerance covers legitimate slack without opening that hole back up.

## Control-flow decisions read state directly, never an agent's prose

When something needs to route or retry, the check is a deterministic read of `session.state` (e.g.
"did the gather step report an error"), never an LLM's own interpretation of another agent's text
output. The reason: a total data-gathering failure still produces a well-formed "Analytical
Thesis... Confidence Level: Low" from downstream agents — which reads as a complete, valid answer to
another LLM, not an unambiguous failure signal, even though a human skimming it would recognize it
immediately. Found live: an early version of the retry logic relied on this kind of inference and
silently never retried. The fix applies the same "decisions come from code, not vibes" principle to
routing that the rest of the project already applies to arithmetic.

## Three deliberate departures from a stricter spec-agent design

1. **No formal planner module** (e.g. a `PlanReActPlanner`). That mainly adds visible step-by-step
   planning text to the trace, not different routing capability — a well-instructed orchestrator
   agent choosing among tools already covers the actual requirement.
2. **The human-in-the-loop gate is three tools on the Orchestrator, not a separate agent.** Framework
   behavior forced this: a sub-agent wrapped as a tool has its long-running "pause and wait for a
   human" signal silently absorbed at that boundary and never forwarded to its caller. A gate that
   lives inside a wrapped sub-pipeline would be a safety gate that silently doesn't gate.
3. **The memo generator is a deterministic tool, not another LLM agent.** By the time it runs, the
   thesis has already been drafted, red-teamed, verified, and gate-cleared, and the system's own rule
   already forbids paraphrasing it when presenting it. Regenerating it through one more model call
   would reintroduce exactly the risk that whole chain exists to remove — for an API call this
   project's free-tier budget doesn't have to spare.

Each of these is "built it differently than the more literal design, here's why" rather than a gap
nobody noticed.

## Why Google ADK

It's the framework this project set out to learn, and it makes a few structural guarantees explicit
rather than implicit: typed shared state (`session.state`) instead of passing strings between agents,
first-class parallel/sequential/loop composition, and lifecycle callbacks as the place guardrails
live — not prompt instructions asking the model nicely. None of that is ADK-exclusive, though: the
same primitives exist under different names in CrewAI, AutoGen, and LangGraph — an orchestrator
choosing among tools, a bounded self-critique loop, shared typed state, and hook-based safety
enforcement translate directly. What's genuinely framework-specific in this codebase is where a
particular abstraction boundary silently drops something (documented at length in DEVLOG.md) — not
the overall shape of the design.
