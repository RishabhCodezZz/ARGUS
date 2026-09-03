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

## Multiple Gemini models, split by call frequency, not uniformly

Every agent ran on one pinned model through Stage 6. Free-tier quota is tracked per model, not
shared, so an agent that fires many times per run (the Orchestrator; the critic/refiner loop) needs
a large-quota pool, while an agent that fires exactly once per run (quant, synthesis, verifier,
retrieval) can safely get its own smaller, dedicated pool without starving anyone else's budget. The
split that shipped protects the two most load-bearing agents (Orchestrator, quant) on the one model
proven reliable across the whole project, rather than spreading everything evenly — after three
different models each passed an isolated single-call test and then failed on their first real
pipeline call, "confirmed valid" and "confirmed reliable under load" stopped being treated as the
same claim.

## An external code review is verified, not trusted or dismissed

Two rounds of review from a second reviewer given the code cold surfaced several real, previously
undetected bugs — including the sharpest one in the project's history: a date's day component in
un-normalized source text could validate a fabricated claim. Every finding was reproduced against the
real code before being acted on, and reproduced again after the fix, the same standard this project
already holds its own agents to. Not every recommendation was taken as given, either: a suggestion to
rename the "contradiction" field, or to fully solve deterministic retry execution in the same pass,
was scoped down to what could be verified safely rather than rushed — the review is a second set of
eyes, not a replacement for verifying its claims the same way every other claim in this project gets
verified.

## Durable memory and deterministic retry execution, deliberately deferred

Both were raised by external review as real gaps, and both are staying open on purpose rather than
being patched hastily. Durable (cross-restart) long-term memory would need a custom SQLite-backed
`MemoryService` — confirmed against ADK's own source that the only two non-in-memory implementations
it ships are both Vertex-AI-backed, which conflicts with this project's own "fully local, no cloud"
decision above. Deterministic retry execution — not just picking the retry name deterministically,
which is already fixed, but actually *performing* the bounded retry in code instead of trusting the
model to decide to call the pipeline again — needs a real controller driving that specific flow, not
a modified tool. Both are real architecture additions, not patches, and this project's own repeated
lesson (rushed fixes to state/memory-boundary issues have twice produced a *new*, harder-to-spot bug)
is reason enough to scope them as separate work rather than force them into the same pass as smaller,
independently-verifiable fixes.

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
