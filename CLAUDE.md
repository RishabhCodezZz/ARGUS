# CLAUDE.md — ARGUS project context

> This file is auto-loaded by Claude Code each session. It's the "what's going on" briefing.
> Full technical spec: **`ARGUS_ADK_Requirements.md`**. Human build guide: **`PLAN.md`**.
> Both exist locally but are **git-ignored** (user's call, 2026-07-24) — working docs, not needed by
> others viewing the public repo. This file (`CLAUDE.md`) is the only planning doc that stays public.

## What this project is

**ARGUS** (Autonomous Reasoning & Grounded Understanding System) is a hierarchical **multi-agent
system** built on **Google's Agent Development Kit (ADK, Python)**. It takes a high-level analytical
question (reference domain: company/financial due-diligence), autonomously plans an investigation,
fans out specialist agents in parallel, runs **real code for all math** (not LLM guessing),
**red-teams its own draft**, verifies every claim is **grounded in a source**, and emits a sourced
insight memo. RAG exists only as one optional subordinate tool, not the whole strategy.

## Who I'm working with

The user is a **beginner** doing this as a learning + portfolio project. So:
- Explain the *what* and *why* in plain language; don't assume prior agent/cloud/git knowledge.
- Build **one new agent/idea at a time**; never dump all 14 agents at once.
- After each change, the system must still run. Prefer small, demoable, verifiable steps.

## Decisions locked in (these shape everything)

- **Goal:** Learn agent-building **and** produce a portfolio/interview piece → the eval harness,
  ablation study, and a cloud deploy stay in scope (spec calls them "resume gold").
- **Scope:** **Simplified-first** — build a Minimum Viable ARGUS (~4 agents) end-to-end, then grow.
- **Data/infra:** **Mock/sample data + a free Gemini API key, fully local.** No Google Cloud, no real
  financial APIs until the final stages. This decouples learning agents from wrangling data APIs.

## Tech stack

- **Framework:** Google ADK (Python), installed version **2.5.0**. Install: `pip install google-adk`.
  Python 3.10+ required; project venv runs **3.12.10**.
- **Models:** all agents use `MODEL_FLASH` from `argus/config.py` — **not** `gemini-3.1-pro`. That ID
  never shipped as a stable model, and as of 2026-04-01 Pro-tier models left the Gemini free tier
  entirely, so a Pro model would break the locked-in "free, fully local" decision. `MODEL_PRO` exists
  in `config.py` as an alias to `MODEL_FLASH` for now — point it at a real Pro ID only if/when the
  project accepts billing.
  **TEMPORARY (as of Stage 1):** `MODEL_FLASH` is pinned to `gemini-3.5-flash-lite`, not the usual
  `gemini-flash-latest` alias — that alias resolves to `gemini-3.6-flash`, whose free tier caps at
  20 requests/day (it GA'd days before this project started) and we hit that limit during Stage 1 dev.
  Flash-Lite's free daily quota is far higher. **User instruction: switch back to `gemini-flash-latest`
  once the full pipeline is built and iteration slows down** — see `config.py` for the full note.
- **Local run:** `adk web` → http://localhost:8000 (visual trace/debug UI). CLI: `adk create`, `adk eval`.
- **Auth (local):** `.env` with `GOOGLE_GENAI_USE_VERTEXAI=FALSE` and `GOOGLE_API_KEY=…` (from Google
  AI Studio). **`.env` is git-ignored — never commit the key.**

## Non-negotiable principles

1. **Numbers come from code, words come from the LLM.** The Quant agent runs real pandas/numpy via
   ADK's code executor; other agents may only *narrate* computed values, never invent them.
2. **Parallel agents each write a UNIQUE `session.state` key** (see `argus/state_keys.py`) — prevents
   race conditions.
3. **Keep the pipeline runnable after every change.** One agent/idea at a time. Commit after each green
   stage — **except git itself is deliberately not initialized yet** (user's call): no `git init` and
   nothing pushed to GitHub until the Stage 1 Minimum Viable ARGUS (~4 agents) is complete.
4. **Every tool has a clear docstring** — the LLM reads it to decide when to call the tool.
5. **Guardrails live in ADK callbacks, not in prompt-begging.**
6. **Verify current ADK API names** (adk.dev / github.com/google/adk-python) before using a class —
   versions drift; don't trust memory for exact signatures (code executor, planner, memory/artifact services).
7. **Any agent reading a prior agent's state must use a callable `instruction` (an `InstructionProvider`
   function receiving `ReadonlyContext`, indexing `context.state[...]` directly) — never ADK's `{var}`
   string-templating.** `{var}` only matches valid Python identifiers, so it silently fails to substitute
   our dotted keys like `evidence.filings` (leaves the literal text in the prompt, no error). See
   `argus/agents/quant.py` for the pattern. Discovered and verified against ADK source in Stage 1.
8. **`output_key` persists an agent's final LLM-generated text, not a tool's raw return value.** If a
   gatherer's job is to relay tool data forward, its instruction must force verbatim output (e.g. "reply
   with ONLY the raw JSON, no prose") — otherwise a paraphrase risks a silent transcription error before
   the Quant agent ever sees the number, which breaks principle #1. See `argus/agents/gather.py`.
9. **`BuiltInCodeExecutor` can only be used alone on an agent** — it cannot be combined with a
   `FunctionTool` on the same agent (confirmed against ADK docs). An agent needing both computed state
   and code execution reads the state via its `instruction` callable (principle 7), not a tool call.
10. **`LoopAgent` early-exit is tool-based, not a return value.** A sub-agent stops the loop by calling a
    `FunctionTool(tool_context: ToolContext)` that sets `tool_context.actions.escalate = True` — the
    loop checks `event.actions.escalate` after each sub-agent runs (confirmed against ADK source).
    `max_iterations` is what actually guarantees termination; escalate is the fast path, not the safety
    net — never build a loop without both. Only give the exit tool to the agent that makes the actual
    pass/fail judgment (here: the Critic, not the Refiner — see OR3), or the loop can exit before the
    judgment-maker ever runs. Don't set `skip_summarization=True` on that tool if you need the model's
    follow-up text in `output_key` — it skips the follow-up turn entirely, so the state ends up holding
    the tool's raw return value instead (observed directly in Stage 2 dev: `review.critique` came back
    as `{}` until this was fixed). Even with correct wiring, expect the model to sometimes state its
    verdict in text without actually calling the tool that iteration (seen on Flash-Lite) — harmless
    since `max_iterations` bounds it, but worth an explicit "these are one action, always do both"
    instruction to reduce how often it happens. **`LoopAgent` (like Sequential/Parallel) is marked
    deprecated in ADK 2.5.0 in favor of the graph Workflow API** — still fully functional, consistent
    with the templated-workflow decision already made for Stages 0-3; see `argus/agents/critic.py`.

## Project structure (built up over the stages)

```
Argus/
  ARGUS_ADK_Requirements.md   # the spec
  CLAUDE.md                   # this file
  PLAN.md                     # human-readable step-by-step plan
  README.md                   # progress log
  .venv/  .gitignore  .env    # env (.venv/.env git-ignored, no git repo yet)
  argus/                      # ADK agent app
    __init__.py  agent.py     # agent.py defines root_agent (entrypoint)
    config.py                 # model IDs, thresholds, budgets (one place)
    state_keys.py             # session.state key namespace (constants)
    agents/
      gather.py                 # filings_agent, market_agent, sentiment_agent + evidence_gather (ParallelAgent)
      quant.py                  # quant_agent (BuiltInCodeExecutor)
      synthesis.py               # synthesis_agent
      critic.py                 # critic_agent + exit_loop tool
      refiner.py                 # refiner_agent
    tools/
      mock_data.py              # get_filings, get_market_data, get_news, get_company_snapshot
    callbacks/                 # not yet built (Stage 3)
  data/
    acme_corp/  globex/       # filings.json, prices.json, news.json each
  eval/  tests/               # not yet built (Stage 6, unit tests as we go)
```

## State-key namespace (spec SM1 — defined once, in `state_keys.py`)

`evidence.filings`, `evidence.market`, `evidence.sentiment`, `analysis.quant`,
`analysis.reconciled`, `draft.thesis`, `review.critique`, `meta.groundedness`, `meta.human`.

## Staged roadmap (detail in PLAN.md)

- **Stage 0** — one `LlmAgent` + one `FunctionTool` running in `adk web`.
- **Stage 1** — Minimum Viable ARGUS on mock data: 3 parallel gatherers → code-execution Quant → Synthesis.
- **Stage 2** — self-critique `LoopAgent` (Critic + Refiner, early-exit on PASS via `escalate=True`).
- **Stage 3** — Groundedness Verifier + guardrail callbacks (PII/citation/budget/injection).
- **Stage 4** — Reconciliation (custom `BaseAgent`) + dynamic Orchestrator + bounded re-planning + RAG-as-tool.
- **Stage 5** — long-term memory, human-in-the-loop gate, memo artifacts, streaming.
- **Stage 6** — evaluation harness + ablation table + CI (portfolio centerpiece).
- **Stage 7** — Cloud Run / Vertex deploy + tracing (do last; everything works locally first).

## Current status

**Stage 2 done and verified** (2026-07-24) — the self-critique loop is live. `critic_agent` red-teams
`draft.thesis` against the actual evidence/computed metrics (not just internal coherence) and either
calls `exit_loop` + replies "PASS", or writes a structured critique to `review.critique` without exiting;
`refiner_agent` rewrites `draft.thesis` to address it, grounding or removing flagged claims, never
patching with an invented number. Wrapped as `LoopAgent(sub_agents=[critic_agent, refiner_agent],
max_iterations=4)`, inserted into `root_agent`'s `SequentialAgent` after Synthesis. Verified live across
3 runs: an instant single-iteration PASS, a run where the critic stated PASS in text but skipped the
`exit_loop` call (so the loop correctly ran a redundant-but-harmless extra iteration before exiting —
`max_iterations` is what actually guarantees this can't run away, not escalate), and a clean run after
an instruction tweak bound the two actions together. See non-negotiable principle 10 for the full
mechanism and gotchas.

Stage 1 recap: Minimum Viable ARGUS (parallel gather → code-execution Quant → Synthesis) — full detail
in `README.md`'s progress log. `data/{acme_corp,globex}/{filings,prices,news}.json` holds 5 years of
financials, a price series, and headlines each (Globex's 2025-01-22 headline is a deliberately
misleading claim, seeded for the Stage 3/4 verifier/reconciler to catch later — not caught yet, expected).

Git is now set up: local repo, pushed to **https://github.com/RishabhCodezZz/ARGUS** (public). Working
docs (`ARGUS_ADK_Requirements.md`, `PLAN.md`, `.claude/`) are git-ignored — local only, user's call
2026-07-24 — `CLAUDE.md` is the only planning doc that stays in the public repo.

Two things still to circle back on: (1) `MODEL_FLASH` is temporarily `gemini-3.5-flash-lite` — switch
back to `gemini-flash-latest` once iteration slows down (see Models note above, `config.py`). (2) the
Dev UI trace shows recurring "Performance" warning icons on most events — no observed functional impact,
worth a look if something actually breaks.

Next: **Stage 3** — Groundedness Verifier (claim↔evidence matcher tool) + guardrail callbacks
(`before_model_callback`/`after_model_callback`/`before_tool_callback`/`after_tool_callback` for
PII/citation/budget/prompt-injection). This is also where Globex's planted misleading headline should
finally get caught — good end-to-end test of whether the verifier actually works.

## How to help me

- Go stage by stage; finish and verify one before starting the next.
- After each stage: run in `adk web`, do a numeric hand-check + a negative test, then commit (git is set
  up now — see Current status). Still confirm before pushing anything, and before any `.gitignore` /
  what's-public decisions, per how the last two were handled.
- Flag whenever we're about to hit a known rabbit hole (real-data APIs early, cloud early, non-converging loop).
- **Verify framework facts before trusting old docs/memory** — ADK and Gemini model IDs both drifted
  from what `ARGUS_ADK_Requirements.md`/`PLAN.md` originally assumed (ADK v1.25+ → actually 2.5.0;
  `gemini-3.1-pro` never existed as stable and Pro left the free tier 2026-04-01). Re-check adk.dev /
  ai.google.dev before pinning any new class name or model ID in a later stage. Stages 1-2 found several
  more API gotchas the hard way — see non-negotiable principles 7-10 — so keep verifying rather than
  assuming, and keep verifying empirically (Dev UI trace/state) when docs themselves are ambiguous.
- **Watch the free-tier quota.** Fan-out architectures burn several raw API calls per user turn — Stage
  2 added up to ~6 more per run. If a run throws `ExceptionGroup` / `RESOURCE_EXHAUSTED` in the server
  logs, that's the quota, not a bug — check `preview_logs` before debugging application code.
