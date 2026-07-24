# ARGUS

Autonomous Reasoning & Grounded Understanding System — a hierarchical multi-agent due-diligence
system built on Google's Agent Development Kit (ADK). Give it a company name and it:

1. **Gathers** filings, market, and sentiment data in parallel (`argus/agents/gather.py`) — three
   specialist agents, each writing to its own session-state key.
2. **Computes** real financial metrics — YoY growth, margins, CAGR, debt trend, price volatility —
   via executed Python (pandas/numpy), never LLM-estimated (`argus/agents/quant.py`).
3. **Drafts** an analytical thesis citing only those computed numbers (`argus/agents/synthesis.py`).
4. **Red-teams itself**: a Critic checks the draft against the evidence for unsupported claims or
   cherry-picking; a Refiner fixes what it finds. Loops until it passes or 4 iterations run out
   (`argus/agents/critic.py`, `refiner.py`).
5. **Verifies groundedness**: every numeric claim in the final draft is checked against the evidence
   by a deterministic matcher (not an LLM's opinion) — ungrounded claims get flagged in place, and a
   `meta.groundedness` score is recorded (`argus/agents/verifier.py`, `argus/tools/claim_matcher.py`).

A **Reconciliation Agent** (`argus/agents/reconciliation.py`) runs between steps 2 and 3 — a custom
`BaseAgent` with zero LLM calls, pure Python, cross-checking each headline's sentiment against the real
fiscal-year performance it discusses and flagging any contradiction before Synthesis ever drafts a word.

Four guardrail callbacks (`argus/callbacks/guardrails.py`) run on every model/tool call across the
whole pipeline: blocking direct trade-advice requests, stripping unhedged action language from any
agent's output, enforcing a per-session tool-call budget, and flagging instruction-like text inside
tool results (prompt-injection defense).

Runs entirely on the Gemini API free tier, no cloud project required.

## Running it

```bash
.venv/Scripts/adk.exe web --port 8000
```

Open http://localhost:8000, pick the `argus` app, and chat with it.

Run the test suite (pure Python, no API calls, no cost):

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

## Setup

1. `python -m venv .venv`
2. `.venv/Scripts/python.exe -m pip install google-adk`
3. Get a free key at [aistudio.google.com](https://aistudio.google.com) → "Get API key"
4. Create `argus/.env`:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=your-key-here
   ```

## Progress log

- **Stage 0 (2026-07-24)** — One `Agent` (`gemini-flash-latest`) with one tool,
  `get_company_snapshot`, backed by hardcoded mock data for two companies (Acme Corp, Globex).
  Verified in the Dev UI: a company-name prompt triggers a real tool call and the answer matches the
  mock data exactly; an unrelated prompt triggers no tool call.
- **Stage 1 (2026-07-24)** — Minimum Viable ARGUS: `filings_agent`, `market_agent`, `sentiment_agent`
  run in parallel (unique `output_key` per spec OR2) over 5 years of mock financials, a monthly price
  series, and news headlines → `quant_agent` computes YoY growth, margins, CAGR, debt trend, and price
  volatility via real executed pandas code (`BuiltInCodeExecutor`, never LLM-estimated) →
  `synthesis_agent` drafts a thesis citing only those computed numbers. Composed as a `SequentialAgent`.
  Verified in the Dev UI: full run on Globex produced a thesis whose every figure matched a hand-check
  of the mock data (CAGR 7.52%, net margin -4.02%, debt +$650M, volatility 7.37%); a negative test on a
  nonexistent company degraded gracefully with no crash and no invented numbers.
  Running temporarily on `gemini-3.5-flash-lite` (higher free-tier quota) instead of the usual
  `gemini-flash-latest` — see `CLAUDE.md` for why, and switch back once iteration slows down.
- **Stage 2 (2026-07-24)** — Self-critique loop: `critic_agent` red-teams `draft.thesis` against the
  actual evidence and computed metrics, either passing it (calls `exit_loop`, a `FunctionTool` that
  sets `tool_context.actions.escalate = True`) or writing a structured critique for `refiner_agent` to
  address. Wrapped as `LoopAgent(max_iterations=4)`, inserted after Synthesis. Verified in the Dev UI
  across 3 runs, including one where the critic stated PASS in text without calling the exit tool that
  iteration — the loop correctly ran one extra harmless pass before exiting, confirming `max_iterations`
  (not escalate) is the actual non-negotiable safety net against a non-converging loop.
- **Stage 3 (2026-07-24)** — Groundedness Verifier + 4 guardrail callbacks. `verifier_agent` extracts
  every numeric claim from the final draft and checks each against the evidence via a deterministic
  matcher tool (`check_grounding`), writing `meta.groundedness` and flagging (not deleting) anything
  ungrounded as `[UNVERIFIED: ...]`. `argus/callbacks/guardrails.py` adds `before_model_callback`
  (blocks direct trade-advice requests), `after_model_callback` (strips unhedged action language),
  `before_tool_callback` (per-session tool-call budget), and `after_tool_callback` (provenance logging
  + injected-instruction screening), wired onto all 8 agents.

  Live adversarial testing (a prompt injecting a fake "2026 projected revenue of $5 billion") caught
  two real bugs before they shipped: the Quant agent was echoing the injected claim back into its own
  "deterministic" output as a hedged footnote, and a magnitude-scaled matching tolerance let the
  fabricated "2026" pass as "close enough" to a real "2025". Both fixed and re-verified live — the
  Quant/Synthesis agents now refuse to acknowledge injected claims at all, and the Verifier still
  catches the residual mention (`meta.groundedness = 0.9545`, `[UNVERIFIED: 2026]` visibly flagged).
  18 pytest tests total, including a regression test for the tolerance bug.
- **Stage 4, Part A (2026-07-24)** — Reconciliation Agent. `ReconciliationAgent` is a custom `BaseAgent`
  (ADK's first non-`LlmAgent` primitive in this project) — its entire body is pure Python
  (`argus/tools/reconciler.py`), no LLM call at all. It cross-checks each news headline's sentiment
  label against the real net income for whatever fiscal year the headline names, and writes the full
  evidence bundle plus any contradictions to `analysis.reconciled`, which `synthesis_agent` now reads
  instead of the three raw evidence keys separately.

  This finally landed the payoff seeded back in Stage 1: Globex's 2025-01-22 headline ("CEO touts
  'strongest year in company history'") was deliberately planted alongside a real FY2024 net loss of
  $57M. Verified both via `pytest` directly against the real mock data and live in the full 9-agent
  pipeline — the agent correctly finds the 1 contradiction, and the resulting thesis now explicitly
  states it: *"a notable reporting contradiction occurred: a headline... framed the company's fiscal
  performance as a 'positive' record revenue year... whereas actual filed figures for FY2024 revealed a
  net loss."* Acme Corp (the clean company) produces zero contradictions. 25 pytest tests total.

  Stage 4 also covers a dynamic Orchestrator, bounded re-planning, and RAG-as-tool — still in progress,
  tracked as Parts B-D.
