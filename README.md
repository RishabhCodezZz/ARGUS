# ARGUS

Autonomous Reasoning & Grounded Understanding System — a hierarchical multi-agent due-diligence
system built on Google's Agent Development Kit (ADK). An **Orchestrator** (`argus/agents/orchestrator.py`)
reads the request and decides how much of the system it actually needs:

- **Narrow question** (just the price, just the filings, just recent news) → routes directly to the one
  matching specialist agent and formats its answer. The other specialists never run.
- **Qualitative background question** (competitive position, strategic outlook, leadership/governance —
  nothing a numeric tool can answer) → routes to a Retrieval agent that searches an internal knowledge
  base of analyst notes (`argus/agents/retrieval.py`, `argus/tools/knowledge_base.py`) and reports back
  what it finds, without inventing anything beyond it.
- **Broad request** ("analyze X", "full due-diligence on X") → routes to the complete pipeline:
  1. **Gathers** filings, market, and sentiment data in parallel (`argus/agents/gather.py`).
  2. **Computes** real financial metrics — YoY growth, margins, CAGR, debt trend, price volatility —
     via executed Python (pandas/numpy), never LLM-estimated (`argus/agents/quant.py`).
  3. **Reconciles**: a custom `BaseAgent` with zero LLM calls cross-checks each headline's sentiment
     against the real fiscal-year performance it discusses, flagging any contradiction
     (`argus/agents/reconciliation.py`).
  4. **Drafts** an analytical thesis citing only computed numbers and calling out any contradiction
     found (`argus/agents/synthesis.py`).
  5. **Red-teams itself**: a Critic checks the draft for unsupported claims or cherry-picking; a
     Refiner fixes what it finds. Loops until it passes or 4 iterations run out
     (`argus/agents/critic.py`, `refiner.py`).
  6. **Verifies groundedness**: every numeric claim is checked against the evidence by a deterministic
     matcher — ungrounded claims get flagged in place, and a `meta.groundedness` score is recorded
     (`argus/agents/verifier.py`, `argus/tools/claim_matcher.py`).

If the requested company isn't found, the Orchestrator gets **one bounded retry**: it picks the closest
available name and tries again, disclosing the substitution rather than silently swapping it in. A
genuinely nonexistent request still fails gracefully after that one attempt, never looping.

A broad analysis also checks (and updates) **long-term memory**: before running, the Orchestrator asks
whether ARGUS has looked at this company before; after a successful run, it saves a one-line distilled
finding for next time (`argus/tools/memory.py`) — so a second request about the same company, even in a
brand-new session, starts warm.

Before presenting a broad result, it passes through a **human-in-the-loop gate**
(`argus/tools/hitl.py`): a high-groundedness, critic-passed result auto-releases with no interruption;
anything below the bar **genuinely pauses the run** and asks a human to approve, reject with a reason,
or redirect scope, resuming only once they reply.

Four guardrail callbacks (`argus/callbacks/guardrails.py`) run on every model/tool call across the
whole system: blocking direct trade-advice requests, stripping unhedged action language from any
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

  Stage 4 also covers a dynamic Orchestrator, bounded re-planning, and RAG-as-tool, tracked as Parts B-D
  below.
- **Stage 4, Part B (2026-07-24)** — Dynamic Orchestrator. `orchestrator_agent` is now `root_agent`,
  replacing the old fixed pipeline as the entry point. It's a plain `LlmAgent` with 4 `AgentTool`s: the
  three gather specialists individually, plus the entire existing pipeline wrapped whole as
  `full_analysis_pipeline`. Its instruction routes narrow questions to one specialist directly and
  broad requests to the full pipeline, presenting the pipeline's result verbatim since it's already
  drafted, red-teamed, and groundedness-checked.

  Verified live: asking about Acme Corp's recent stock price called *only* `market_agent` — the other
  specialists and the full pipeline never fired. Asking for "a full due-diligence analysis of Globex"
  called `full_analysis_pipeline`, which ran the complete 6-stage chain internally (every state key
  populated in one combined delta) and returned the drafted thesis, including the Reconciliation
  Agent's contradiction callout — zero regressions to Stages 1-4A through the new routing layer.
- **Stage 4, Part C (2026-07-24)** — Bounded re-planning (OR4). `check_gather_status`
  (`argus/agents/orchestrator.py`) gives the Orchestrator a way to re-enter gathering with a "refined
  subtask" — concretely, the closest available company name — capped by a real deterministic counter
  (`meta.replan_count`, budget 1), the same belt-and-suspenders pattern as Stage 2's
  `exit_loop`/`max_iterations`.

  Live testing found and fixed two real bugs: the first design had the Orchestrator infer "missing
  evidence" from the pipeline's returned prose, which silently never triggered a retry (a total gather
  failure still produces a well-formed "Analytical Thesis... Confidence Level: Low," which doesn't read
  as an unambiguous failure signal); fixed by checking `evidence.*` state directly for an error, instead
  of trusting the LLM's own reading of the text. Second, a successful retry was presenting a substituted
  company's analysis with no disclosure that a substitution happened; fixed by requiring the Orchestrator
  to say so explicitly. Verified end-to-end: asking for "Initech" (absent from the dataset) correctly
  fails, retries once with "Globex" (the closest match), succeeds, and opens with *"I couldn't find data
  for Initech, so here's a full due-diligence analysis of Globex instead:"* before the verbatim,
  still-correct thesis. This test needed two full pipeline runs in one turn, which sits right at the
  free tier's 15-requests/minute ceiling — confirmed live that this quota is per-project, not per-key
  (a second API key hit the identical limit immediately).
- **Stage 4, Part D (2026-07-25)** — RAG-as-tool. `retrieval_agent` (`argus/agents/retrieval.py`) is a
  plain `LlmAgent` wrapping one new tool, `retrieve_knowledge` (`argus/tools/knowledge_base.py`), over a
  new mock knowledge base (`data/knowledge_base/{acme_corp,globex}.json`) of analyst-note-style
  documents — competitive landscape, strategic outlook, leadership/governance. Retrieval itself is
  deliberately a stub, keyword overlap rather than embeddings/vector search, matching the spec's own
  framing of RAG as "one subordinate tool among many," not a required path. Wired into the Orchestrator
  as a fourth `AgentTool` with its own routing branch: qualitative background questions that no
  structured data tool can answer go here, writing to a new `retrieval.context` state key.

  Verified live: asking *"What's Globex's competitive position in its industry?"* routed to
  `retrieval_agent` alone — a genuinely new third path, distinct from the narrow-specialist and
  broad-pipeline routes — which retrieved both the "Competitive Landscape" and "Strategic Outlook and
  Risk Factors" documents and synthesized a faithful answer with no facts beyond what they said. 6 new
  pytest tests, 31 total project-wide.

  **Stage 4 is now fully complete** — Reconciliation, dynamic Orchestrator, bounded re-planning, and
  RAG-as-tool, all built and verified live, one part at a time.
- **Stage 5, Part A (2026-07-25)** — Long-term memory. `recall_prior_findings` and `remember_finding`
  (`argus/tools/memory.py`) are both tools the Orchestrator calls itself: before a broad analysis, check
  whether ARGUS has a prior distilled finding for this company; after a successful run, save one. This
  is a genuinely different store from `session.state` — it persists across sessions, not just across
  turns in one chat.

  The first design put the save step in an `after_agent_callback` on `full_analysis_pipeline` instead —
  it looked like the natural place ("the run just finished, persist it"), passed every test, and ran
  with no visible error. It was also completely broken: `full_analysis_pipeline` only ever runs as an
  `AgentTool` from the Orchestrator, and ADK's `AgentTool` runs the wrapped agent through a brand-new,
  throwaway memory service every single call — writes landed somewhere real, just somewhere nobody would
  ever read from again. Found live only because a second run's memory lookup came back empty despite a
  clean first run with no errors anywhere in the trace. Fixed by moving the save into an explicit tool
  on the Orchestrator itself, the one place actually connected to the app's real memory store.

  Verified live end-to-end: asked for a full analysis of Acme Corp — no prior finding, pipeline ran,
  finding saved. Started a genuinely new chat session and asked again — the recall tool found the exact
  sentence the first run saved. 7 new pytest tests, 38 total project-wide.
- **Stage 5, Part B (2026-08-01)** — Human-in-the-Loop Gate. `evaluate_gate`, `request_human_approval`
  (a `LongRunningFunctionTool` — the run genuinely suspends), and `record_human_decision`
  (`argus/tools/hitl.py`) are three more Orchestrator-level tools, not a separate agent: a high-
  groundedness, critic-passed result auto-releases; anything below the bar pauses for a human to
  approve, reject with a reason, or redirect scope. The threshold (0.98) is set against real observed
  scores — a clean run scores 1.00, Stage 3's adversarial prompt scores ≈0.95 — so both paths are
  reachable with genuine data, not contrived inputs.

  Same root cause as Part A, this time caught by reading ADK's source before writing any code instead
  of live: `AgentTool` doesn't forward a paused sub-agent's long-running signal to its parent, only
  its state — so the gate has to live at the Orchestrator level, exactly like the memory tools. A
  second bug WAS only found live: the model initially explained an escalation in chat text without
  actually calling the tool that pauses the run, so nothing genuinely paused — the same "narrates
  instead of acting" gap `exit_loop` already had back in Stage 2, fixed the same way (explicit "these
  are two parts of one action" instruction language).

  Verified live across all three reachable outcomes with real data: a clean run auto-approves with no
  interruption; the Stage 3 adversarial prompt reliably escalates and genuinely suspends (confirmed via
  the run's `longRunningToolIds`, not assumed); replying `{"decision": "approve"}` in the Dev UI's
  pending-call box resumes the run, discloses "(Released after human review)", and saves to memory;
  replying `{"decision": "reject", "reason": "..."}` instead withholds the analysis entirely, states
  the reason, and correctly skips saving to memory. `redirect` is implemented but not live-verified —
  a disclosed gap given free-tier quota cost. 7 new pytest tests, 45 total project-wide.
