# ARGUS — Autonomous Multi-Agent Insight Engine (Google ADK)

> **ARGUS** = **A**utonomous **R**easoning & **G**rounded **U**nderstanding **S**ystem.
> A hierarchical, self-critiquing multi-agent system built on Google's **Agent Development Kit (ADK, v1.25+)** that ingests a high-level analytical question, autonomously plans an investigation, dispatches specialist agents in parallel, runs **deterministic computation** (not just retrieval), adversarially red-teams its own conclusions, and emits a grounded, citation-backed insight memo.

**Flagship reference domain:** Company / sector **due-diligence & competitive-financial intelligence** (investor-grade theses).
**Key architectural flex:** the domain is *pluggable* via a capability registry — the same orchestration core retargets to any analytical domain by swapping the specialist roster and tool bindings.

---

## 1. Why this is explicitly *not* "basic RAG"

Basic RAG = embed query → retrieve chunks → stuff into context → answer. ARGUS is disqualified from that pattern by design:

| Basic RAG | ARGUS |
|---|---|
| Single retrieve-then-generate hop | Multi-hop autonomous **plan → fan-out → compute → critique → verify → synthesize** |
| Vector similarity is the only "reasoning" | **Deterministic arithmetic & statistical analysis via code execution** (the ground truth layer) |
| No self-correction | **Adversarial red-team `LoopAgent`** forces revision until a confidence gate passes |
| Answers can hallucinate freely | **Groundedness verifier** rejects any claim not traceable to a tool output |
| Stateless | **Session state + cross-session long-term memory** builds a reusable world model |
| One model, one prompt | **10–14 specialist agents** with distinct tools, instructions, and I/O contracts |

RAG *exists* inside ARGUS — but only as **one subordinate tool** among many, wrapped as an `AgentTool` the planner may or may not invoke.

---

## 2. Goals & Non-Goals

### Goals
- **G1** — Accept an open-ended analytical prompt (e.g. *"Assess Company X's competitive moat and 18-month downside risk"*) and return a structured, sourced memo with an explicit confidence score.
- **G2** — Demonstrate **dynamic (LLM-driven) orchestration** *and* **deterministic (workflow-agent) orchestration** in the same system.
- **G3** — Every quantitative claim is produced by **executed code**, never by an LLM "estimating."
- **G4** — Every qualitative claim is **grounded**: traceable to a specific tool result or rejected.
- **G5** — The system **critiques and revises itself** before a human ever sees output.
- **G6** — Ship with a **reproducible evaluation harness** (trajectory + groundedness + rubric scoring).
- **G7** — Deployable to **Cloud Run / Vertex AI Agent Engine** with streaming and tracing.

### Non-Goals
- Not a chatbot / conversational assistant (single-turn deep-analysis job, though sessions persist).
- Not a trading system — it produces analysis, never executes financial transactions.
- Not domain-locked — no hard-coded business logic outside the pluggable registry.

---

## 3. High-Level Architecture

```
                         ┌─────────────────────────────┐
                         │   Runner (SSE / bidi stream) │
                         └───────────────┬─────────────┘
                                         │
                              ┌──────────▼───────────┐
                              │   ORCHESTRATOR (LLM)  │  dynamic router
                              │  + Planner (ReAct)    │  builds task DAG
                              └──────────┬───────────┘
                                         │ transfer_to_agent / AgentTool
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                 │
 ┌──────▼──────┐   ParallelAgent (fan-out) specialists   ┌────────────────▼─────────┐
 │ Retrieval   │  ┌───────────┐ ┌───────────┐ ┌────────┐ │  Quantitative Agent      │
 │ Agent       │  │ Filings   │ │ Market/   │ │ News/  │ │  (Code Execution tool)   │
 │ (RAG-as-    │  │ Agent     │ │ Pricing   │ │ Senti- │ │  pandas / numpy / stats  │
 │  a-tool)    │  │           │ │ Agent     │ │ ment   │ │  → deterministic truth   │
 └─────────────┘  └───────────┘ └───────────┘ └────────┘ └──────────────────────────┘
        │                                │                                 │
        └────────────────┬───────────────┴─────────────────────────────────┘
                         │  all write to session.state[unique_key]
              ┌──────────▼──────────┐
              │  Synthesis Agent    │  drafts thesis from state
              └──────────┬──────────┘
                         │
        ┌────────────────▼───────────────────┐   LoopAgent (max_iterations, escalate=True on PASS)
        │  ADVERSARIAL REVIEW LOOP            │
        │  ┌───────────┐   ┌───────────────┐  │
        │  │ Red-Team  │──▶│ Refiner       │  │
        │  │ Critic    │   │ (revises)     │  │
        │  └───────────┘   └───────────────┘  │
        └────────────────┬───────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Groundedness       │  rejects unsourced claims → back to loop
              │  Verifier Agent     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Human-in-the-Loop  │  long-running tool: approve / redirect
              │  Gate (optional)    │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Memo / Report      │  → artifact (markdown / PDF) + memory write
              │  Generator Agent    │
              └─────────────────────┘
```

---

## 4. Agent Roster (specification)

Each agent lists: **ADK primitive**, **responsibility**, **tools**, **reads/writes** (`session.state` keys).

| # | Agent | ADK primitive | Responsibility | Tools | Reads → Writes |
|---|-------|---------------|----------------|-------|----------------|
| 1 | **Orchestrator** | `LlmAgent` (root, dynamic router) | Interpret the request, build a task DAG, delegate via `transfer_to_agent` / invoke sub-agents as `AgentTool`, decide when to stop | Planner (`BuiltInPlanner` / ReAct), sub-agents-as-tools | `user_query` → `plan_dag` |
| 2 | **Planner sub-module** | `PlanReActPlanner` attached to Orchestrator | Decompose into ordered/parallelizable subtasks; re-plan on failure | — | `plan_dag` → `revised_plan` |
| 3 | **Retrieval Agent** | `LlmAgent` wrapped as `AgentTool` | The *optional* RAG path — semantic search over an internal corpus/knowledge base | Vertex AI Search / vector store tool | `subtask` → `retrieval_ctx` |
| 4 | **Filings Agent** | `LlmAgent` (in `ParallelAgent`) | Pull & parse structured filings / documents; extract tables | OpenAPI tool (EDGAR-style), doc-parse function tool | `entity` → `filings_data` |
| 5 | **Market/Pricing Agent** | `LlmAgent` (in `ParallelAgent`) | Time-series prices, ratios, volatility windows | Function tool → market API, `google_search` fallback | `entity` → `market_data` |
| 6 | **News/Sentiment Agent** | `LlmAgent` (in `ParallelAgent`) | Recent events, sentiment scoring, catalyst detection | `google_search` built-in tool + sentiment function tool | `entity` → `sentiment_data` |
| 7 | **Quantitative Agent** | `LlmAgent` + **Code Execution tool** | The **deterministic truth layer** — runs real pandas/numpy/statsmodels over collected data; computes growth, margins, z-scores, anomalies | `built_in_code_execution` | `filings_data`, `market_data` → `quant_results` |
| 8 | **Cross-Join / Reconciliation Agent** | Custom `BaseAgent` | Joins facts across sources; flags contradictions between agents | function tools (fuzzy match, dedupe) | all `*_data` → `reconciled_facts` |
| 9 | **Synthesis Agent** | `LlmAgent` | Draft the thesis/insight from `reconciled_facts` + `quant_results` | — | `reconciled_facts`, `quant_results` → `draft_thesis` |
| 10 | **Red-Team Critic** | `LlmAgent` (in `LoopAgent`) | Attack the draft: find unsupported leaps, survivorship bias, cherry-picking; output `PASS` or structured `critique` | — | `draft_thesis` → `critique` |
| 11 | **Refiner** | `LlmAgent` (in `LoopAgent`) | Rewrite draft addressing `critique`; loop exits on `PASS` via `escalate=True` | — | `draft_thesis`, `critique` → `draft_thesis` (revised) |
| 12 | **Groundedness Verifier** | `LlmAgent` | Every claim must map to a source in state; strip/flag ungrounded claims; compute groundedness ratio | function tool (claim↔evidence matcher) | `draft_thesis`, all `*_data` → `verified_thesis`, `groundedness_score` |
| 13 | **HITL Gate** | `LlmAgent` + **long-running tool** | Pause for human approval/redirect on low-confidence or high-stakes outputs | `LongRunningFunctionTool` | `verified_thesis` → `human_decision` |
| 14 | **Memo Generator** | `LlmAgent` | Produce final memo artifact + persist key findings to long-term memory | artifact service, memory service | `verified_thesis` → `final_memo` (artifact) |

**Top-level composition (pseudocode):**

```python
gather = ParallelAgent(
    name="evidence_gather",
    sub_agents=[filings_agent, market_agent, sentiment_agent],  # each unique output_key
)

review_loop = LoopAgent(
    name="adversarial_review",
    sub_agents=[red_team_critic, refiner],
    max_iterations=4,          # hard ceiling
    # critic signals escalate=True in EventActions when critique == "PASS"
)

root_agent = SequentialAgent(
    name="argus_pipeline",
    sub_agents=[
        gather,
        quantitative_agent,        # code execution — deterministic layer
        reconciliation_agent,      # custom BaseAgent
        synthesis_agent,
        review_loop,
        groundedness_verifier,
        hitl_gate,                 # optional, config-gated
        memo_generator,
    ],
)
# The Orchestrator (LlmAgent) sits above root_agent for dynamic re-planning
# and can re-enter gather with new entities if the verifier finds gaps.
```

---

## 5. Orchestration Requirements

- **OR1** — Support **both** deterministic workflow orchestration (`SequentialAgent`/`ParallelAgent`/`LoopAgent`) *and* LLM-driven dynamic routing (`transfer_to_agent`, agents-as-`AgentTool`) in one system.
- **OR2** — Parallel specialists **must** write to **unique `session.state` keys** to avoid race conditions (ADK parallel agents share state).
- **OR3** — The adversarial loop **must** exit early on quality-gate pass (critic emits `escalate=True`) and also respect `max_iterations` as a hard stop.
- **OR4** — The Orchestrator may **re-plan**: if the Verifier reports missing evidence, it re-enters the gather phase with refined subtasks (bounded re-plan budget).
- **OR5** — Fan-out degree and specialist selection are **driven by the plan**, not hard-coded — a request touching only pricing shouldn't wake the Filings Agent.

---

## 6. Tooling Layer

| Tool | Type | Used by | Purpose |
|------|------|---------|---------|
| `google_search` | Built-in | Sentiment, Market | Current events / fallback facts |
| `built_in_code_execution` | Built-in | Quantitative Agent | Deterministic math/stats — **core differentiator** |
| Market data API | `FunctionTool` | Market Agent | Prices, fundamentals |
| Filings/doc API | `OpenAPIToolset` | Filings Agent | Structured document pull |
| Vector search | `AgentTool` (Retrieval Agent) | Orchestrator | Optional internal-corpus RAG |
| Claim↔evidence matcher | `FunctionTool` | Verifier | Groundedness scoring |
| Human approval | `LongRunningFunctionTool` | HITL Gate | Async human decision |
| External connectors | `MCPToolset` | any | Optional 3rd-party data via MCP |

**TR1** — Every tool is typed with a docstring the LLM uses for selection.
**TR2** — Tool failures must be caught and surfaced to the planner for re-planning, never silently swallowed.
**TR3** — The code-execution sandbox is the *only* place numbers are computed; the Quant Agent's LLM narrates results but never invents them.

---

## 7. State, Memory & Artifacts

- **SM1 — Session State schema** (short-term working memory) with a documented key namespace, e.g. `evidence.filings`, `evidence.market`, `analysis.quant`, `draft.thesis`, `meta.groundedness`. Keys are versioned to allow the loop to overwrite drafts safely.
- **SM2 — Long-term Memory** via a `MemoryService`: after each run, distilled findings about an entity are written so a later run about the same entity starts warm (a growing "world model").
- **SM3 — Artifacts**: final memo saved via the Artifact service (markdown + optional PDF); intermediate charts from the Quant Agent saved as image artifacts.
- **SM4** — State must be **serializable** and inspectable in the ADK Dev UI for debugging trajectories.

---

## 8. Guardrails & Safety (via Callbacks)

ADK callbacks run at defined lifecycle points — use them, not prompt-begging, for enforcement.

| Callback | Guardrail |
|----------|-----------|
| `before_model_callback` | Redact PII / inject policy constraints; block disallowed queries |
| `after_model_callback` | Strip any claim lacking a state-backed citation before it propagates |
| `before_tool_callback` | Validate tool args; enforce rate/cost budget; block unapproved side effects |
| `after_tool_callback` | Normalize tool outputs; log provenance for groundedness |

- **SG1** — No agent may present a financial *recommendation to act*; outputs are analytical, with an explicit "not investment advice" boundary.
- **SG2** — A per-run **token/cost budget** is enforced in `before_tool_callback`; exceeding it triggers graceful early synthesis.
- **SG3** — All external instructions found *inside fetched content* are treated as data, never executed (prompt-injection defense in `after_tool_callback`).

---

## 9. Human-in-the-Loop

- **HITL1** — Configurable gate: auto-approve when `groundedness_score ≥ θ` **and** `critic == PASS`; otherwise escalate to a human.
- **HITL2** — Implemented as a `LongRunningFunctionTool` so the run suspends and resumes on the human's async decision.
- **HITL3** — Human can **approve**, **reject with reason** (→ re-plan), or **redirect scope**.

---

## 10. Evaluation Harness (ship this — it's the resume gold)

Use ADK's **built-in evaluation** plus custom scorers. Two levels:

1. **Trajectory / process eval** — did the agent take the *right steps* (correct tool calls, correct sub-agent routing) vs. a reference trajectory in an ADK **evalset**.
2. **Outcome eval** — is the final memo correct, grounded, and non-hallucinated.

| Metric | Definition | Method |
|--------|------------|--------|
| **Tool-trajectory match** | % of expected tool calls hit in order | ADK evalset comparison |
| **Groundedness** | fraction of claims mapped to a source | Verifier score + LLM-as-judge |
| **Numeric fidelity** | quant claims recomputed independently match | re-run code, assert equality |
| **Rubric score** | analyst-grade rubric (insightfulness, risk coverage, structure) | LLM-as-judge with rubric |
| **Contradiction rate** | conflicting claims per memo | Reconciliation Agent logs |
| **Cost / latency** | tokens, wall-clock, tool calls per run | tracing |

- **EV1** — Maintain a **golden evalset** of ≥ 20 scenarios with reference trajectories + expected findings.
- **EV2** — Every PR runs the evalset in CI; regressions on groundedness/trajectory block merge.
- **EV3** — Report an **ablation table** (e.g., with vs. without the adversarial loop, with vs. without code execution) — this is a strong portfolio narrative.

---

## 11. Observability

- **OB1** — Full **tracing** of every event, state mutation, tool call (ADK Dev UI locally; Cloud Trace / OpenTelemetry in prod).
- **OB2** — Per-agent latency, token, and cost breakdown dashboards.
- **OB3** — Persist trajectories for offline replay and eval curation.

---

## 12. Deployment & Runtime

- **DEP1** — Local dev via ADK CLI + Dev UI (`adk web`).
- **DEP2** — Prod deploy to **Cloud Run** (containerized) or **Vertex AI Agent Engine** (managed sessions/memory).
- **DEP3** — **Streaming**: stream intermediate reasoning + partial memo to the client (SSE / bidi via the Live toolkit) so the user watches the investigation unfold.
- **DEP4** — Stateless request handling with externalized session/memory services for horizontal scale.
- **DEP5** — Model-agnostic config: default Gemini 3 Flash for specialists, a stronger model for Synthesis/Critic; swappable per agent.

---

## 13. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Latency | End-to-end deep-analysis run ≤ target budget (e.g. 90–180s) with streaming feedback throughout |
| Cost | Per-run token budget enforced via callback; ablation shows cost/quality tradeoff |
| Concurrency | ≥ N concurrent runs via externalized state; parallel specialists async, non-blocking |
| Reliability | Any single specialist failure degrades gracefully (partial evidence flagged, not crash) |
| Security | Secrets in Secret Manager; least-privilege tool creds; injection defense on all fetched content |
| Reproducibility | Fixed seeds where possible; evalset + CI; pinned ADK version |

---

## 14. Data Sources (reference domain)

- Structured filings / documents (OpenAPI).
- Market/pricing time series (function-tool API).
- News & web (built-in search).
- Optional internal corpus (vector store, RAG-as-a-tool).
> All swappable per domain via the capability registry.

---

## 15. Phased Roadmap

| Phase | Deliverable | Proves |
|-------|-------------|--------|
| **P0** | Single `LlmAgent` + one tool, running in Dev UI | ADK basics |
| **P1** | `ParallelAgent` fan-out of 3 specialists → `SequentialAgent` synthesis | multi-agent orchestration + shared state |
| **P2** | Quantitative Agent with code execution | deterministic (non-RAG) analysis |
| **P3** | `LoopAgent` adversarial critic/refiner with `escalate` exit | self-correction |
| **P4** | Groundedness Verifier + callback guardrails | trustworthiness |
| **P5** | Dynamic Orchestrator + re-planning | dynamic orchestration |
| **P6** | Memory + HITL + streaming | production shape |
| **P7** | Eval harness + ablations + CI | rigor / measurability |
| **P8** | Cloud Run / Agent Engine deploy + tracing | production deployment |

---

## 16. Tech Stack

| Layer | Choice |
|-------|--------|
| Agent framework | Google ADK (Python, v1.25+) |
| Models | Gemini 3 Flash (specialists) / Gemini 3 Pro (synthesis, critic) — model-agnostic |
| Compute layer | ADK code-execution sandbox (pandas, numpy, statsmodels) |
| Retrieval | Vertex AI Search / vector store (as a subordinate tool only) |
| Sessions/Memory | ADK Session + Memory services (Vertex Agent Engine in prod) |
| Deploy | Cloud Run or Vertex AI Agent Engine |
| Observability | ADK Dev UI, Cloud Trace / OpenTelemetry |
| Eval | ADK built-in eval + custom LLM-as-judge scorers |
| CI | evalset gate on groundedness + trajectory |

---

## 17. Stretch Goals

- **Debate mode**: two Synthesis agents argue opposing theses (bull vs. bear), a Judge agent scores — richer than a single critic.
- **Confidence-calibrated output**: verifier emits calibrated probability, validated against outcomes over time.
- **Self-improving memory**: memory writes include "what surprised me," so future runs pre-load known blind spots.
- **Multi-domain registry demo**: same core retargeted live from finance → scientific-literature review → incident RCA, proving domain-pluggability.
- **Cost-aware planner**: planner chooses cheaper tool paths when the budget tightens.

---

## 18. Resume / Interview Talking Points

- "Built a **hierarchical multi-agent system on Google ADK** combining **deterministic workflow orchestration** (Sequential/Parallel/Loop agents) with **LLM-driven dynamic routing** and re-planning."
- "Separated a **deterministic computation layer** (code-execution agent over pandas/statsmodels) from LLM reasoning, so no quantitative claim is ever hallucinated."
- "Implemented an **adversarial self-critique loop** (`LoopAgent` with `escalate`-based early exit) plus a **groundedness verifier** that rejects unsourced claims — reduced hallucinated claims by *[measured %]* in ablation."
- "Shipped a **reproducible evaluation harness** with trajectory matching, groundedness, and LLM-as-judge rubric scoring, gated in CI."
- "Enforced **safety and cost budgets via ADK callbacks**, added **human-in-the-loop** via long-running tools, and deployed to **Vertex AI Agent Engine** with full tracing and streaming."

---

## 19. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Loop non-convergence | `max_iterations` hard cap + convergence-on-PASS + budget cutoff |
| Parallel state races | unique `output_key` per specialist (mandated in OR2) |
| Cost blowup | per-run token budget in `before_tool_callback` |
| Verifier over-rejects | tune groundedness threshold θ; keep flagged-not-dropped mode |
| Over-engineering | phased roadmap — each phase is independently demoable |

---

*Spec is intentionally domain-pluggable: swap Section 4's specialist roster and Section 6's tool bindings to retarget ARGUS to any analytical domain without touching the orchestration core.*
