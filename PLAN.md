# ARGUS — Step-by-Step Execution Plan (Beginner Edition)

## Context

`ARGUS_ADK_Requirements.md` specifies an ambitious hierarchical **multi-agent** system on
Google's Agent Development Kit (ADK): it plans an investigation, fans out specialist agents in
parallel, runs **real code** for math (not LLM guessing), **red-teams its own draft**, verifies
every claim is **grounded** in a source, and emits a sourced memo. The reference domain is
company/financial due-diligence, but the architecture is meant to be domain-pluggable.

You are a **beginner**. The full spec is ~14 agents plus evals, cloud deploy, and tracing — far
too much to build in one go. This plan converts the spec into an **incremental, low-risk build**
where every stage is independently runnable and demoable, following the doc's own P0–P8 roadmap
(Section 15) but front-loading a working end-to-end skeleton.

**Direction chosen (drives this plan):**
- **Goal:** Learn agent-building hands-on **and** produce a portfolio/interview piece → we keep the
  eval harness, ablation study, and a cloud deploy as later stages (the spec calls these "resume gold").
- **Scope:** **Simplified-first** — build a "Minimum Viable ARGUS" (~4 agents) end-to-end, get it
  green, then add one capability at a time.
- **Data/infra:** **Mock/sample data + a free Gemini API key, fully local.** No cloud billing, no
  real financial APIs until much later. This decouples *learning agents* from *wrangling data APIs*
  (the #1 beginner time-sink).

**Verified framework facts (mid-2026):** install `google-adk`; needs Python 3.10+; scaffold with
`adk create`, run the visual debugger with `adk web` at http://localhost:8000. Parallel agents share
`session.state` → each MUST write a unique key. Loops stop on `max_iterations` **or** an agent
emitting `escalate=True`. Real current model IDs: **`gemini-3.6-flash`** (fast/cheap, for specialists)
and **`gemini-3.1-pro`** (stronger, for synthesis/critic). Gemini API key comes free from Google AI
Studio; `GOOGLE_GENAI_USE_VERTEXAI=FALSE` + `GOOGLE_API_KEY=…` runs everything locally without GCP.

---

## The mindset (read this first)

1. **Never build more than one new agent/idea at a time.** After each stage the whole pipeline must
   still run in `adk web`. If it's broken, don't add anything until it's green again.
2. **Commit to git after every green stage.** Each stage is a demoable checkpoint (and a resume bullet).
3. **Numbers come from code, words come from the LLM.** This one rule (spec G3/TR3) is the heart of
   the project — protect it.
4. **Mock data is a feature, not a shortcut.** Fake-but-realistic company data lets you exercise 100%
   of the orchestration logic without ever touching a real API. Swap in real data only at the end.
5. **Don't touch the cloud until Stage 7.** Everything works locally first.

---

## Prerequisites & environment setup (do once, before Stage 0)

- [ ] Install **Python 3.11** (3.10+ works). Confirm: `python --version`.
- [ ] Install **git** and **VS Code** (or your editor of choice).
- [ ] Get a **free Gemini API key** from Google AI Studio (aistudio.google.com → "Get API key").
- [ ] Create an isolated Python virtual environment in the project folder:
      `python -m venv .venv` then activate (`.venv\Scripts\Activate.ps1` in PowerShell).
- [ ] `pip install google-adk` (this pulls in the ADK CLI + libraries).
- [ ] Confirm the CLI works: `adk --version`.
- [ ] `git init` and add a `.gitignore` that excludes `.venv/`, `.env`, `__pycache__/`, `*.pyc`.

> Security note: the API key lives in a `.env` file that is **git-ignored** — never commit it.

---

## Recommended project structure (built up over the stages)

```
Argus/
  ARGUS_ADK_Requirements.md        # the spec (already here)
  CLAUDE.md                        # context Claude auto-loads each session
  PLAN.md                          # this plan
  .venv/                           # virtualenv (git-ignored)
  .gitignore
  argus/                           # the ADK "agent app" package
    __init__.py                    # `from . import agent`
    agent.py                       # defines `root_agent` (ADK entrypoint)
    config.py                      # model IDs, thresholds, budgets in ONE place
    state_keys.py                  # the session.state key namespace (constants)
    agents/                        # one file per specialist as they're added
      gather.py  quant.py  synthesis.py  critic.py  refiner.py  verifier.py ...
    tools/                         # FunctionTools (mock data loaders, matchers, etc.)
      mock_data.py  sentiment.py  claim_matcher.py ...
    callbacks/                     # guardrail callbacks (added in Stage 3)
  data/                            # mock/sample company data (JSON/CSV)
    acme_corp/ globex/ ...
  eval/                            # evalsets + custom scorers (Stage 6)
  tests/                           # pytest unit tests
  README.md                        # architecture + how-to-run (write as you go)
```

**Conventions to lock in on day one:**
- **State-key namespace** (spec SM1) — decide it once in `state_keys.py`, e.g.
  `evidence.filings`, `evidence.market`, `evidence.sentiment`, `analysis.quant`,
  `analysis.reconciled`, `draft.thesis`, `review.critique`, `meta.groundedness`, `meta.human`.
  Parallel specialists each own a **unique** key (spec OR2) — this prevents the classic race bug.
- **All models/thresholds/budgets live in `config.py`** so you swap models in one place (spec DEP5).
- **Every tool has a clear docstring** — the LLM reads it to decide when to call the tool (spec TR1).

---

## The staged build plan

Each stage below = one or more roadmap phases from the spec, ordered so you always have a running
system. "DoD" = Definition of Done (how you know the stage is complete).

### Stage 0 — Hello, ADK (spec P0)
**Goal:** One `LlmAgent` with one trivial tool, visible in the Dev UI.
- [ ] Scaffold the `argus/` package (`agent.py`, `__init__.py`, `.env`).
- [ ] Define `root_agent = LlmAgent(model="gemini-3.6-flash", instruction=…, tools=[one_tool])`.
- [ ] Write one `FunctionTool` (e.g. `get_company_snapshot(name)` returning a hardcoded dict).
- [ ] Run `adk web`, open localhost:8000, chat with it, watch it call the tool in the trace view.
- **DoD:** You can ask a question and see the agent invoke your tool in the Dev UI trace.
- **You learn:** ADK project shape, `LlmAgent`, `FunctionTool`, the Dev UI, API-key config.
- **Pitfall:** wrong folder layout → `adk web` won't find the agent. Match the structure above exactly.

### Stage 1 — Minimum Viable ARGUS (spec P1 + P2) ← the big one
**Goal:** A real end-to-end pipeline on mock data: gather (parallel) → compute (code) → synthesize.
- [ ] Create **mock data** in `data/` for 1–2 fake companies: filings-style financials (revenue,
      net income, debt by year), a price time-series, and a few news headlines. (JSON/CSV.)
- [ ] Build 3 gather specialists as `LlmAgent`s, each with a mock-data `FunctionTool` and a **unique
      `output_key`**: Filings, Market/Pricing, News/Sentiment. Wrap them in a **`ParallelAgent`**.
- [ ] Build the **Quantitative Agent** using ADK's **built-in code executor** (Gemini code execution).
      It reads the gathered numbers from state and runs **real pandas/numpy** to compute growth rates,
      margins, and z-scores. *(Verify the exact executor class name in current ADK docs.)*
- [ ] Build the **Synthesis Agent** (`LlmAgent`, use `gemini-3.1-pro`) that drafts a thesis from the
      quant results + gathered evidence in state.
- [ ] Compose with **`SequentialAgent`**: `[parallel_gather, quant_agent, synthesis_agent]`. Make
      this the new `root_agent`.
- **DoD:** One prompt → 3 agents run in parallel → code computes real numbers → a drafted memo comes
      out, all visible step-by-step in the Dev UI. The numbers in the memo match a hand-check of the
      mock data.
- **You learn:** parallel fan-out, shared state with unique keys, sequential orchestration, and the
      core differentiator — **deterministic computation vs. LLM guessing** (G3).
- **Pitfall:** letting the LLM "estimate" numbers. The Synthesis Agent must only *narrate* values the
      Quant Agent computed, never invent them (TR3).

### Stage 2 — Self-critique loop (spec P3)
**Goal:** The system attacks and improves its own draft before you see it.
- [ ] **Red-Team Critic** (`LlmAgent`): reads `draft.thesis`, hunts unsupported leaps / cherry-picking,
      outputs either `PASS` or a structured critique.
- [ ] **Refiner** (`LlmAgent`): rewrites the draft to address the critique, writes back to `draft.thesis`.
- [ ] Wrap `[critic, refiner]` in a **`LoopAgent`** with `max_iterations=4`; the critic emits
      `escalate=True` (via EventActions / an `exit_loop` tool) when it says `PASS` so the loop stops early.
- [ ] Insert the loop into the sequence after Synthesis.
- **DoD:** In the trace you can watch the draft get criticized and rewritten, and the loop exit early
      on `PASS` (not always running all 4 iterations).
- **You learn:** iterative `LoopAgent`, early-exit via `escalate`, the LLM-as-critic pattern (G5).
- **Pitfall:** infinite/non-converging loops → the `max_iterations` hard cap is mandatory (Risk table).

### Stage 3 — Groundedness + guardrail callbacks (spec P4)
**Goal:** Trustworthiness — no claim survives unless it traces to a source; safety enforced in code.
- [ ] **Groundedness Verifier** (`LlmAgent` + a `claim↔evidence` matcher `FunctionTool`): checks each
      claim against the `evidence.*` state; flags/strips unsupported ones; writes a `meta.groundedness`
      ratio and a `verified.thesis`.
- [ ] Add **callbacks** (not prompt-begging) in `callbacks/`:
      `after_model_callback` strips citation-less claims; `before_tool_callback` enforces a token/cost
      budget and validates tool args; `after_tool_callback` logs provenance + treats any instructions
      found *inside fetched content* as data, not commands (prompt-injection defense, SG3).
- [ ] Add the "**not investment advice**" boundary to final output (SG1).
- **DoD:** Feed the system a deliberately unsupported claim in the mock data and watch the verifier
      flag it and the groundedness score drop.
- **You learn:** anti-hallucination/groundedness, and ADK **callbacks** as the real enforcement layer (G4).
- **Pitfall:** verifier over-rejecting → keep a "flag, don't silently drop" mode and a tunable threshold θ.

### Stage 4 — Reconciliation + dynamic orchestration (spec P5)
**Goal:** Cross-check sources, and make routing LLM-driven instead of a fixed sequence.
- [ ] **Reconciliation Agent** (custom **`BaseAgent`**): joins facts across sources, flags contradictions
      into `analysis.reconciled`. (Insert before Synthesis.)
- [ ] **Orchestrator** (`LlmAgent` with a planner, e.g. `PlanReActPlanner`): sits on top, interprets the
      request, and delegates via `transfer_to_agent` / agents-as-`AgentTool`. Fan-out is **plan-driven**,
      not hard-coded — a pricing-only question shouldn't wake the Filings agent (OR5).
- [ ] **Bounded re-planning** (OR4): if the verifier reports missing evidence, the orchestrator can
      re-enter the gather phase with refined subtasks, within a re-plan budget.
- [ ] Turn the optional **RAG path** into an `AgentTool` the orchestrator *may* call (this is where
      "RAG-as-just-one-tool" from the spec lives) — can start as a stub over your mock docs.
- **DoD:** Two different prompts cause *different* specialists to run (dynamic routing visible in trace);
      a forced "missing evidence" case triggers one bounded re-plan.
- **You learn:** custom agents, dynamic vs. deterministic orchestration in one system (G2), re-planning.
- **Pitfall:** unbounded re-planning → always cap the re-plan budget.

### Stage 5 — Production shape: memory, human-in-the-loop, streaming (spec P6)
**Goal:** The system remembers across runs, can pause for a human, and streams its thinking live.
- [ ] **Long-term memory** via a `MemoryService`: after a run, distill findings about the entity and
      persist them so a later run on the same entity "starts warm" (SM2). Start with the in-memory/local
      service.
- [ ] **HITL Gate**: a `LongRunningFunctionTool` that suspends the run for human approve / reject-with-
      reason / redirect when `groundedness < θ` or stakes are high; auto-approve otherwise (HITL1–3).
- [ ] **Memo Generator** (`LlmAgent`): writes the final memo as an **artifact** (markdown; optional PDF),
      plus saves Quant charts as image artifacts (SM3).
- [ ] Turn on **streaming** in the Dev UI so intermediate reasoning + partial memo stream to the client.
- **DoD:** Run the same company twice and see run #2 use remembered context; trigger the HITL pause and
      resume it; get a saved memo artifact.
- **You learn:** session vs. long-term memory, human-in-the-loop with long-running tools, artifacts, streaming.
- **Pitfall:** memory schema sprawl — keep distilled findings small and structured.

### Stage 6 — Evaluation harness + ablations + CI (spec P7) ← resume gold
**Goal:** *Prove* the system works and that each feature matters. This is your strongest portfolio asset.
- [ ] Build a **golden evalset** of ≥ 20 scenarios (ADK evalset format) with reference trajectories +
      expected findings (EV1).
- [ ] Wire ADK's built-in eval (`adk eval`) plus custom scorers: **tool-trajectory match**,
      **groundedness**, **numeric fidelity** (re-run the code, assert equal), **rubric score**
      (LLM-as-judge), **contradiction rate**, **cost/latency** (Section 10 table).
- [ ] **Ablation table** (EV3): run with vs. without the critic loop, and with vs. without code
      execution; record how much groundedness / correctness drops. *This becomes your headline metric.*
- [ ] Add a **CI workflow** (GitHub Actions) that runs the evalset; regressions on groundedness/
      trajectory fail the build (EV2).
- **DoD:** `adk eval` runs green locally and in CI; you have a filled-in ablation table with real numbers.
- **You learn:** how to actually *measure* an AI system — the most under-taught, most valued skill.
- **Pitfall:** evals that only check final answers. Include **trajectory** (did it take the right steps).

### Stage 7 — Cloud deploy + tracing (spec P8) ← resume gold
**Goal:** Ship it and observe it in production shape.
- [ ] Containerize and deploy to **Cloud Run** (simplest) or **Vertex AI Agent Engine** (managed
      sessions/memory). This is where you finally set up a Google Cloud project + billing.
- [ ] Externalize session/memory services so requests are stateless/horizontally scalable (DEP4).
- [ ] Turn on **tracing**: Cloud Trace / OpenTelemetry; per-agent latency, token, and cost breakdowns (OB1–3).
- [ ] Secrets in **Secret Manager**, least-privilege tool creds (spec Security NFR).
- **DoD:** A public/authenticated URL runs a full analysis with streaming, and traces show up in the dashboard.
- **You learn:** deploying + observing agentic systems in the cloud.
- **Pitfall:** doing this too early. Everything above works locally first — only lift-and-shift at the end.

---

## 14 agents → where each one lands

| Stage | Agents added |
|------|----------------|
| 1 | Filings, Market/Pricing, News/Sentiment (parallel), Quantitative, Synthesis |
| 2 | Red-Team Critic, Refiner (loop) |
| 3 | Groundedness Verifier |
| 4 | Reconciliation (custom BaseAgent), Orchestrator + Planner, Retrieval/RAG-as-tool |
| 5 | HITL Gate, Memo Generator |

## Requirements coverage map (so nothing is silently dropped)

- **Orchestration OR1–OR5** → Stages 1 (workflow), 4 (dynamic + re-plan).
- **State/Memory/Artifacts SM1–SM4** → SM1 conventions day one; SM2/SM3 Stage 5; SM4 (Dev UI) throughout.
- **Guardrails SG1–SG3 + callbacks** → Stage 3.
- **Tooling TR1–TR3** → docstrings from Stage 0; TR3 (code-only numbers) Stage 1; TR2 (fail→re-plan) Stage 4.
- **HITL1–3** → Stage 5. **Eval EV1–EV3** → Stage 6. **Deploy/Obs DEP/OB** → Stage 7.
- **Stretch goals (Section 17)** — debate mode, cost-aware planner, multi-domain registry demo — leave
  for *after* Stage 6 if time allows; each is a great extra portfolio bullet, none are required.

---

## Data strategy (mock now, real later)

- **Now:** hand-authored JSON/CSV per fake company under `data/`, loaded by `FunctionTool`s. Include one
  company with a deliberately contradictory/unsupported fact to exercise the verifier and reconciler.
- **Later (optional, post-Stage 6):** swap the mock loaders for real tools — an EDGAR-style filings
  `OpenAPIToolset`, a market-data `FunctionTool`, `google_search` for news — behind the *same* interfaces,
  so the orchestration core doesn't change (this proves the "pluggable registry" claim).

## Top risks & how to avoid rabbit holes

- **Trying to build all 14 agents at once** → follow the stages; keep the pipeline runnable at every step.
- **Fighting real data APIs early** → mock data until Stage 6+.
- **Cloud setup friction** → deferred entirely to Stage 7.
- **Loop never converging / cost blowup** → `max_iterations` cap + `escalate` exit + token budget callback.
- **Parallel state races** → unique `output_key` per specialist, defined once in `state_keys.py`.
- **Chasing exact ADK API signatures from memory** → check current ADK docs (adk.dev / the GitHub repo)
  for exact class names (code executor, planner, memory/artifact services) when you reach each stage.

## How to verify (end-to-end, per stage)

1. **Primary loop:** after each stage, run `adk web`, send a test prompt, and step through the trace in
   the Dev UI — confirm the newly added agent/behavior fires and state keys populate as expected.
2. **Numeric fidelity:** hand-compute a couple of metrics from the mock data and confirm the memo matches.
3. **Negative tests:** feed an unsupported/contradictory fact and confirm the verifier/reconciler catch it.
4. **Unit tests:** `pytest` over pure functions (tools, matchers) as you add them.
5. **From Stage 6:** `adk eval` over the golden evalset, locally and in CI, must stay green.

## Immediate next actions

1. Set up the environment (venv + `pip install google-adk` + Gemini API key + `.env` + `.gitignore`).
2. Scaffold the `argus/` package and get **Stage 0** (one agent + one tool) running in `adk web`.
3. Author the first mock company dataset and begin **Stage 1**.

> Each stage ends with a git commit and a one-line note in `README.md` — that running log becomes both
> your progress tracker and your resume/interview narrative.
