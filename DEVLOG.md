# ARGUS — Full Development Log

Every stage of this project, what was built, and how it was verified live. See the main [README](README.md) for the project overview and results.

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
  `gemini-flash-latest`, and switch back once iteration slows down.
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
- **Stage 5, Part A (2026-07-25)** — Runtime memory (called "long-term memory" at the time; renamed
  2026-09-03 — it's an in-process store cleared on restart, see that entry below).
  `recall_prior_findings` and `remember_finding` (`argus/tools/memory.py`) are both tools the
  Orchestrator calls itself: before a broad analysis, check whether ARGUS has a prior distilled finding
  for this company; after a successful run, save one. This is a genuinely different store from
  `session.state` — it persists across sessions, not just across turns in one chat (though only for the
  life of the running process).

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
- **Stage 5, Parts C + D (2026-08-01)** — Memo artifacts and streaming. **Stage 5 is now fully
  complete.** `save_memo` (`argus/tools/memo.py`) writes the final approved thesis to a real,
  downloadable markdown artifact — spec agent #14, implemented as a plain tool rather than another
  `LlmAgent`, since the thesis is already drafted, red-teamed, verified, and gate-cleared by the time
  it runs; regenerating it through one more model call would reintroduce the exact risk that chain
  exists to remove, for an API call the free-tier quota can't spare. Gated by the same
  approved-only instruction clause as `remember_finding`, and called alongside it. Streaming needed no
  new code at all: the Dev UI already ships a working toggle, so Part D was pure verification.

  This closes out a running finding across the whole stage: `AgentTool`'s inner runner now has a
  complete, checked four-row picture — session state and artifacts cross the boundary back to the
  parent; the runtime memory service, the long-running-pause signal, and (confirmed this part) the
  streaming mode do not. That last one explains, precisely, why streaming works for the Orchestrator's own turns
  and not for anything happening inside the wrapped pipeline.

  Verified live: two independent auto-approved runs each correctly saved a memo artifact; fetching it
  back and decoding it as UTF-8 confirmed the embedded thesis matches session state byte-for-byte (a
  first, naive decode attempt falsely suggested a mismatch — a bug in the verification script's
  character handling, caught and fixed mid-check, not in the product). The streaming split was
  confirmed for free from server logs already produced by these same runs — zero extra API calls —
  showing `stream: True` for every Orchestrator-level model call and `stream: False` for every call
  made inside the pipeline, consistently across three separate runs. The reject-then-no-memo path is
  verified by construction rather than an independent live test this time: the adversarial prompt that
  reliably triggered escalation in Part B didn't reproduce a sub-threshold score on repeat attempts
  today (real LLM output variance), and `save_memo`'s gating instruction is identical to
  `remember_finding`'s, already proven correctly followed on rejection. 10 new pytest tests, 55 total
  project-wide.
- **Stage 6 (2026-08-02)** — Evaluation harness, live baseline, and ablation study. **This is the
  portfolio centerpiece.** `argus/eval/custom_metrics.py` holds 4 deterministic `adk eval` metrics
  (spec EV1) — tool-trajectory, numeric fidelity, independent groundedness, contradiction correctness
  — all pure re-derivations from the same mock data the agents themselves use, never an LLM judge:
  this project's "numbers come from code, not vibes" rule applied to the eval harness itself.
  `eval/argus_evalset.json` covers 8 golden scenarios (not the spec's literal ≥20, deliberately —
  a single broad scenario alone is ~16-20 live API calls against a 15-req/min free-tier ceiling this
  whole project has lived under). CI (`.github/workflows/tests.yml`) runs the pytest suite only, never
  `adk eval`, so it stays free and fast on every push.

  Running the first live batch immediately paid for itself: it surfaced 6 real bugs in the metrics'
  own number-parsing logic before they ever reached the more expensive broad scenarios — a comma in
  `"$2,850 million"` silently splitting into two claims, a `$` between a minus sign and its digits
  flipping a real net loss positive, a bare year range (`"2021-2025"`) misparsing the same way, a
  memo-filename date leaking through the date-stripper because of a boundary-regex edge case, a
  hyphenated negation phrase (`"No headline-to-filing contradictions"`) missing a same-class match, and
  a missing 4th grounding source (computed quant metrics, not just raw filings/market/sentiment) that
  meant every legitimate computed claim in a broad response scored "unfounded" even in a clean run.
  Every one fixed with regression pytest coverage (90 tests project-wide) and re-verified live. Two
  more things the harness surfaced are real findings, not bugs: `independent_groundedness` correctly
  refuses to credit a unit-converted restatement ("$2.85 billion" for a source's "2850 million") as
  grounded — a genuine, literal-matching characteristic shared with production `claim_matcher.py`, not
  something to loosen without reopening the fabricated-year-adjacency hole `_MATCH_TOLERANCE` already
  guards against; and the Orchestrator's bounded re-planning is **not fully reliable** — across 2 live
  runs of the missing-company scenario it once correctly retried with the closest available match and
  once gave up immediately with a false "no alternative available" — a disclosed, live-verified gap for
  a follow-up pass, not fixed this session.

  With the harness itself trustworthy, the live baseline: the 4 narrow scenarios and the 2 clean broad
  scenarios pass every applicable deterministic metric; the adversarial-injection scenario correctly
  refuses to incorporate a fabricated 2026 revenue figure injected in the prompt; the missing-company
  retry scenario is the one genuinely unreliable case, matching the finding above.

  The ablation study (`argus/eval/ablation.py`) builds 3 pipeline variants — baseline, one with
  the critic/refiner loop removed, one with `quant_agent` (code execution) removed entirely — each
  independently cloned via `BaseAgent.clone()` (confirmed against ADK source: a sub-agent can only
  have one parent, so the real pipeline's singletons can't be reused directly in a second tree) and
  run through a real `Runner`, not a simulation. Full results table in the main README.
- **Portfolio-readiness pass (2026-08-12)** — a full external-reviewer audit of the repo (as it would
  look to someone who only ever sees the 48 tracked files, never this log) surfaced real problems that
  the project's own claim-checking discipline had missed on itself:

  The most serious: `README.md` stated the bounded-retry feature as reliably working, while this very
  log already documented it as ~50% reliable across live testing — a genuine contradiction between two
  public files, caught only by deliberately cross-reading them against each other. Fixed by making the
  README state the same honest, disclosed status this log already had; the disclosure is the strength,
  hiding it was the liability. The ablation study's headline result also had zero committed evidence —
  a strong finding resting entirely on prose. Fixed by committing the real completed run
  (`eval/results/ablation_2026-08-02.md` and its raw JSON), so anyone can check it, not just trust it.
  The documented `adk eval` command also couldn't run from a fresh clone (`requirements.txt` didn't
  include the extras `google-adk[eval]` needs), and every setup command was Windows-only despite CI
  proving the project runs fine on Linux — both fixed (`requirements-eval.txt`; cross-platform
  `python -m` invocations throughout).

  A second class of finding: 21 tracked Python files cited `CLAUDE.md`, `PLAN.md`, and internal spec
  IDs (`spec P5`, `principle 18`, `agent #14`, `SG1-3`, `OR2`, `SM2`, `EV1`, …) that are all
  git-ignored — every one a dead pointer to an external reader, and collectively they made the repo
  read like a private workspace carved down for publication rather than a standalone project. Fixed by
  rewriting every citation as inline plain-language rationale instead of a pointer — the reasoning was
  almost always already spelled out in the surrounding prose, so nothing substantive was lost. The
  design rationale that genuinely had nowhere public to live now does: a new `DECISIONS.md`.

  A few smaller stale artifacts, all now fixed: the CI workflow's own comment still said "55 tests"
  and pointed at a README section that no longer exists; `argus/config.py` shipped a `TEMPORARY:`
  note-to-self from Stage 1 about switching models "once the pipeline is built" — it was built, and
  the real decision (keep Flash-Lite, re-confirmed at Stage 6 for stated reasons) was only ever written
  down in the gitignored `CLAUDE.md`; `tests/test_memory.py` named a function
  (`persist_finding_to_memory`) that doesn't exist — it's `remember_finding`; and one genuinely dead
  tool (`get_company_snapshot`, a Stage 0 leftover registered on no agent) was removed.

  Also added: a `LICENSE` (MIT) and `argus/.env.example`, both simply absent before. And 10 new tests
  closing the sharpest fair criticism from the audit — 90 tests and not one touched an agent, a
  callback, or the orchestrator, the framework code that's the actual subject of the project. Added
  direct tests for `check_gather_status` (the deterministic retry gate) and `enforce_tool_budget`
  (the per-session guardrail, including its exact boundary condition), both plain dict-and-counter
  logic testable with a minimal stand-in `ToolContext` and no live ADK runtime. 100 tests total,
  zero regressions, all found and fixed before any of it went out the door rather than after.

- **Multi-model routing, and two external-reviewer passes (2026-09-03)** — every agent had run on a
  single pinned model (`gemini-3.5-flash-lite`) since Stage 1. Free-tier quota is tracked per model,
  not shared, so splitting agents across several models raises the effective ceiling for one full
  pipeline run instead of exhausting one 15-req/min bucket. Landing the final lineup took three real,
  live failures first, each one instructive in its own way:

  `gemini-3.8-flash` (quant_agent's first assignment) returned a genuine `503 UNAVAILABLE` ("high
  demand") on its very first real pipeline call — after passing an isolated single-call smoke test
  cleanly. `gemini-3.7-flash` (critic_agent, next attempt) did the same. `gemini-3.6-flash`
  (verifier_agent, third attempt) did too. All three were confirmed real, valid model IDs; none of
  them were confirmed *reliable* by a one-off "say OK" test — the actual lesson, not obvious going
  in: an isolated test proves an ID exists, not that it holds up under real load, and that gap bit
  the same way three separate times before it was treated as a pattern instead of a fluke. Separately,
  hit-and-trial testing confirmed `gemini-3.0-flash`, `gemini-3-flash`, and the exact spelling
  `gemini-2.5-flash-lite` are all `404 NOT_FOUND` — real dead IDs, not just unreliable ones, caught
  with a direct `google.genai.Client` call before ever being wired into an agent again. Final,
  live-verified lineup: `gemini-3.5-flash-lite` for `orchestrator_agent`, `quant_agent`, and the 3
  gatherers (the highest-frequency / most load-bearing agents stay on the one model with zero real
  failures across the whole project); `gemini-3.1-flash-lite` shared by `critic_agent`, `refiner_agent`,
  and `verifier_agent` (the high-frequency loop needs the large 500-req/day pool, and verifier only
  adds one call/run so it barely dents it); `gemini-3.5-flash` for `synthesis_agent`; `gemini-2.5-flash`
  for `retrieval_agent` (lowest priority, rarely invoked, the safest place to run a freshly-validated
  model).

  A separate, purely mechanical bug ate an entire round of "why didn't my fix work" debugging: editing
  the agent files does not affect an `adk web` process that's already running — Python doesn't
  re-import already-loaded modules mid-process, so every test after an edit was silently still running
  the *previous* model assignment until the server was explicitly restarted. And `meta.tool_call_count`
  (the guardrail budget from Stage 3) is cumulative for the whole Dev UI session, not per-message — a
  long day of testing in one continuously-reused session hit the 20-call cap mid-run, which looked like
  a new bug but was the existing guardrail working exactly as designed.

  **First external-review pass.** A second reviewer given the code cold, findings verified against the
  real code before being trusted and reproduced again after each fix (not taken on the review's word
  either way): the Groundedness Verifier let the calling LLM enumerate which numbers to check, so an
  empty or under-enumerated list scored a vacuous, perfect `1.0` — fixed by having `check_grounding`
  read `draft.thesis` and extract its own numbers deterministically, removing the model's discretion
  over what gets checked at all. `save_memo` and `remember_finding` only refused to run after a
  rejection because their own docstrings said so — reproduced live: both ran anyway when called right
  after a real `meta.human.decision == "reject"`. Both now fail closed in code unless the decision is
  exactly `"approve"` or `"auto_approved"`. `check_gather_status` read completely missing
  `evidence.*` state (e.g. the model never actually calling `full_analysis_pipeline`) as
  `gather_ok: True`, since an empty `{}` has no `"error"` key — now checked explicitly.
  The Orchestrator's one bounded retry picked its "closest match" company name via the model's own
  judgment, with nothing guaranteeing it matched a real string-distance comparison — `check_gather_status`
  now computes `suggested_retry_name` itself via `difflib` before the model ever sees the choice.
  `recompute_quant_metrics`'s CAGR used row count as the elapsed-years denominator, silently correct
  only because the mock data has no year gaps. Ruff and Pyright were added to CI in the same pass (a
  new, zero-API-cost lint job) and found three more real issues on the first run: a missing
  venv-relative re-export in `argus/__init__.py`, an under-typed recursive function in the
  injection-screening guardrail (`_flag_injection`, now given proper `@overload`s), and an
  un-null-checked `Session` lookup in the ablation script.

  **Second external-review pass**, the same day, found the sharpest bug of the whole project so far:
  the Verifier fix above normalized the *draft's* text before extracting its claims, but
  `compute_grounding` still extracted "known" numbers from raw, un-normalized *source* text. An
  un-normalized ISO date's day component is itself a valid-looking small number — reproduced exactly
  as reported: a fabricated `"-$1M loss"` claim (normalizes to `-1.0`) matched the `"-01"` inside an
  unrelated source date (`"2025-01-01"`) and scored a false, perfect `1.0`. Fixed inside
  `compute_grounding` itself, which also silently closed the identical gap in the eval harness's own
  `independent_groundedness` metric — same function, same bug, one fix, re-verified afterward against
  real Globex evidence to confirm no real number stopped grounding correctly. Regression tests were
  added for the memo/memory rejection gates from the first pass — confirmed live at the time, but
  genuinely untested until now, using `asyncio.run()` directly rather than adding a `pytest-asyncio`
  dependency for 7 tests. `meta.groundedness` is now documented explicitly as *numeric* groundedness —
  a prose-only fabricated claim with no numbers in it scores a vacuous `1.0` by design, a real scope
  limit worth stating rather than leaving implicit. `synthesis_agent`'s instruction now defaults to
  describing a reconciliation finding as a tension between narrative framing and real profitability,
  reserving "contradiction" for when the evidence actually supports that stronger word.

  Explicitly not done, and why, rather than silently deferred: deterministic retry *execution* (not
  just name selection) needs a real code-driven controller in place of a model-decided tool call — a
  genuine architecture change, scoped separately rather than rushed. Durable local memory was
  investigated, not built: confirmed against ADK's own source that the only two non-in-memory
  `MemoryService` implementations it ships are both Vertex-AI-backed, which conflicts with this
  project's own locked-in "fully local" decision — a real durable option means writing a new
  SQLite-backed service from scratch, a separate task. Re-running the eval/ablation harness on the
  current commit remains deferred by explicit choice, given the real API cost.

- **Naming fix (2026-09-03)** — the cross-run memory feature was renamed from "long-term memory" to
  **"runtime memory"** everywhere in the code and docs, following the same external review. It uses
  ADK's `InMemoryMemoryService`: a second run on an already-seen entity starts warm, but only within
  the same running process — the store is cleared on restart. "Long-term" overclaimed that; "session
  memory" (the other name the review offered) is also wrong, because the feature's whole purpose is to
  persist *across* ADK sessions — what it doesn't survive is a process restart. Docstring/comment
  change only, no behavior change: `argus/tools/memory.py` module + tool docstrings,
  `orchestrator.py` and `pipeline.py` comments, plus a new "known gap" bullet in the README and the
  standing-decision paragraph in DECISIONS.md. The durable SQLite-backed `MemoryService` stays the
  real deferred item; this is the honest interim name for what's actually there.

- **Wording / disclosure pass (2026-09-03)** — three small accuracy fixes closing out the external
  review before the portfolio draft:
  - The pipeline's own tool description said it "verifies every claim is grounded" — but the
    deterministic verifier only checks numeric literals. Changed to "reviews its claims and verifies
    every numeric claim against the evidence" in `pipeline.py`; the parallel "check every claim"
    phrasing in `critic.py`'s instruction and the "verify all claims" shorthand in this log were
    softened to match what each step actually does (the critic is an LLM red-team, so it *reviews*
    every claim; only the numeric check is mechanical).
  - The memo artifact's displayed row `Cross-source contradictions found` is now
    `Narrative-profitability tensions found`, matching `synthesis_agent`'s own "tension between the
    narrative framing and the real profitability" wording. Display label only — the `contradictions`
    state key and `find_contradictions` function keep their names.
  - The value-only nature of groundedness matching is now a stated known limitation in the README
    (and DECISIONS.md): a number can score as grounded merely because an unrelated figure of the
    same value exists elsewhere in the evidence. Distinct from the ISO-date-fragment bug (fixed);
    closing it needs structured metric/unit/year matching, deferred rather than rushed. The score is
    now framed as a strong check on numeric *fabrication*, not proof of correct-context use.

  113 tests total (up from 100), `ruff check` clean, `pyright` clean — both added to CI in this pass,
  not just run locally.

- **State-lifecycle pass (2026-09-03)** — two more real bugs from the external review, both about
  state surviving longer than it should across turns in one `adk web` session:
  - **A failed gate could inherit a prior approval.** `evaluate_gate` (`hitl.py`) wrote `meta.human`
    only on auto-approval. So if an earlier run in the session auto-approved, then a later run failed
    the gate, the stale `"auto_approved"` stayed in state — and a mistaken later `remember_finding` /
    `save_memo` (or a resumed turn after a quota 429 on the wrap-up, a documented real scenario) would
    pass their code-level approval checks and persist an unreviewed result. Fixed: every
    non-auto-approved outcome now overwrites `meta.human` with `{"decision": "pending_review"}`, a
    value no persistence tool accepts. `record_human_decision` still writes the real verdict once the
    human replies, so `pending_review` only exists during the suspension window. Regression test in
    `test_hitl.py`: start with an old approval, fail the gate, assert `meta.human` is `pending_review`
    and that both `remember_finding` and `save_memo` refuse.
  - **Gather-status accepted stale evidence as fresh.** `check_gather_status` (`orchestrator.py`) only
    checked that the three `evidence.*` keys were populated and error-free. In a multi-turn session,
    evidence from a prior successful broad request stays exactly that. If the model skipped the
    pipeline on the next broad request but still called the status tool, it returned `gather_ok=True`
    and could release stale analysis. Fixed with a per-request stamp: `_stamp_request_id` (a
    `before_agent_callback` on the Orchestrator — runs every turn, before the model picks a route, so
    it can't be skipped) writes a fresh `meta.request_id`; each gatherer's new `after_agent_callback`
    copies that id next to its evidence (`gather.py`, unique key per gatherer — the parallel
    anti-collision rule); `check_gather_status` now reports `gather_ok` only when all three stamps
    match the current `meta.request_id`, else returns `stale=true` and the instruction tells the model
    to actually run the pipeline. `meta.request_id` crosses into the pipeline's `AgentTool` run via
    the state snapshot `agent_tool.py` takes at call time (verified by reading the source, lines
    277-286), and the stamps cross back out via `state_delta` — the same boundary behaviour
    `output_key` already relies on. Regression tests in `test_orchestrator.py`: fully stale, partially
    stale, missing `meta.request_id`, and a same-turn gather error still routing to the retry path
    (not the stale path). Multi-turn end-to-end still wants a live `adk web` pass — noted in the
    README's known gaps, same treatment `check_gather_status` has always had.

  Also this pass: README corrected (`100 pytest tests` → 120, and the project tree now says CI runs
  Ruff + Pyright, not "tests only"); the blanket "everything is verified live" line now names the two
  paths that aren't (`redirect`, retry reliability); and `analysis.quant` being both a grounding
  source and the narrated value is now disclosed as the largest remaining design improvement.
  120 tests total (up from 113), `ruff check` clean, `pyright` clean, `compileall` clean.

- **Transient-503 hardening (2026-09-03)** — first live `adk web` test after the state-lifecycle pass
  hit a `google.genai.errors.ServerError: 503 UNAVAILABLE` ("this model is currently experiencing high
  demand") on one of the three parallel gatherers, which took down the whole run: ADK's `ParallelAgent`
  uses an `asyncio.TaskGroup`, so the first sub-agent's `ServerError` cancels its siblings and
  propagates as an `ExceptionGroup` to the root — the Dev UI just shows "an error occurred". Not a code
  bug (transient upstream overload), but the pipeline had no retry: `google-genai` does **not** retry
  unless `retry_options` is set explicitly, and it wasn't. Fixed in `config.py` — the `MODEL_*` names
  are now configured `Gemini` instances (built from the same pinned IDs, all comments kept) carrying
  `types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=30.0, exp_base=2.0)`, so a call gets
  ~2s/4s/8s/16s backoff over google-genai's default retryable set (408, 429, 5xx). Agent files are
  unchanged — `model=MODEL_FLASH` still works, ADK accepts a `BaseLlm` instance anywhere it takes an
  id. **Verified live the same session:** a fresh narrow run hit another 503 mid-request, logged
  `Retrying ... in 2.36 seconds as it raised ServerError: 503`, recovered on the retry, and returned
  the correct answer (`$89.60 as of December 31, 2025`) instead of crashing. Does not mask a sustained
  `RESOURCE_EXHAUSTED` — that still surfaces once ~30s of backoff is exhausted. 120 tests still green,
  `ruff` / `pyright` clean.

## Reproducing the results

Both commands below need the eval-only extras, which are deliberately not in the base
`requirements.txt` (they pull in pandas/nltk/rouge-score, only needed for this):

```bash
pip install -r requirements-eval.txt
```

Then, needs `GOOGLE_API_KEY` set in `argus/.env`. Run from the repo root in a POSIX shell (bash/zsh —
Git Bash on Windows, or a native shell on macOS/Linux):

```bash
# Golden evalset (8 scenarios)
NLTK_DISABLE_IMPORT_SECURITY=1 PYTHONPATH="$(pwd)" python -m google.adk.cli eval argus \
  eval/argus_evalset.json --config_file_path eval/test_config.json --print_detailed_results

# Ablation experiment (3 pipeline variants, 6 live runs)
PYTHONPATH="$(pwd)" python -m argus.eval.ablation
```

(`NLTK_DISABLE_IMPORT_SECURITY=1` works around a false-positive in nltk's own CWD-import guard,
triggered here only because `.venv/` happens to live inside the project directory — see
`argus/eval/custom_metrics.py`'s module docstring. `python -m google.adk.cli` is the `adk` CLI
invoked through the venv's active Python, which works identically whether that Python is on
`.venv/Scripts/` (Windows) or `.venv/bin/` (macOS/Linux) — no need to hardcode either path.)

Ablation progress saves to `.ablation_checkpoint.json` at the repo root (git-ignored, since it's a
resumable working file — see `eval/results/` for a committed snapshot of a completed run), so a
rate-limit hit mid-run resumes instead of restarting.
