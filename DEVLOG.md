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
  parent; long-term memory, the long-running-pause signal, and (confirmed this part) the streaming
  mode do not. That last one explains, precisely, why streaming works for the Orchestrator's own turns
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
  the project's own "verify all claims" discipline had missed on itself:

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
