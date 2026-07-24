# ARGUS

Autonomous Reasoning & Grounded Understanding System — a hierarchical multi-agent due-diligence
system built on Google's Agent Development Kit (ADK). See [`ARGUS_ADK_Requirements.md`](ARGUS_ADK_Requirements.md)
for the full spec and [`PLAN.md`](PLAN.md) for the staged build plan.

## Running it

```bash
.venv/Scripts/adk.exe web --port 8000
```

Open http://localhost:8000, pick the `argus` app, and chat with it.

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
