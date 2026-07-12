# 📊 Financial Analyst Agent

> Explainable quantitative financial analysis powered by a custom LangGraph agent — combining Hurst exponent regime detection, signal-quality validation (IC/IR), and classic risk analytics with LLM reasoning. Built as both a portfolio project and a usable analysis tool.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](#tech-stack)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](#running-with-docker)
[![Tests](https://img.shields.io/badge/routing%20tests-13%2F13%20passing-brightgreen)](#running-tests)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](#running-locally)

---

## Why this project

Most "AI stock analyst" demos are a thin LLM wrapper around an API call. This one is different in three ways:

1. **The quant layer is real.** Hurst exponent regime detection, Information Coefficient (signal quality), Information Ratio (signal consistency), Risk Contribution to Variance, and a full risk-analytics suite (Sharpe, Sortino, Calmar, VaR/CVaR, Ulcer Index, rolling correlation, CAPM alpha/beta) are computed with actual time-series math — not LLM-generated numbers.
2. **The engineering is production-shaped.** A custom LangGraph `StateGraph` (planner → agent → tools) with deterministic routing, explicit LangSmith tracing (not env-var dependent), async SQLAlchemy persistence, a FastAPI backend, a Streamlit UI, and a Docker image that's been built, run, and verified end-to-end — not just "works on my notebook."
3. **The guardrails are adversarially tested, not assumed.** Every numeric claim the agent makes is backed by a tool result — but getting an LLM to *describe* those numbers correctly is its own engineering problem. This project caught and fixed three separate ways the agent misrepresented correct tool output (see [Design Decisions](#design-decisions)): claiming a hypothetical portfolio's losses were the user's actual losses, inventing stop-loss suggestions despite an explicit rule against it, and describing a correlation coefficient as a literal proportion of time two assets move together. Each fix was verified against the adversarial query that caused it, then checked against the full regression suite.

**Positioning:** Explainable Quant Analytics + Agent Orchestration. This project intentionally does *not* compete on model complexity (no Monte Carlo, no HMM regime models, no opaque factor weights) — every number the agent states is traceable to a deterministic calculation, and every design choice that was *not* made is documented with a reason.

---

## What it can do

| Use case | Example query | What happens |
|---|---|---|
| **Single-stock analysis** | "NVDA วิเคราะห์ให้หน่อย" | Fetches live price, fundamentals, and computes Hurst regime + signal quality |
| **Pre-purchase portfolio risk** (what-if) | "วิเคราะห์ risk ของ portfolio: {NVDA: 5000, AMD: 3000}" | Full risk report: Sharpe/Sortino/Calmar, VaR/CVaR, Ulcer Index, rolling correlation, alpha/beta vs SPY, and Risk Contribution to Variance — framed as hypothetical, never as the user's actual losses |
| **Portfolio tracking** (actual holdings) | "ติดตาม portfolio id: my_portfolio" | Live unrealized P&L, current weights, correlation, and Risk Contribution from persisted holdings with real cost basis |
| **Market news & context** | "ทำไม TSLA ร่วงวันนี้" | Grounded web search with citations — never hallucinates a reason |

The agent communicates in Thai mixed with English technical terms, and is explicit about what it doesn't know rather than estimating from memory. A Streamlit UI (3 tabs, mirroring the use cases above) sits on top of the same API for anyone who'd rather click than write JSON — see [Running locally](#running-locally).

---

## Architecture

```
User query
    ↓
run_financial_agent() ──@traceable──→ LangSmith (returns trace_id)
    ↓
StateGraph Agent: planner → agent → tools (LangGraph + Groq gpt-oss-120b)
    │   planner classifies the required tool subset; the agent node then
    │   reconciles the model's tool calls to that plan (deterministic routing)
    ├── get_stock_price            → live price, 52W range, P/E, market cap
    ├── get_stock_financials       → revenue, margins, growth, EPS, D/E
    ├── get_hurst_exponent         → R/S analysis + Rolling Hurst + IC + IR
    ├── analyze_portfolio_risk     → Sharpe/Sortino/Calmar/VaR/CVaR/Ulcer/Alpha-Beta + Risk Contribution
    ├── track_portfolio            → SQLite-backed live P&L + correlation + Risk Contribution
    └── search_market_news         → OpenAI gpt-4o-mini + grounded web search
        ↓
FastAPI (5 endpoints) ──→ SQLite (async) for portfolio persistence
    ↓
Streamlit UI (3 tabs) — calls FastAPI over HTTP, no direct agent access
```

All tool calls are traced end-to-end in LangSmith. Every numeric claim in an agent response is backed by a tool result — the system prompt explicitly forbids stating a number that didn't come from a tool call.

---

## The quant layer, briefly

**Hurst Exponent (R/S analysis)** classifies the current price regime:
- `H > 0.55` → Trending · `H < 0.45` → Mean-reverting · else → Random walk
- **Rolling Hurst** (126-day window) shows whether the regime is strengthening or fading — a single point can't tell you that.

**Information Coefficient (IC)** — Spearman correlation between rolling Hurst and forward 5-day returns — measures whether the regime signal actually has predictive power, not just a plausible-looking number.

**Information Ratio (IR)** = mean(IC) / std(IC) — because a single IC reading can be noise. This is the metric that catches a "strong-looking" IC that isn't actually consistent over time.

**Portfolio risk** uses historical simulation (not parametric/normal-distribution assumptions): Sharpe, Sortino, Calmar, VaR/CVaR at 95%, Max Drawdown, Ulcer Index (drawdown *pain*, not just depth), drawdown duration, rolling 60-day correlation, and CAPM alpha/beta against SPY.

**Risk Contribution to Variance** decomposes portfolio risk by ticker: `MCR_i = w_i × (Σw)_i / (w^T Σw)`, which sums to exactly 100% by construction. This is the metric that tells you a stock contributing 38% of your capital can be responsible for 54% of your portfolio's variance — because contribution depends on volatility and correlation, not just allocation size. Available for both the what-if form (investment-amount weighted) and the tracking view (current-market-value weighted).

**What's deliberately out of scope:** walk-forward backtesting, HMM regime models, hardcoded factor weights, and a standalone "diversification score" — each either requires a position-sizing/execution model, per-factor IC validation, or a metric the tools don't actually compute. See [Design Decisions](#design-decisions) below.

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | Custom LangGraph `StateGraph` (planner → agent → tools) |
| LLM (reasoning) | Groq `openai/gpt-oss-120b` |
| LLM (news search) | OpenAI `gpt-4o-mini` + `web_search_preview` |
| Market data | yfinance, behind a `DataProvider` protocol (swappable) |
| Observability | LangSmith — explicit client binding, not env-var dependent |
| Backend | FastAPI + Pydantic v2 |
| UI | Streamlit — calls FastAPI over HTTP, no direct agent access |
| Persistence | SQLite + SQLAlchemy async (`aiosqlite`) |
| Package management | `uv` |
| Containerization | Docker (multi-stage, `python:3.11-slim`) |

---

## Observability

Every agent run returns a `trace_id` (the LangSmith `run_id`), giving 1:1 traceability between an API response and its full execution trace — tool calls, intermediate reasoning, and timing.

```json
{"response": "...", "trace_id": "019ee376-8c81-7d53-9233-f594b7955858"}
```

**Note on public trace links:** LangSmith's `share_run()` generates a public URL, but these expire after a limited window — a link committed to this README today would be a dead 404 within weeks. Rather than maintain a stale link, the project demonstrates observability via the `trace_id` returned on every request: run the agent locally (or via the live container) and `trace_id` will be a real, freshly-generated LangSmith run that can be shared on demand with `ls_client.share_run(run_id)`.

---

## Running locally

### Prerequisites
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- API keys: Groq, OpenAI, LangSmith (`.env` — see `.env.example`)

```bash
uv sync
uv run uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger UI — every endpoint can be tested directly from the browser, including Thai-language queries (Swagger sends UTF-8 correctly; some Windows terminal clients do not).

For a point-and-click interface instead of raw JSON, run the Streamlit UI in a second terminal (FastAPI must already be running):

```bash
uv run streamlit run streamlit_app.py
```

This opens three tabs mirroring the use cases above — general questions, what-if portfolio risk, and portfolio tracking — each with quick-question buttons and a two-tier response display (plain-language summary up front, full technical breakdown in an expander) aimed at users who don't know what a Sharpe ratio is.

### API endpoints

```
GET  /health
POST /analyze/stock       → {query, response, ticker, trace_id}
POST /analyze/portfolio   → {portfolio, query, response, trace_id}
                             body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}      → {portfolio_id, response, trace_id}
```

---

## Running with Docker

```bash
docker build -t financial-analyst-agent .
docker run -d -p 8000:8000 --env-file .env \
  -v ${PWD}/data:/app/data \
  --name fa-agent financial-analyst-agent

curl http://localhost:8000/health
# {"status":"ok"}
```

The volume mount persists the SQLite database across container restarts — without it, tracked portfolios are lost when the container stops.

Verified end-to-end: container build, startup (no import/DB errors), and a live `/analyze/stock` call with a real LangSmith `trace_id` returned — see [Design Decisions](#design-decisions) for what specifically was tested.

---

## Design decisions

A few choices are deliberate and worth explaining if you're reviewing this project:

**Why explicit LangSmith binding instead of environment variables?**
`langsmith.utils.get_env_var` is wrapped in `lru_cache`, which caches its value on first import. If `langsmith` is imported before `.env` is loaded, tracing silently stays disabled for the entire session — with no error. Binding the client explicitly (`client=ls_client` in every `@traceable`) avoids this failure mode entirely. (One environment variable, `LANGCHAIN_TRACING_V2`, is a genuine exception — it's a SDK-level on/off switch with no parameter equivalent, so it's set via `os.environ.setdefault()` rather than treated as a credential.)

**Why does the database only store `ticker`, `shares`, and `avg_cost`?**
Everything else — current price, market value, unrealized P&L, current weights — is derived live from market data on every request. This avoids any possibility of stale cached prices and removes the need for a sync job entirely.

**Why did the "P/E and profit margin" routing quirk get fixed with a `StateGraph`, not a better prompt?**
Under the old ReAct agent, asking for AMD's P/E and profit margin made the agent call both `get_stock_financials` *and* `get_stock_price`, even though only the former is needed — P/E is semantically tied to price, and docstring-level negative-routing instructions can't fully separate the two for an LLM router. This was a known v1 limitation. v2 fixes it *structurally*: a dedicated planner node classifies the required tool subset **before** execution, and the agent node reconciles the model's tool calls to that plan — dropping any off-plan call (over-fetch) and synthesizing any planned call the model omitted (under-call). Because routing is enforced on the plan rather than coaxed via prompt wording, the fundamentals-only route now holds deterministically (verified across 5 repeated runs), and the case is no longer an exempted failure in the regression suite.

**Why no backtesting or factor models?**
Walk-forward backtesting requires a full signal-to-execution pipeline (position sizing, transaction costs, rebalancing rules) that doesn't exist yet — a backtest without those is not a validation, it's noise dressed as a result. Hardcoded factor weights (e.g. `0.3 × momentum + 0.3 × Hurst`) were considered and rejected for the same reason: without per-factor IC validation, fixed weights are narrative quant, not real signal combination.

**Why is multi-asset support (forex/crypto/options) out of scope?**
`yfinance` technically returns spot prices for forex and crypto tickers, so the Hurst/IC/IR tools would likely run without modification. But `get_stock_financials` (revenue, margins, EPS) is equity-specific, the Calmar/IC benchmarks referenced are validated against equity data only, and options analytics would require an entirely different data source plus Black-Scholes/Greeks — a scope expansion on the same order as the factor-engine work above. This is a deliberate scope boundary, not an oversight.

**Why does the what-if portfolio tool sometimes say "this hypothetical portfolio" instead of "your portfolio"?**
`analyze_portfolio_risk` only ever receives `{ticker: amount}` — no purchase date, no cost basis. Early testing caught the agent saying "your portfolio lost 23%," which is a real misattribution: the tool has no idea when (or whether) the user actually bought anything. The system prompt now explicitly separates two personas — `analyze_portfolio_risk` (UC-2a, hypothetical, must use "if held during the period...") and `track_portfolio` (UC-2b, actual holdings with real `avg_cost`, where "your portfolio" is accurate). The same testing also caught the agent computing its own stress-test numbers (e.g. "a 10% market drop × 2.5 beta = a 25% portfolio drop") — a projection with no tool backing, despite the system prompt's explicit rule against stating ungrounded numbers. Both are now blocked by name in the prompt, with the specific phrasing that triggered them included as negative examples.

**Why does the prompt explain *why* a correlation coefficient can't be described as a percentage of time?**
A keyword blocklist turned out to be fragile: an early guardrail banned describing correlation as "matching half the time," and the model satisfied it literally by switching to "matching in about half of all movements" — same statistical error, different wording. Correlation measures the strength of a linear relationship across an entire dataset; it isn't decomposable into "X% of instances match, Y% don't." The fix that held up under repeat testing wasn't a longer blocklist — it was rewriting the rule to state the underlying reason a proportion-based description is wrong, so the model can't satisfy the letter of the rule while violating its intent. The same round of testing caught the agent inventing a bare "Diversification = 0" figure with no tool backing, alongside the correlation issue — both required the same kind of fix: state what the number actually represents, not just what wording to avoid.

---

### Running tests

```bash
uv run python -m pytest tests/test_routing_regression.py -s -v
```

13 routing regression cases verify that the agent calls the correct tool(s) for a given query — covering under-trigger guards (news queries that must call `search_market_news`), over-trigger guards (numeric queries that must *not* trigger news), multi-tool co-triggering, exact-match regression for the original single-stock use case, and the deterministic portfolio-id pre-route (a saved-portfolio query with risk/stop-loss phrasing must route to `track_portfolio`, not `analyze_portfolio_risk`). All 13 pass — the former P/E over-fetch limitation is now fixed structurally by the `StateGraph` planner (see [Design Decisions](#design-decisions)), and a separate consistency test re-runs the P/E case 5× to confirm the fundamentals-only route is deterministic, not luck.

---

## Project structure

```
src/
├── config.py              # env loading, LangSmith client/tracer (explicit binding)
├── tools/
│   ├── data_provider.py   # DataProvider protocol + YFinanceProvider
│   ├── price.py           # get_stock_price
│   ├── financials.py      # get_stock_financials
│   ├── hurst.py           # get_hurst_exponent (Rolling Hurst + IC + IR)
│   ├── portfolio_risk.py  # analyze_portfolio_risk (+ Risk Contribution)
│   ├── portfolio_track.py # track_portfolio (+ Risk Contribution + correlation)
│   └── news.py            # search_market_news
├── database/
│   ├── models.py          # Portfolio, Position (SQLAlchemy)
│   └── session.py
├── agent/
│   ├── prompts.py         # system prompt + guardrails
│   └── core.py             # agent build + run_financial_agent()
└── api/
    ├── schemas.py
    └── routes.py
main.py                     # app assembly, uvicorn entry point
streamlit_app.py             # 3-tab UI calling the FastAPI backend
Dockerfile
tests/
└── test_routing_regression.py   # 13-case routing regression suite
scripts/
└── test_risk_contribution.py    # standalone Risk Contribution sanity check
notebooks/
└── financial_analyst_agent.ipynb   # original algorithm development environment
```

The notebook remains in the repo as a historical reference for how each metric was first prototyped — it is **not** kept in sync with `src/` after the production migration (several bugfixes, like a timezone-mismatch issue affecting Alpha/Beta, exist only in `src/`). All production logic lives in `src/`, which is the single source of truth for the running system.

---

## Disclaimer

This tool provides quantitative analysis, not investment advice. It does not give price targets, entry/exit points, or stop-loss levels. All metrics are computed from historical data and carry no guarantee about future performance.