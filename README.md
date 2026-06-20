# 📊 Financial Analyst Agent

> Physics-informed quantitative financial analysis powered by a ReAct agent — combining Hurst exponent regime detection, signal-quality validation (IC/IR), and classic risk analytics with LLM reasoning. Built as both a portfolio project and a usable analysis tool.

[![LangSmith Trace](https://img.shields.io/badge/LangSmith-View%20Trace-1C3C3C?logo=langchain)](#observability)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](#tech-stack)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](#running-with-docker)

---

## Why this project

Most "AI stock analyst" demos are a thin LLM wrapper around an API call. This one is different in two ways:

1. **The quant layer is real.** Hurst exponent regime detection, Information Coefficient (signal quality), Information Ratio (signal consistency), and a full risk-analytics suite (Sharpe, Sortino, Calmar, VaR/CVaR, Ulcer Index, rolling correlation, CAPM alpha/beta) are computed with actual time-series math — not LLM-generated numbers.
2. **The engineering is production-shaped.** LangGraph ReAct orchestration, explicit LangSmith tracing (not env-var dependent), async SQLAlchemy persistence, a FastAPI backend, and a Docker image that's been built, run, and verified end-to-end — not just "works on my notebook."

**Positioning:** Explainable Quant Analytics + Agent Orchestration. This project intentionally does *not* compete on model complexity (no Monte Carlo, no HMM regime models, no opaque factor weights) — every number the agent states is traceable to a deterministic calculation, and every design choice that was *not* made is documented with a reason.

---

## What it can do

| Use case | Example query | What happens |
|---|---|---|
| **Single-stock analysis** | "NVDA วิเคราะห์ให้หน่อย" | Fetches live price, fundamentals, and computes Hurst regime + signal quality |
| **Pre-purchase portfolio risk** | "วิเคราะห์ risk ของ portfolio: {NVDA: 5000, AMD: 3000}" | Full risk report: Sharpe/Sortino/Calmar, VaR/CVaR, Ulcer Index, rolling correlation, alpha/beta vs SPY |
| **Portfolio tracking** | "ติดตาม portfolio id: my_portfolio" | Live unrealized P&L, current weights, position-level detail from persisted holdings |
| **Market news & context** | "ทำไม TSLA ร่วงวันนี้" | Grounded web search with citations — never hallucinates a reason |

The agent communicates in Thai mixed with English technical terms, and is explicit about what it doesn't know rather than estimating from memory.

---

## Architecture

```
User query
    ↓
run_financial_agent() ──@traceable──→ LangSmith (returns trace_id)
    ↓
ReAct Agent (LangGraph + Groq gpt-oss-120b)
    ├── get_stock_price            → live price, 52W range, P/E, market cap
    ├── get_stock_financials       → revenue, margins, growth, EPS, D/E
    ├── get_hurst_exponent         → R/S analysis + Rolling Hurst + IC + IR
    ├── analyze_portfolio_risk     → Sharpe/Sortino/Calmar/VaR/CVaR/Ulcer/Alpha-Beta
    ├── track_portfolio            → SQLite-backed live P&L tracking
    └── search_market_news         → OpenAI gpt-4o-mini + grounded web search
        ↓
FastAPI (4 endpoints) ──→ SQLite (async) for portfolio persistence
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

**What's deliberately out of scope:** walk-forward backtesting, HMM regime models, and hardcoded factor weights — each requires either a position-sizing/execution model or per-factor IC validation that wasn't justified for a v1 scope. See [Design Decisions](#design-decisions) below.

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph `create_react_agent` (ReAct) |
| LLM (reasoning) | Groq `openai/gpt-oss-120b` |
| LLM (news search) | OpenAI `gpt-4o-mini` + `web_search_preview` |
| Market data | yfinance, behind a `DataProvider` protocol (swappable) |
| Observability | LangSmith — explicit client binding, not env-var dependent |
| Backend | FastAPI + Pydantic v2 |
| Persistence | SQLite + SQLAlchemy async (`aiosqlite`) |
| Package management | `uv` |
| Containerization | Docker (multi-stage, `python:3.11-slim`) |

---

## Observability

Every agent run returns a `trace_id` (the LangSmith `run_id`), giving 1:1 traceability between an API response and its full execution trace — tool calls, intermediate reasoning, and timing.

> Public trace links require a LangSmith `share_run()` call per trace and are not committed to the repo (they embed a workspace ID). Run the agent locally and check the `trace_id` field in any response, or see the project owner for example traces.

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

**Why is there a known routing quirk with "P/E and profit margin" queries?**
Asking for AMD's P/E and profit margin causes the agent to call both `get_stock_financials` *and* `get_stock_price`, even though only the former is needed — P/E is semantically tied to price, and docstring-level negative-routing instructions can't fully separate the two for an LLM router. It's benign (the price tool is fast and free) and is left as-is for v1; a v2 fix would use explicit `StateGraph` conditional routing instead of relying on prompt engineering alone.

**Why no backtesting or factor models?**
Walk-forward backtesting requires a full signal-to-execution pipeline (position sizing, transaction costs, rebalancing rules) that doesn't exist yet — a backtest without those is not a validation, it's noise dressed as a result. Hardcoded factor weights (e.g. `0.3 × momentum + 0.3 × Hurst`) were considered and rejected for the same reason: without per-factor IC validation, fixed weights are narrative quant, not real signal combination.

**Why is multi-asset support (forex/crypto/options) out of scope?**
`yfinance` technically returns spot prices for forex and crypto tickers, so the Hurst/IC/IR tools would likely run without modification. But `get_stock_financials` (revenue, margins, EPS) is equity-specific, the Calmar/IC benchmarks referenced are validated against equity data only, and options analytics would require an entirely different data source plus Black-Scholes/Greeks — a scope expansion on the same order as the factor-engine work above. This is a deliberate scope boundary, not an oversight.

---

## Project structure

```
src/
├── config.py              # env loading, LangSmith client/tracer (explicit binding)
├── tools/
│   ├── data_provider.py   # DataProvider protocol + YFinanceProvider
│   ├── price.py / financials.py / hurst.py
│   ├── portfolio_risk.py / portfolio_track.py
│   └── news.py
├── database/
│   ├── models.py          # Portfolio, Position (SQLAlchemy)
│   └── session.py
├── agent/
│   ├── prompts.py         # system prompt
│   └── core.py             # build_agent(), run_financial_agent()
└── api/
    ├── schemas.py
    └── routes.py
main.py                     # app assembly, uvicorn entry point
Dockerfile
notebooks/
└── financial_analyst_agent.ipynb   # original algorithm development environment
```

The notebook remains in the repo as the algorithm-development reference — all production logic has been migrated to `src/` and is the source of truth for the running system.

---

## Disclaimer

This tool provides quantitative analysis, not investment advice. It does not give price targets, entry/exit points, or stop-loss levels. All metrics are computed from historical data and carry no guarantee about future performance.