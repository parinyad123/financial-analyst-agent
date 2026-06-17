# 📊 Financial Analyst Agent — CLAUDE.md

## Project overview

Physics-informed financial analysis ที่ผสม quantitative signals (Hurst exponent, IC score) กับ LLM reasoning ผ่าน ReAct agent โดยมี LangSmith tracing สำหรับ observability และ SQLite สำหรับ portfolio persistence

เป้าหมายสองอย่างพร้อมกัน: portfolio project สำหรับสมัครงาน FinTech และ tool ใช้งานจริงสำหรับนักลงทุน

หลักการ scope: **จบและ demo ได้ สำคัญกว่าทะเยอทะยานแล้วค้าง**

---

## Tech stack

| Layer | Technology | หมายเหตุ |
|---|---|---|
| LLM (dev) | Groq — `openai/gpt-oss-120b` | `reasoning_effort="low"`, tool orchestration เสถียร |
| LLM (prod) | Gemini 2.5 Flash | swap ตอน deploy + Google Search grounding |
| Agent framework | LangGraph `create_react_agent` | prebuilt ReAct |
| Market data | yfinance | real-time price, fundamentals, history |
| Observability | LangSmith | **explicit binding — ไม่พึ่ง env vars** |
| Backend | FastAPI + Pydantic v2 | async endpoints + ngrok สำหรับ Colab |
| Database | SQLite + SQLAlchemy async | `aiosqlite` + `nest_asyncio` สำหรับ Colab |
| Package manager | uv | `pyproject.toml` + `.venv` (Python 3.11) |
| Environment | Google Colab / local (uv) → Docker (planned) | |

### Model decision log
- ❌ Llama 3.3 70B — ภาษาไทยดีกว่า แต่ tool orchestration อ่อนกว่า
- ✅ gpt-oss-120b — reasoning model สำหรับ agentic tasks
- Trade-off: ภาษาไทยอ่อนกว่า Llama 3.3

---

## Architecture

```
User query
    ↓
run_financial_agent() @traceable → return trace_id (run_id)
    ↓
ReAct Agent (LangGraph + gpt-oss-120b)
    ├── get_stock_price            → yfinance (5d)
    ├── get_stock_financials       → yfinance .info
    ├── get_hurst_exponent         → yfinance 1Y + numpy R/S + IC score (merged)
    ├── analyze_portfolio_risk     → yfinance 1Y + numpy/pandas (UC-2a)
    ├── track_portfolio            → SQLite + yfinance 5d (UC-2b)
    └── search_market_news         → Gemini 2.5-flash + Google Search grounding
        ↓
LangSmith (traces ทุก step)
    ↓
FastAPI (4 endpoints) + ngrok tunnel
```

---

## Use cases

### UC-1: วิเคราะห์หุ้นรายตัว ✅
- Input: ticker หรือ natural language query
- Tools: `get_stock_price` + `get_stock_financials` + `get_hurst_exponent`

### UC-2a: วางแผน portfolio ก่อนซื้อ ✅ (amount-based)
- Input: `{ticker: amount}` — จำนวนเงินลงทุน tool คำนวณ weights เอง
- Metrics: Annualized Return/Volatility, Sharpe, Sortino, **Calmar**, VaR 95%, CVaR 95%, Max Drawdown, correlation matrix
- Tools: `analyze_portfolio_risk` (Cell F — canonical version)
- Bug 1 fixed: dict→string guard ใน @tool wrapper

### UC-2b: ติดตาม portfolio หลังซื้อ ✅
- Input: `portfolio_id` string → load จาก SQLite
- Output: unrealized P&L per position, total MV, total P&L, current weights
- Tools: `track_portfolio`

### UC-news: ข่าว + analyst commentary ✅
- Tools: `search_market_news` → Gemini 2.5-flash + Google Search grounding
- Routing: 10/12 ผ่าน (ดู Known limitations)

---

## Database schema (SQLite — Cell B)

```sql
CREATE TABLE portfolios (
    portfolio_id TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE positions (
    position_id  TEXT PRIMARY KEY,
    portfolio_id TEXT REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    shares       REAL NOT NULL,
    avg_cost     REAL NOT NULL,
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);
```

SQLAlchemy async models — swap PostgreSQL ได้โดยเปลี่ยน `DATABASE_URL` เดียว

---

## Notebook structure (สถานะจริงใน v3)

```
Cell 1:   Imports + _get_secret() helper (Colab Secrets / .env fallback) — ไม่มี !pip แล้ว
Cell 2:   LangSmith client + tracer + assert gate
Cell 3:   Tools — price, financials, hurst+IC (merged)
Cell 5:   analyze_portfolio_risk weights-based ← DEPRECATED ⚠️ (header marked)
Cell 6:   get_ic_score ← DEPRECATED ⚠️ (IC merged เข้า Cell 3)
Cell 9:   Gate check Gemini quota (non-fatal try/except)
Cell 10:  search_market_news (Gemini + grounding)
Cell 11:  track_portfolio (SQLite via _load_positions_async)
Cell 13:  Agent setup (Cell 7) — ChatGroq + tools list + SYSTEM_PROMPT
Cell 15:  run_financial_agent (Cell 8) — @traceable entry point
Cell 17–18: Tests UC-1
Cell 20:  Test UC-2a
Cell 24:  Test UC-2b
Cell 26–29: Routing regression tests + consistency checks
Cell 33:  Cell A — pip install sqlalchemy aiosqlite nest_asyncio (Colab only)
Cell 34:  Cell B — SQLAlchemy models + async engine (DB_PATH env-aware)
Cell 35:  Cell C — Seed MOCK_PORTFOLIOS เข้า DB
Cell 36:  Cell D — _load_positions_async() ← swap MOCK → SQLite
Cell 37:  Cell F — analyze_portfolio_risk amount-based ← CANONICAL (Bug 1+2 fixed)
Cell 40:  Cell G — FastAPI app (4 endpoints)
Cell 41:  Cell H — Test endpoints
```

**กฎการรัน:** รันจากบนลงล่าง ถ้า Cell 2 ไม่ผ่านห้ามรันต่อ

---

## Tools (สถานะจริง)

### get_stock_price ✅
```python
@traceable(name="fetch_stock_price", run_type="tool",
           tags=["market-data", "yfinance"], client=ls_client)
```
Period 5d เผื่อ market ปิด
Output: price, change%, 52W range, P/E TTM + Forward, market cap, position in 52W range

### get_stock_financials ✅
```python
@traceable(name="fetch_financials", run_type="tool",
           tags=["fundamentals", "yfinance"], client=ls_client)
```
Output: revenue, net income, profit margin, revenue growth YoY, EPS, D/E

### get_hurst_exponent ✅
```python
@traceable(name="calc_hurst_exponent", run_type="tool",
           tags=["quant", "regime-detection"], client=ls_client)
```
R/S analysis, lags 2–20, 1Y daily log returns
H > 0.55 → Trending | H < 0.45 → Mean-Reverting | else → Random Walk

### get_hurst_exponent ✅ (IC merged — Bug 4 fixed)
IC Score (Spearman corr rolling Hurst 20d → fwd return 5d) ถูก merge เข้า `_calc_hurst_logic` output แล้ว
NVDA result: IC=-0.1065 (p=0.112) — strong magnitude แต่ contrarian, p ยังไม่ significant
Cell 6 (`get_ic_score` standalone) marked DEPRECATED

### analyze_portfolio_risk ✅ — Cell F เป็น canonical
```python
@traceable(name="portfolio_risk_analysis", run_type="tool",
           tags=["quant", "risk"], client=ls_client)
```
Input: `{ticker: amount}` — tool normalize → weights เอง (`total_amount = sum(raw.values())`)
Metrics: Annualized Return/Volatility, Sharpe, Sortino, **Calmar**, VaR 95%, CVaR 95%, Max Drawdown, per-ticker vol, correlation matrix
Bug 1 fixed: `@tool` wrapper ทำ `json.dumps(portfolio)` ถ้า input เป็น dict
Bug 2 fixed: Calmar merged เข้า return block แล้ว — `_portfolio_risk_logic_patch` ลบออก

### search_market_news ✅
```python
@traceable(name="search_market_news", run_type="tool",
           tags=["search", "news", "gemini"], client=ls_client)
```
Model แยก: `_news_model = ChatGoogleGenerativeAI(...).bind(tools=[{"google_search": {}}])`
ห้ามใส่ `response_schema` คู่กัน → grounding_chunks จะว่าง
Graceful error: quota 429 → คืน error string ไม่ทำ agent พัง

### track_portfolio ✅
```python
@traceable(name="track_portfolio", run_type="tool",
           tags=["portfolio", "tracking"], client=ls_client)
```
`_load_positions_async()` (Cell D) แทน MOCK_PORTFOLIOS แล้ว
Batch fetch 5d, graceful สำหรับ delisted ticker — ไม่ fail ทั้งพอร์ต

---

## Tool decorator pattern — กฎสำคัญ

```python
@tool                          # LLM เห็น docstring ใช้ตัดสินใจ routing
def tool_name(input: str) -> str:
    """USE THIS TOOL when... Do NOT call this for..."""
    return _tool_logic(input)

@traceable(                    # LangSmith trace
    name="descriptive_action", # convention: verb + noun, ไม่ใช่ชื่อ function
    run_type="tool",
    tags=["category", "source"],
    client=ls_client,
)
def _tool_logic(input: str) -> str: ...
```

**ห้ามซ้อน `@tool` + `@traceable` บนฟังก์ชันเดียวกัน** — decorator ตีกัน

---

## Known bugs

### ✅ Bug 1: BadRequestError — model ส่ง dict แทน JSON string (Fixed)

Root cause: Groq schema validation reject ก่อนถึง function แม้ `_portfolio_risk_logic` มี dict guard แล้ว

Fix applied (Cell F — `analyze_portfolio_risk` @tool wrapper):
```python
if not isinstance(portfolio, str):
    portfolio = json.dumps(portfolio)
```
เสริม docstring ระบุ `Input MUST be a JSON string — NOT a JSON object.`

---

### ✅ Bug 2: Calmar Ratio — Fixed (merged into Cell F)

`_portfolio_risk_logic_patch` ลบออกแล้ว — Calmar merge เข้า `_portfolio_risk_logic` (Cell F):
```python
calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")
calmar_str = f"{calmar:.2f}" if calmar == calmar else "N/A"
```
QuantaAlpha reference: Calmar=3.48 (ARR=27.75%, MDD=7.98%) → rule of thumb > 1.0 = acceptable

---

### ✅ Bug 3: Smoke test Cell 5 — Fixed (marked deprecated)

Cell 5 (legacy `analyze_portfolio_risk`) marked DEPRECATED — Cell F เป็น canonical

---

### ✅ Bug 4: get_ic_score — Fixed (IC merged into get_hurst_exponent, Option A)

IC calculation (Spearman corr rolling Hurst 20d → fwd return 5d) ถูก merge เข้า `_calc_hurst_logic` output แล้ว
Cell 6 (`get_ic_score` standalone) marked DEPRECATED — ไม่ต้องมี tool แยก

---

### ⚪ Known limitation: Routing case 5

Query: "P/E กับ profit margin ของ AMD"
Expected: `{get_stock_financials}` | Actual: `{get_stock_financials, get_stock_price}` — 5/5 ครั้ง
Root cause: P/E ผูกกับราคาเชิงความหมาย
Decision: ไม่แก้ใน v1 — benign, v2 ใช้ StateGraph conditional routing

---

## System prompt (Cell 13)

```
You are a quantitative financial analyst assistant.
Always fetch real-time data before answering.
NEVER state any number that did not come from a tool result in this conversation.
If you lack data, call the appropriate tool or say you don't have it — do not estimate from memory.
For portfolio risk questions, use analyze_portfolio_risk.
Provide objective analysis with data. Note that this is not financial advice.
Do not give specific price targets, entry points, or stop-loss levels.
Respond in Thai mixed with English technical terms.
When a question is qualitative (analyst views, news, why a stock moved),
call search_market_news ONLY — do not add get_stock_price unless price is explicitly mentioned.
```

**TODO — Interpretation framework (ยังไม่ implement):**
Agent ควร synthesize cross-signals ไม่ใช่แค่ list ตัวเลข:
- Price position in 52W + Hurst regime → momentum context
- Correlation matrix → true diversification ("AMD-NVDA corr=0.89 = semiconductor block ก้อนเดียว")
- Weight drift (UC-2b) → rebalancing signal
- เมื่อ signals ขัดแย้ง ให้ระบุความขัดแย้งชัดๆ ไม่เลือกข้าง

เพิ่มได้ใน system prompt แต่ต้อง regression test 12 cases ทุกครั้ง

---

## LangSmith tracing pattern — explicit binding

**ทำไมไม่ใช้ env vars:** `langsmith.utils.get_env_var` ถูก cache ด้วย `lru_cache` ตอน import ครั้งแรก — ถ้า import ก่อน set env ค่าค้าง disabled ตลอด session

```python
# Cell 2
ls_client = Client(api_key=_get_secret("LANGCHAIN_API_KEY"),
                   api_url="https://api.smith.langchain.com")
tracer = LangChainTracer(project_name=PROJECT_NAME, client=ls_client)

# run_financial_agent
config = RunnableConfig(callbacks=[tracer], ...)
rt = get_current_run_tree()
return {..., "run_id": str(rt.id) if rt else None}

# ตอนเรียก
with tracing_context(enabled=True, client=ls_client):
    result = run_financial_agent(...)
ls_client.flush()
```

Trace URL:
- Private: `ls_client.read_run(run_id).url`
- Public: `ls_client.share_run(run_id)` — ห้ามประกอบ URL เอง
- ห้าม commit trace URL (มี workspace ID)

ใน FastAPI prod: กลับใช้ env ได้เพราะ set ก่อน import ตอน container start

---

## FastAPI endpoints (Cell G)

```
GET  /health
POST /analyze/stock     → {query, response, ticker, trace_id}
POST /analyze/portfolio → {portfolio, query, response, trace_id}
                          Body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}   → {portfolio_id, response, trace_id}
```

`trace_id` = `run_id` จาก `run_financial_agent` — 1:1 กับ LangSmith

---

## QuantaAlpha integration (arXiv 2602.07085)

เอา concept ไม่ใช่ codebase (architecture ต่างกัน — เขา: Qlib + evolutionary loop)

**Calmar Ratio** → reference: benchmark Calmar=3.48, rule of thumb > 1.0 = acceptable
**IC Score (Rank IC / Spearman)** → วัด signal quality, baseline IC=0.1501 (CSI300 weekly)
NVDA result: IC=-0.1065 (p=0.112) — magnitude strong แต่ contrarian, p ยังไม่ significant

ไม่เอา: Qlib pipeline, evolutionary loop, Next.js frontend

---

## Colab dev notes

- `nest_asyncio.apply()` จำเป็นสำหรับ async SQLAlchemy (Cell A)
- ngrok ต้องการ `NGROK_TOKEN` ใน Colab Secrets
- หลัง refactor: Runtime → Restart → Run all ต้องผ่านครบก่อน push
- ห้าม print/commit API key หรือ trace URL

---

## Build order — สถานะจริง

- [x] Tools: price, financials, hurst
- [x] LangSmith explicit binding
- [x] Agent + UC-1 ✅
- [x] analyze_portfolio_risk amount-based (Cell F) ✅ logic ถูก
- [x] search_market_news Gemini grounding ✅
- [x] track_portfolio SQLite ✅
- [x] FastAPI 4 endpoints + ngrok (ยังไม่ validate ครบ)
- [x] get_ic_score (Cell 6) — exists แต่ architecture debt
- [x] **Fix Bug 1:** normalize dict→string ใน @tool wrapper ✅
- [x] **Fix Bug 2:** merge Calmar เข้า Cell F ✅
- [x] **Fix Bug 3:** Cell 5 marked deprecated ✅
- [x] **Fix Bug 4:** IC merged เข้า get_hurst_exponent (Option A) ✅
- [x] uv setup: `pyproject.toml` + `.venv` (Python 3.11) + kernel registered
- [x] `.env` + `.gitignore` สำหรับ local dev
- [x] `_get_secret()` helper — Colab Secrets / `.env` fallback (Cell 1)
- [x] `DB_PATH` env-aware — `/content` = Colab, else `portfolio.db` local
- [ ] Interpretation framework ใน system prompt
- [ ] Dockerfile
- [ ] Streamlit UI
- [ ] README + public trace link + Colab badge

---

## Out of scope v1 → v2

- PostgreSQL + multi-user (JWT auth)
- Custom StateGraph (conditional routing — แก้ case 5)
- Monte Carlo VaR
- React frontend
- RAG จาก SEC filings
- Kalman filter, Shannon entropy
