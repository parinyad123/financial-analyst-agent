# 📊 Financial Analyst Agent — CLAUDE.md

## Project overview

Physics-informed financial analysis ที่ผสม quantitative signals (Hurst exponent, IC Score, IR) กับ LLM reasoning ผ่าน ReAct agent โดยมี LangSmith tracing สำหรับ observability และ SQLite สำหรับ portfolio persistence

เป้าหมายสองอย่างพร้อมกัน: portfolio project สำหรับสมัครงาน FinTech และ tool ใช้งานจริงสำหรับนักลงทุน

หลักการ scope: **จบและ demo ได้ สำคัญกว่าทะเยอทะยานแล้วค้าง**

Positioning: **Explainable Quant Analytics + Agent Orchestration** — ไม่แข่งเรื่อง model complexity

---

## Tech stack

| Layer | Technology | หมายเหตุ |
|---|---|---|
| LLM (dev) | Groq — `openai/gpt-oss-120b` | `reasoning_effort="low"`, tool orchestration เสถียร |
| LLM (prod) | Gemini 2.5 Flash | swap ตอน deploy |
| LLM (news) | OpenAI `gpt-4o-mini` | `_news_model` ใน `search_market_news` — Gemini free tier 20 req/day หมดเร็ว |
| Agent framework | LangGraph `create_react_agent` | prebuilt ReAct |
| Market data | yfinance (via `YFinanceProvider`) | real-time price, fundamentals, history |
| Observability | LangSmith | **explicit binding — ไม่พึ่ง env vars** |
| Backend | FastAPI + Pydantic v2 | async endpoints + ngrok สำหรับ local dev |
| Database | SQLite + SQLAlchemy async | `aiosqlite` + `nest_asyncio` สำหรับ Windows/Colab |
| Package manager | uv | `pyproject.toml` + `.venv` (Python 3.11) |
| Environment | local (uv) / Google Colab → Docker (planned) | |

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
    ├── get_stock_price            → YFinanceProvider → 5d history
    ├── get_stock_financials       → YFinanceProvider → .info
    ├── get_hurst_exponent         → YFinanceProvider 1Y + numpy R/S + Rolling Hurst + IC + IR
    ├── analyze_portfolio_risk     → YFinanceProvider 1Y + numpy/pandas (amount-based)
    ├── track_portfolio            → SQLite + YFinanceProvider 5d
    └── search_market_news         → OpenAI gpt-4o-mini (Gemini quota: 20 req/day)
        ↓
LangSmith (traces ทุก step)
    ↓
FastAPI (4 endpoints) + ngrok tunnel
```

---

## Data layer

### DataProvider Protocol (Cell 3.5 — ใหม่ใน v1.5)

```python
class DataProvider(Protocol):
    def get_history(self, ticker: str, period: str) -> pd.DataFrame: ...
    def get_info(self, ticker: str) -> dict: ...

class YFinanceProvider:
    """concrete implementation — สลับ Polygon/DuckDB ใน v2 ได้โดยไม่แตะ logic"""
    def get_history(self, ticker, period):
        return yf.Ticker(ticker).history(period=period)
    def get_info(self, ticker):
        return yf.Ticker(ticker).info
```

ประโยชน์: test ง่าย (mock provider แทน yfinance จริง), เปลี่ยน data source ใน v2 โดยไม่ refactor tools

---

## Use cases

### UC-1: วิเคราะห์หุ้นรายตัว ✅
- Input: ticker หรือ natural language query
- Tools: `get_stock_price` + `get_stock_financials` + `get_hurst_exponent`

### UC-2a: วางแผน portfolio ก่อนซื้อ ✅ (amount-based)
- Input: `{ticker: amount}` — tool คำนวณ weights เอง
- Metrics: Annualized Return/Volatility, Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%, Max Drawdown, Ulcer Index, Drawdown Duration, Rolling Correlation (60d), Benchmark Alpha/Beta vs SPY
- Tools: `analyze_portfolio_risk` (Cell F — canonical)

### UC-2b: ติดตาม portfolio หลังซื้อ ✅
- Input: `portfolio_id` string → load จาก SQLite
- Output: unrealized P&L per position, total MV, total P&L, current weights
- Tools: `track_portfolio`

### UC-news: ข่าว + analyst commentary ✅
- Tools: `search_market_news` → OpenAI gpt-4o-mini (Gemini free tier 20 req/day หมดเร็ว)
- Routing: 10/11 ผ่าน (ดู Known limitations)

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

## Notebook structure (v1.5 target)

```
Cell 1:    Imports + _get_secret() helper (Colab Secrets / .env fallback)
Cell 2:    LangSmith client + tracer + assert gate  🛑 ไม่ผ่าน = หยุด
Cell 3:    Tools — price, financials, hurst+IC+IR (merged)
Cell 3.5:  DataProvider Protocol + YFinanceProvider  ← ใหม่ v1.5
Cell 4:    ⚠️ DEPRECATED — analyze_portfolio_risk weights-based (Cell F คือ canonical)
           (get_ic_score DEPRECATED — inline comment ใน Cell 3, IC+IR merged แล้ว)
Gate:      Gate check ก่อน Cell 5 — Gemini quota (non-fatal try/except)
Cell 5:    search_market_news (OpenAI gpt-4o-mini fallback)
Cell 6:    track_portfolio (SQLite via _load_positions_async)
Cell 13:   Agent setup — ChatGroq + tools list + SYSTEM_PROMPT
Cell 15:   run_financial_agent — @traceable entry point
Cell 17–18: Tests UC-1
Cell 20:   Test UC-2a
Cell 24:   Test UC-2b
Cell 26–29: Routing regression tests (11 cases) + consistency checks
Cell A:    pip install sqlalchemy aiosqlite nest_asyncio (Colab only)
Cell B:    SQLAlchemy models + async engine
Cell C:    Seed MOCK_PORTFOLIOS เข้า DB
Cell D:    _load_positions_async()
Cell F:    analyze_portfolio_risk amount-based ← CANONICAL (v1.5: เพิ่ม Ulcer/DrawdownDur/RollingCorr/Alpha-Beta)
Cell G:    FastAPI app (4 endpoints) ← ต้องแก้ port conflict bug
Cell H:    Test endpoints ← ต้องแก้ port conflict bug
```

**กฎการรัน:** รันจากบนลงล่างเสมอ ถ้า Cell 2 ไม่ผ่านห้ามรันต่อ
**หลัง refactor:** Runtime → Restart → Run all ต้องผ่านครบก่อน push

---

## Tools (สถานะจริง v1.5)

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

### get_hurst_exponent ✅ (v1.5 complete)
```python
@traceable(name="calc_hurst_exponent", run_type="tool",
           tags=["quant", "regime-detection"], client=ls_client)
```
R/S analysis, lags 2–20, 1Y daily log returns
H > 0.55 → Trending | H < 0.45 → Mean-Reverting | else → Random Walk

**Output includes (v1.5):**
- IC Score = Spearman(rolling Hurst 20d, fwd return 5d) — quality: Strong/Usable/Weak
- IR = mean(IC_monthly) / std(IC_monthly) — consistency: IR > 0.5 usable, > 1.0 strong
- Rolling Hurst (window=126d, step=5d): early → recent trend (↗ Rising / ↘ Falling)
- Graceful: ทุก metric มี try/except — ถ้า data ไม่พอ → skip ไม่ fail

### analyze_portfolio_risk ✅ — Cell F เป็น canonical
```python
@traceable(name="portfolio_risk_analysis", run_type="tool",
           tags=["quant", "risk"], client=ls_client)
```
Input: `{ticker: amount}` — normalize → weights เอง (`total_amount = sum(raw.values())`)

**Metrics (v1.5 — all implemented in Cell F):**
- Annualized Return/Volatility, Sharpe, Sortino, Calmar
- VaR 95%, CVaR 95%, Max Drawdown
- Ulcer Index = `sqrt(mean(drawdown²))` — pain metric รวม severity + duration
- Drawdown Duration: median + max days ต่ำกว่า peak
- Per-ticker Annualized Volatility
- Pearson Correlation Matrix (1Y static)
- Rolling Correlation (last 60d) — จับ tail dependency ที่ static Pearson มองไม่เห็น
- Alpha (annualized, vs SPY) + Beta vs SPY — CAPM market-model

### search_market_news ✅
```python
@traceable(name="search_market_news", run_type="tool",
           tags=["search", "news", "openai"], client=ls_client)
```
Model แยก: `_news_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`
Gemini free tier = 20 req/day หมดเร็ว → switched to OpenAI fallback ใน dev
Graceful error: API failure → คืน error string ไม่ทำ agent พัง

### track_portfolio ✅
```python
@traceable(name="track_portfolio", run_type="tool",
           tags=["portfolio", "tracking"], client=ls_client)
```
`_load_positions_async()` (Cell D) โหลดจาก SQLite
Batch fetch 5d, graceful สำหรับ delisted ticker

---

## Tool decorator pattern — กฎสำคัญ

```python
@tool                          # LLM เห็น docstring ใช้ตัดสินใจ routing
def tool_name(input: str) -> str:
    """USE THIS TOOL when... Do NOT call this for..."""
    return _tool_logic(input)

@traceable(
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

### ✅ Bug 1: BadRequestError — dict→JSON string (Fixed)
Fix: `@tool` wrapper ทำ `json.dumps(portfolio)` ถ้า input เป็น dict

### ✅ Bug 2: Calmar Ratio (Fixed — merged เข้า Cell F)
`calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")`

### ✅ Bug 3: Cell 5 smoke test (Fixed — marked DEPRECATED)

### ✅ Bug 4: get_ic_score standalone (Fixed — IC merged เข้า get_hurst_exponent)

### 🔴 Bug 5: FastAPI 404 — port conflict (BLOCKING — ยังไม่แก้)

Root cause: Cell G รัน `app = FastAPI()` + uvicorn ซ้ำ → OSError 10048 (port 8000 in use)
Cell H test ล้มเหลว: `/health` → 404, `/analyze/stock` → 404

**วิธีแก้ที่ถูกต้อง:**
```python
# Cell G ต้องเช็คก่อนว่า server กำลังรันอยู่ไหม
import socket
def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if not _port_in_use(8000):
    # start server + ngrok
else:
    print("⚠️ Port 8000 already in use — reusing existing server")
    # set PUBLIC_URL จาก ngrok tunnels ที่มีอยู่แล้ว
    tunnels = ngrok.get_tunnels()
    PUBLIC_URL = tunnels[0].public_url if tunnels else None
```

ถ้า endpoint ยัง 404 หลังแก้ port: ปัญหาคือ FastAPI app instance ที่ serve อยู่ไม่ใช่ instance เดียวกับที่ define routes → **Kernel Restart → Run all** เท่านั้น

---

## Known limitations

### ⚪ Routing case 5: P/E + margin over-fetches get_stock_price

Query: "P/E กับ profit margin ของ AMD"
Expected: `{get_stock_financials}` | Actual: `{get_stock_financials, get_stock_price}` — 5/5 ครั้ง
Root cause: P/E ผูกกับราคาเชิงความหมาย — docstring-level negative routing งัดไม่ขึ้น
Decision: ไม่แก้ใน v1 — benign (price tool เร็ว/ถูกสุด), v2 ใช้ StateGraph conditional routing

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
- เมื่อ signals ขัดแย้ง ให้ระบุความขัดแย้งชัดๆ ไม่เลือกข้าง

เพิ่มได้ใน system prompt แต่ต้อง regression test 11 cases ทุกครั้งหลังแก้

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
POST /analyze/stock       → {query, response, ticker, trace_id}
POST /analyze/portfolio   → {portfolio, query, response, trace_id}
                            Body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}      → {portfolio_id, response, trace_id}
```

`trace_id` = `run_id` จาก `run_financial_agent` — 1:1 กับ LangSmith

**สถานะ:** endpoints define แล้วใน Cell G แต่ยังมี port conflict bug (Bug 5) ทำให้ test ใน Cell H ล้มเหลว

---

## QuantaAlpha integration (arXiv 2602.07085)

เอา concept ไม่ใช่ codebase

**Calmar Ratio** → benchmark = 3.48 (ARR=27.75%, MDD=7.98%), rule of thumb > 1.0
**IC Score (Rank IC / Spearman)** → signal quality, baseline IC=0.1501 (CSI300 weekly)
**IR = mean(IC)/std(IC)** → consistency: > 0.5 usable, > 1.0 strong, > 2.0 exceptional
NVDA result: IC=-0.1065 (p=0.112) — magnitude strong แต่ contrarian, p ยังไม่ significant

ไม่เอา: Qlib pipeline, evolutionary loop, Next.js frontend, fixed factor weights (pseudo-quant ถ้าไม่มี IC validation per-factor)

---

## Key learnings & principles

- **LangSmith tracing:** `lru_cache` บน `get_env_var` → ใช้ explicit binding เสมอ
- **`@tool` / `@traceable` ห้ามซ้อน** บนฟังก์ชันเดียวกัน — แยก outer/inner
- **`traceable` naming:** verb + noun (เช่น `"fetch_stock_price"`), ไม่ใช่ชื่อ function
- **Tool hallucination:** tool หายออกจาก `tools = [...]` → agent fabricate metrics convincingly — LangSmith trace ช่วย debug
- **Agent routing:** docstring engineering มี limit — structural fix (StateGraph) คือทางออกที่ถูก
- **Tool input format:** JSON string input over dict + normalize guard ใน `@tool` wrapper
- **Database design:** เก็บเฉพาะ source-of-truth (ticker, shares, avg_cost) — derive ทุกอย่าง on-the-fly
- **Error messages:** แยก "invalid ticker" vs "transient API failure" ไม่ให้ agent เข้าใจผิด
- **asyncio Windows:** `try/except` กับ `loop.run_until_complete()` — ไม่ใช้ `asyncio.run()`
- **FastAPI cell ordering:** `app = FastAPI()` ต้องอยู่ในเซลล์เดียวกับ endpoints — cell ว่างก่อน = 404
- **Port conflicts:** duplicate uvicorn start → kernel restart เท่านั้น หรือเช็ค port ก่อน start
- **Rolling Hurst > single Hurst:** single point บอกไม่ได้ว่า regime กำลัง shift — time-series ของ H มีประโยชน์กว่า
- **IR สำคัญกว่า IC เดียว:** IC snapshot อาจ noise — IR วัด consistency ข้ามเวลา
- **Factor weights ต้องมาจากข้อมูล:** hardcode weights = pseudo-quant — ต้อง learn (Lasso/ElasticNet) หรือ validate IC per-factor ก่อน

---

## Build order — สถานะจริง

### v1 Complete ✅
- [x] Tools: price, financials, hurst+IC
- [x] LangSmith explicit binding
- [x] Agent + UC-1
- [x] analyze_portfolio_risk amount-based (Cell F)
- [x] search_market_news (OpenAI gpt-4o-mini fallback — Gemini free tier quota หมดเร็ว)
- [x] track_portfolio SQLite
- [x] FastAPI 4 endpoints (defined — port bug pending)
- [x] Fix Bug 1: dict→string normalize
- [x] Fix Bug 2: Calmar merged เข้า Cell F
- [x] Fix Bug 3: Cell 5 marked deprecated
- [x] Fix Bug 4: IC merged เข้า get_hurst_exponent
- [x] uv setup: pyproject.toml + .venv (Python 3.11)
- [x] .env + .gitignore สำหรับ local dev
- [x] _get_secret() helper — Colab Secrets / .env fallback
- [x] DB_PATH env-aware — /content = Colab, else portfolio.db local
- [x] Routing regression: 10/11 pass

### v1.5 Complete ✅
- [x] DataProvider Protocol + YFinanceProvider (Cell 3.5)
- [x] Rolling Hurst (126d window, step 5d) ใน `_calc_hurst_logic`
- [x] IR = mean(IC_monthly) / std(IC_monthly) ใน `_calc_hurst_logic`
- [x] Rolling Correlation 60d ใน Cell F
- [x] Ulcer Index + Drawdown Duration ใน Cell F
- [x] Alpha/Beta vs SPY ใน Cell F

### Remaining 🔄
- [ ] **🔴 Fix Bug 5: FastAPI port conflict** — ต้องแก้ก่อน demo API ได้
- [ ] Regression test 11 cases หลังเพิ่ม v1.5 metrics
- [ ] Dockerfile
- [ ] Streamlit UI
- [ ] README + public trace link + Colab badge

---

## Out of scope v1 → v2

- Walk-forward Backtesting (ต้องการ signal definition + position sizing + transaction cost model ก่อน)
- Hidden Markov Model / Bayesian Change Point (interpretability cost > value สำหรับ interview demo)
- Factor engine ด้วย hardcode weights (ต้องการ IC validation per-factor ก่อน)
- PostgreSQL + multi-user (JWT auth)
- Custom StateGraph (conditional routing — แก้ case 5)
- Monte Carlo VaR
- React frontend
- RAG จาก SEC filings
- Kalman filter, Shannon entropy