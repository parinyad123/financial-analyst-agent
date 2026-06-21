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

## Notebook structure (current — API cells removed)

```
Cell 1:    Imports + _get_secret() helper (Colab Secrets / .env fallback)
Cell 2:    LangSmith client + tracer + assert gate  🛑 ไม่ผ่าน = หยุด
Cell 3:    Tools — price, financials, hurst+IC+IR (merged)
Cell 3.5:  DataProvider Protocol + YFinanceProvider
Cell 4:    ⚠️ DEPRECATED — analyze_portfolio_risk weights-based (Cell F คือ canonical)
           (get_ic_score DEPRECATED — inline comment ใน Cell 3, IC+IR merged แล้ว)
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
Cell F:    analyze_portfolio_risk amount-based ← CANONICAL (v1.5: Ulcer/DrawdownDur/RollingCorr/Alpha-Beta)

⚠️ REMOVED: Cell G (FastAPI app) + Cell H (endpoint tests)
   เหตุผล: port conflict bug (Bug 5) เป็นปัญหาเฉพาะ Jupyter+threading+nest_asyncio
   API ถูกเขียนใหม่ตรงใน src/api/ แทน — ดู "FastAPI Target Spec" ด้านล่าง
   notebook ตอนนี้ใช้สำหรับ algorithm/tool development เท่านั้น ไม่ serve API
```

**กฎการรัน:** รันจากบนลงล่างเสมอ ถ้า Cell 2 ไม่ผ่านห้ามรันต่อ
**หลัง refactor:** Runtime → Restart → Run all ต้องผ่านครบก่อน push

---

## Tools (สถานะจริง v1.5)

> **Source of truth กำลังย้าย:** `get_stock_price`, `get_stock_financials`, `get_hurst_exponent` ถูก migrate ไป `src/tools/*.py` แล้ว (logic เหมือนกับ notebook cell ด้านล่าง) — notebook cell ยังเก็บไว้สำหรับ experimentation/algorithm iteration เท่านั้น ของจริงที่ agent ใช้ใน production คือ `src/`

### get_stock_price ✅ (migrated → `src/tools/price.py`)
```python
@traceable(name="fetch_stock_price", run_type="tool",
           tags=["market-data", "yfinance"], client=ls_client)
```
Period 5d เผื่อ market ปิด
Output: price, change%, 52W range, P/E TTM + Forward, market cap, position in 52W range

### get_stock_financials ✅ (migrated → `src/tools/financials.py`)
```python
@traceable(name="fetch_financials", run_type="tool",
           tags=["fundamentals", "yfinance"], client=ls_client)
```
Output: revenue, net income, profit margin, revenue growth YoY, EPS, D/E

### get_hurst_exponent ✅ (migrated → `src/tools/hurst.py`, v1.5 complete)
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

### analyze_portfolio_risk ✅ — Cell F เป็น canonical (ยังไม่ migrate ไป src/)
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

### search_market_news ✅ (ยังไม่ migrate ไป src/)
```python
@traceable(name="search_market_news", run_type="tool",
           tags=["search", "news", "openai"], client=ls_client)
```
Model แยก: `_news_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`
Gemini free tier = 20 req/day หมดเร็ว → switched to OpenAI fallback ใน dev
Graceful error: API failure → คืน error string ไม่ทำ agent พัง

### track_portfolio ✅ (ยังไม่ migrate ไป src/)
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

### ✅ Bug 5: FastAPI port conflict — RESOLVED by removal

Root cause (เดิม): Cell G รัน `app = FastAPI()` + uvicorn (threading + `nest_asyncio`) ซ้ำข้าม cell execution → OSError 10048 (port 8000 in use)

**Resolution:** Cell G และ Cell H **ถูกลบออกจาก notebook แล้ว** ไม่ใช่แค่ deprioritized — เพราะ root cause คือข้อจำกัดของการรัน uvicorn ใน Jupyter cell ซึ่งไม่มีทางแก้ให้สะอาดในบริบทนั้น API ถูกเขียนใหม่ตรงใน `src/api/` แทน (ดู "FastAPI Target Spec") โดยรันผ่าน `uvicorn main:app` ตรงจาก terminal — ไม่มี kernel state ค้าง ไม่มี cell re-execution ให้ port ชนกัน

Notebook ตอนนี้มีหน้าที่เป็น **algorithm/tool development environment เท่านั้น** ไม่ใช่ full-stack demo ที่ serve API ด้วย

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

**Scope ของ "explicit binding":** หมายถึง **credential/client** เท่านั้น (`client=ls_client` ใน `@traceable`, ไม่ใช้ env var สำหรับ API key) — ไม่ได้ครอบคลุม **tracing on/off switch** ซึ่งเป็นคนละ concern กัน `LANGCHAIN_TRACING_V2` ยังต้องตั้งใน env เพราะเป็น switch ระดับ SDK ที่ LangGraph internal เช็คจาก env เสมอ ไม่มี param ให้ผ่านตรง

`src/config.py` ต้องมี 2 อย่างเรียงลำดับนี้เพื่อให้ trace ขึ้นจริง:

```python
from dotenv import load_dotenv
load_dotenv()  # ต้องมาก่อน import langsmith เสมอ — ไม่งั้น lru_cache อ่านค่าผิดลำดับ

import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")  # switch เปิด tracing
# setdefault (ไม่ใช่ =) — ให้ env จริงที่ตั้ง LANGCHAIN_TRACING_V2=false override ได้ ไม่ hardforce

from langsmith import Client
ls_client = Client(api_key=_get_secret("LANGCHAIN_API_KEY"), ...)  # credential ผ่าน explicit param เสมอ
```

**สัญญาณว่าผิดลำดับ:** `run_id` ใน `run_financial_agent()` คืน `None` แม้ agent ทำงานได้ปกติ — ตรวจสอบด้วย end-to-end test แล้วดูค่า `run_id` ใน return value โดยตรง ไม่พึ่งแค่ "import ผ่าน"

```python
# Cell 2 (notebook) / config.py (src/)
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

## FastAPI Target Spec (สำหรับ `src/api/routes.py` — ไม่ใช่ notebook code)

> Cell G/H ใน notebook ถูกลบแล้ว (ดู Bug 5) นี่คือ **business requirement** ของ endpoint ที่ยังต้อง implement ใหม่ใน `src/api/` ไม่ใช่ spec ที่ผูกกับ Colab/ngrok เดิม

```
GET  /health
POST /analyze/stock       → {query, response, ticker, trace_id}
POST /analyze/portfolio   → {portfolio, query, response, trace_id}
                            Body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}      → {portfolio_id, response, trace_id}
```

`trace_id` = `run_id` จาก `run_financial_agent` — 1:1 กับ LangSmith

**สถานะ:** ยังไม่ implement ใน `src/api/` — เป็น next step หลัง tools/db/agent migrate เสร็จ
**Implementation note:** รันผ่าน `uvicorn main:app --reload` ปกติจาก terminal — ไม่ต้องมี threading wrapper, nest_asyncio, หรือ ngrok (ngrok ใช้แค่ตอน local dev ที่ต้องการ public URL ชั่วคราว ถ้า deploy จริงใช้ reverse proxy/hosting ปกติ)

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

- **LangSmith tracing:** `lru_cache` บน `get_env_var` → ใช้ explicit binding เสมอ "explicit binding" ครอบคลุมแค่ credential (`client=ls_client`) ไม่ใช่ tracing on/off switch — `LANGCHAIN_TRACING_V2` ยังต้องอยู่ใน env (ผ่าน `setdefault`) เพราะ LangGraph internal เช็คจาก env เท่านั้น คนละ concern กับ credential
- **`@tool` / `@traceable` ห้ามซ้อน** บนฟังก์ชันเดียวกัน — แยก outer/inner
- **`traceable` naming:** verb + noun (เช่น `"fetch_stock_price"`), ไม่ใช่ชื่อ function
- **Tool hallucination:** tool หายออกจาก `tools = [...]` → agent fabricate metrics convincingly — LangSmith trace ช่วย debug
- **Agent routing:** docstring engineering มี limit — structural fix (StateGraph) คือทางออกที่ถูก
- **Tool input format:** JSON string input over dict + normalize guard ใน `@tool` wrapper
- **Database design:** เก็บเฉพาะ source-of-truth (ticker, shares, avg_cost) — derive ทุกอย่าง on-the-fly
- **Error messages:** แยก "invalid ticker" vs "transient API failure" ไม่ให้ agent เข้าใจผิด
- **asyncio Windows:** `try/except` กับ `loop.run_until_complete()` — ไม่ใช้ `asyncio.run()`
- **Jupyter + uvicorn ไม่เข้ากันสำหรับ production demo:** threading + nest_asyncio ซ้อน event loop ทำให้ port conflict แก้ไม่จบ — บทเรียนคือไม่ต้องพยายาม "fix" pattern นี้ใน notebook อีก ให้ออกจาก notebook ไปเขียน script ตรงๆ ดีกว่า
- **Rolling Hurst > single Hurst:** single point บอกไม่ได้ว่า regime กำลัง shift — time-series ของ H มีประโยชน์กว่า
- **IR สำคัญกว่า IC เดียว:** IC snapshot อาจ noise — IR วัด consistency ข้ามเวลา
- **Factor weights ต้องมาจากข้อมูล:** hardcode weights = pseudo-quant — ต้อง learn (Lasso/ElasticNet) หรือ validate IC per-factor ก่อน
- **SPY tz mismatch ทำให้ Alpha/Beta = N/A เงียบ:** `yf.download()` คืน tz-naive index, `yf.Ticker().history()` (ผ่าน DataProvider) คืน tz-aware — `DatetimeIndex.intersection()` บน mixed-tz ได้ empty set โดยไม่ error ถ้า data fetch มาจาก 2 source ต่างกัน ต้องเช็ค tz-awareness ให้ตรงกันก่อน join เสมอ
- **Thai text "?" ใน log ≠ input เพี้ยน เสมอไป:** ทดสอบด้วย `repr(query)` ก่อนสรุปว่า input ผิด — กรณีนี้ปัญหาคือ (1) bash curl บน Windows ส่ง encoding ผิดตั้งแต่ shell (ไม่ใช่ FastAPI bug) และ (2) Windows stdout default เป็น cp1252 พิมพ์ Thai unicode ไม่ได้ (ไม่ใช่ agent logic ผิด) — แก้ด้วย `sys.stdout.reconfigure(encoding="utf-8")` ใน `main.py` เท่านั้น ไม่ต้องแก้ agent หรือ encoding ของ request — ทดสอบจริงผ่าน Swagger UI (`/docs`) เสมอ เพราะส่ง UTF-8 ถูกต้องอัตโนมัติ ไม่ผ่าน bash curl บน Windows ที่มี encoding ของ shell เองเป็นตัวแปรกวน
- **Git Bash curl บน Windows ไม่เหมาะทดสอบ Thai API:** ส่ง Thai chars เป็น `????` ตั้งแต่ shell ก่อนถึง server เสมอ ทำให้ routing test ดู "ผิด" ทั้งที่ server ทำงานถูก (เคยทำให้เข้าใจผิดว่า news routing regress) — ทดสอบ Thai query ผ่าน **Swagger UI** หรือ **PowerShell `Invoke-RestMethod`** หรือ **Python `urllib`/`requests`** เท่านั้น ไม่ใช้ bash curl บน Windows

---

## Build order — สถานะจริง

### v1 Complete ✅
- [x] Tools: price, financials, hurst+IC
- [x] LangSmith explicit binding
- [x] Agent + UC-1
- [x] analyze_portfolio_risk amount-based (Cell F)
- [x] search_market_news (OpenAI gpt-4o-mini fallback — Gemini free tier quota หมดเร็ว)
- [x] track_portfolio SQLite
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
- [x] Bug 5 resolved by removal: Cell G/H ลบออกจาก notebook

### Production Migration 🔄
- [x] src/config.py — env loading, LangSmith client/tracer
- [x] src/tools/price.py — get_stock_price
- [x] src/tools/financials.py — get_stock_financials
- [x] src/tools/hurst.py — get_hurst_exponent (v1.5: Rolling Hurst + IR + IC)
- [x] src/tools/portfolio_risk.py — analyze_portfolio_risk (v1.5 metrics complete + tz mismatch bug fixed)
- [x] src/tools/news.py — search_market_news
- [x] src/tools/portfolio_track.py — track_portfolio
- [x] src/database/models.py + session.py
- [x] src/agent/prompts.py + core.py
- [x] src/api/schemas.py + routes.py
- [x] main.py — app assembly + uvicorn entry point + UTF-8 stdout reconfigure
- [x] End-to-end test ทั้ง 4 endpoints ผ่าน Swagger UI — verified ด้วย response จริง (Alpha/Beta ไม่ N/A, Thai text ถูกต้อง)
- [x] UC-news (search_market_news) verified — citation annotations จริงจาก web_search_preview, ไม่ hallucinate
- [x] Routing case 5 verified — behavior เหมือนเดิม ไม่ regress
- [x] Error handling (invalid/delisted ticker) verified — แยก "invalid ticker" vs "transient API failure" ถูกต้อง, ไม่ throw 500
- [x] Dockerfile — multi-stage build (uv + python:3.11-slim), build+run+test ผ่านจริงใน container
  (health check 200, agent+LangSmith ทำงานจริง, trace_id เป็น UUID7 จริง, Thai encoding ถูกต้องบน Linux base)

### Remaining
- [x] Regression test 11 cases หลังเพิ่ม v1.5 metrics — `tests/test_routing_regression.py`,
  10/11 passed (case 5 known limitation, deterministic over-fetch ยืนยันด้วย 5x consistency check)
- [ ] README + public trace link + Colab badge
- [x] Streamlit UI — `streamlit_app.py`, 3 tabs, two-tier display, quick-question buttons, error handling

### Backlog (non-blocking)
- [ ] `create_react_agent` deprecation warning (LangGraph v1.0 moved to `langchain.agents`) —
  `src/agent/core.py:37` — fix แล้วต้องรัน `tests/test_routing_regression.py` ซ้ำก่อน merge

---

## Production Migration

### Strategy
- **tools / db / agent**: migrate logic ตรงๆ จาก notebook — logic เหมือนเดิม แค่เปลี่ยน import
- **API**: implement ใหม่ตาม "FastAPI Target Spec" — ไม่มี Cell G/H ให้ migrate แล้ว (ลบออกจาก notebook)
- **Priority**: tools → database → agent → API

### What changes notebook → src/

| Notebook pattern | src/ replacement |
|---|---|
| `_get_secret(key)` Colab/dotenv hybrid | `os.getenv(key)` via `load_dotenv()` ใน `src/config.py` |
| `ls_client`, `tracer` (global) | `from src.config import ls_client, tracer` |
| `nest_asyncio.apply()` | ลบออก — ไม่ต้องการนอก notebook |
| `ngrok` tunnel | ลบออก — ใช้ reverse proxy / Railway จริง |
| `DB_PATH = "/content/..." if Colab` | `DB_PATH = os.getenv("DB_PATH", "portfolio.db")` |
| `!pip install ...` Cell A | อยู่ใน `pyproject.toml` แล้ว |
| Cell G/H (FastAPI in Jupyter, threading+ngrok) | ลบทิ้ง — เขียนใหม่เป็น `src/api/routes.py` ปกติ รันด้วย `uvicorn main:app` |

### Target file structure
```
src/
├── config.py            ✅ done
├── tools/
│   ├── price.py         ✅ done
│   ├── financials.py    ✅ done
│   ├── hurst.py         ✅ done (v1.5)
│   ├── portfolio_risk.py    analyze_portfolio_risk (v1.5 metrics)
│   ├── news.py               search_market_news
│   └── portfolio_track.py    track_portfolio
├── database/
│   ├── models.py        SQLAlchemy models
│   └── session.py       engine, AsyncSessionLocal, init_db()
├── agent/
│   ├── prompts.py       SYSTEM_PROMPT
│   └── core.py          build_agent(), run_financial_agent()
└── api/
    ├── schemas.py       Pydantic request/response models
    └── routes.py        FastAPI router — implement ตาม "FastAPI Target Spec"
main.py                  app assembly + uvicorn entry point
Dockerfile
tests/
└── test_routing_regression.py   ✅ done — 11 cases, mirrors notebook Cell 26-29
```

---

## Streamlit UI — design decisions (ก่อน implement)

**Architecture:** เรียกผ่าน FastAPI ที่มีอยู่ (HTTP request → `localhost:8000`) — ไม่เรียก agent ตรงจาก Streamlit เพื่อให้ FastAPI ยังเป็น single source of truth ของ business logic

**3 sections:**

| Section | Endpoint | Fields |
|---|---|---|
| Chat-style (ถามทั่วไป) | `POST /analyze/stock` | query เดียว (free text) |
| Portfolio Risk (what-if) | `POST /analyze/portfolio` | ticker + **จำนวนเงิน** (ไม่ใช่จำนวนหุ้น) + query เสริม (ไม่บังคับ) |
| Portfolio Tracking | `POST /portfolio/positions` + `GET /portfolio/{id}` | ticker + จำนวนหุ้น + ราคาเฉลี่ยที่ซื้อ |

**Why Portfolio Risk ไม่ต้องมีจำนวนหุ้น:** `analyze_portfolio_risk` คำนวณจาก weight (`amount / total_amount`) ไม่ใช่จำนวนหุ้นจริง — เป็น what-if ก่อนซื้อ ไม่ใช่ของที่ถือแล้ว ต่างจาก Portfolio Tracking ที่ต้องมี `shares` + `avg_cost` จริงเพราะคำนวณ unrealized P&L จากต้นทุนจริง

### Scope ที่ไม่รองรับ (ป้องกัน scope creep ที่ UI layer)

ระบบ**ไม่มี** what-if scenario engine หรือ cross-asset causal analysis — คำถามแบบนี้จะถูกป้องกันด้วย **helper text ที่ UI** (ไม่ใช่ agent classify เอง เพราะเพิ่ม LLM call ที่ไม่ predictable และไม่ประหยัด token จริง):

- ❌ "ถ้าราคา AMD ตก 20% ความเสี่ยงพอร์ตจะเป็นไง" — ไม่มี stress test module (เป็น v2 backlog: correlation-based stress test)
- ❌ "ถ้า Intel ฟื้นตัวจะกระทบพอร์ตยังไง" — ไม่มี causal cross-asset model (out of scope เพราะ false-precision risk เดียวกับ factor engine)
- ✅ "เน้นอธิบายเรื่อง correlation/drawdown" — ได้ เพราะเป็นการตีความข้อมูลที่ tool คำนวณอยู่แล้ว ไม่ต้องคำนวณใหม่

**No conversation memory ข้าม turn** — แต่ละ query ต้อง self-contained (พิมพ์ ticker ครบทุกครั้ง, ห้ามอ้าง "ตัวที่ถามไปแล้ว") เพราะ `run_financial_agent()` เป็น stateless และ DB ไม่เก็บ conversation history Streamlit อาจเก็บ history ไว้แสดงผลใน `st.session_state` แต่**ไม่ส่งกลับเข้า agent**

### UX สำหรับ user ที่ไม่รู้ศัพท์การเงิน

1. **Quick-question buttons** แทนกล่อง text เปล่า ลด barrier ตอนเริ่มถาม (เนื้อหาข้อความ — ดูตัวอย่างที่ตกลงไว้แล้วในการสนทนาออกแบบ ก่อน implement)
2. **Two-tier display**: สรุปภาษาง่าย (ดึงจาก paragraph สุดท้ายของ agent response) แสดงเด่น, ตัวเลขเทคนิค (Sharpe, Calmar, Ulcer Index ฯลฯ) เก็บใน `st.expander` ที่พับได้ — ไม่ต้องแก้ agent/system prompt เพราะ response มีทั้งสองส่วนอยู่แล้วในตัว งานคือ parse/split ฝั่ง UI เท่านั้น

### v2 backlog ที่เกิดจาก design discussion นี้

- **Correlation-based stress test** — รับ shock input เช่น `{"AMD": -0.20}` ใช้ correlation matrix + volatility ที่มีอยู่แล้วประมาณผลกระทบแบบ linear (ไม่ใช่ Monte Carlo) ต้องระบุขอบเขตความแม่นยำชัดในผลลัพธ์ (เป็น linear approximation จาก correlation ในอดีต ไม่ใช่การพยากรณ์ — correlation breakdown ตอนตลาดเครียดจริงเป็นความเสี่ยงที่ต้องบอกตรงๆ)
- **What-if เพิ่ม asset นอกพอร์ต** (เช่นเพิ่ม INTC เป็น hypothetical position) — ได้แค่ correlation/diversification effect เชิงตัวเลข ไม่ใช่ causal impact จากข่าว
- **Conversation memory ข้าม turn** — ต้องเพิ่ม `ConversationTurn` table (`session_id`, `role`, `content`, `trace_id`) + pass message history เข้า `run_financial_agent()` แทนสร้าง `[HumanMessage(...)]` ใหม่ทุกครั้ง

---

## Out of scope v1 → v2

- Walk-forward Backtesting (ต้องการ signal definition + position sizing + transaction cost model ก่อน)
- Hidden Markov Model / Bayesian Change Point (interpretability cost > value สำหรับ interview demo)
- Factor engine ด้วย hardcode weights (ต้องการ IC validation per-factor ก่อน)
- Causal cross-asset impact analysis (เช่น "ข่าว X กระทบพอร์ต Y เท่าไหร่") — false-precision risk เดียวกับ factor engine
- PostgreSQL + multi-user (JWT auth)
- Custom StateGraph (conditional routing — แก้ case 5)
- Monte Carlo VaR
- React frontend
- RAG จาก SEC filings
- Kalman filter, Shannon entropy
- Multi-asset class (forex/crypto/options) — yfinance รองรับ spot price ของ forex/crypto ในทางเทคนิค
  แต่ `get_stock_financials` (revenue/EPS/D-E) ผูกกับ equity เท่านั้น, benchmark ของ Calmar/IC
  (QuantaAlpha paper) validate กับ equity เท่านั้น, options ไม่มี data source ใน yfinance เลย
  (ต้องการ Black-Scholes + Greeks + IV surface — scope ใหญ่ระดับเดียวกับ factor engine)
  Decision: คง positioning "Explainable Quant Analytics for equity" ไว้ใน v1 ไม่เจือจาง scope