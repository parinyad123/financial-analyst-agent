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
| UI | Streamlit | เรียกผ่าน FastAPI (HTTP) — ไม่เรียก agent ตรง |

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
    ├── analyze_portfolio_risk     → YFinanceProvider 1Y + numpy/pandas (amount-based) + Risk Contribution
    ├── track_portfolio            → SQLite + YFinanceProvider 5d + Risk Contribution
    └── search_market_news         → OpenAI gpt-4o-mini (Gemini quota: 20 req/day)
        ↓
LangSmith (traces ทุก step)
    ↓
FastAPI (5 endpoints)
    ↓
Streamlit UI (3 tabs) — calls FastAPI via requests
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

### UC-2a: วางแผน portfolio ก่อนซื้อ ✅ (amount-based, what-if)
- Input: `{ticker: amount}` — tool คำนวณ weights เอง
- Metrics: Annualized Return/Volatility, Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%, Max Drawdown, Ulcer Index, Drawdown Duration, Rolling Correlation (60d), Benchmark Alpha/Beta vs SPY, **Risk Contribution to Variance**
- Tools: `analyze_portfolio_risk` (Cell F — canonical)
- **Persona:** what-if/hypothetical เท่านั้น — ห้ามพูด "ของคุณ", ห้าม self-generated stress test (ดู System prompt fix ด้านล่าง)

### UC-2b: ติดตาม portfolio หลังซื้อ ✅
- Input: `portfolio_id` string → load จาก SQLite
- Output: unrealized P&L per position, total MV, total P&L, current weights, **Risk Contribution to Variance (MV-weighted)**
- Tools: `track_portfolio`
- **Persona:** actual holdings — มี real cost basis รองรับ ใช้ "ของคุณ"/"you are up/down" ได้

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

## Notebook structure (current — API cells removed, NOT source of truth)

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

**⚠️ Notebook ไม่ sync กับ `src/` แล้ว ตั้งแต่ production migration — ห้ามใช้ทดลอง feature ใหม่อีกต่อไป**

Verified diff (เช็คตอน implement Risk Contribution Analysis):

| จุด | Notebook | `src/` |
|---|---|---|
| `_portfolio_risk_logic` Alpha/Beta | ไม่มี tz fix — อาจได้ N/A เงียบๆ | มี tz-naive strip ก่อน intersection |
| `_track_portfolio_logic` error message | แสดง `MOCK_PORTFOLIOS.keys()` (hardcoded) | query DB จริงผ่าน `_list_portfolio_ids()` |
| `SYSTEM_PROMPT` | Base prompt เท่านั้น | + portfolio JSON guardrail + UC-2a/UC-2b persona separation |

**Decision:** notebook คงไว้เป็น **historical reference เท่านั้น** ไม่ sync ตามทุกครั้งที่แก้ `src/` (maintenance cost ไม่คุ้มกับประโยชน์ — `src/` คือ source of truth เพียงที่เดียว)

---

## Tools (สถานะจริง v1.5)

> **Source of truth กำลังย้าย:** ทุก tool ถูก migrate ไป `src/tools/*.py` แล้ว (logic เหมือนกับ notebook cell ด้านล่าง แต่มี bugfix เพิ่มที่ notebook ไม่มี — ดู diff ด้านบน) — notebook cell ยังเก็บไว้สำหรับอ้างอิงประวัติเท่านั้น ของจริงที่ agent ใช้ใน production คือ `src/`

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

### analyze_portfolio_risk ✅ (migrated → `src/tools/portfolio_risk.py`, v1.5 + Risk Contribution complete)
```python
@traceable(name="portfolio_risk_analysis", run_type="tool",
           tags=["quant", "risk"], client=ls_client)
```
Input: `{ticker: amount}` — normalize → weights เอง (`total_amount = sum(raw.values())`)

**Metrics (v1.5 — all implemented):**
- Annualized Return/Volatility, Sharpe, Sortino, Calmar
- VaR 95%, CVaR 95%, Max Drawdown
- Ulcer Index = `sqrt(mean(drawdown²))` — pain metric รวม severity + duration
- Drawdown Duration: median + max days ต่ำกว่า peak
- Per-ticker Annualized Volatility
- Pearson Correlation Matrix (1Y static)
- Rolling Correlation (last 60d) — จับ tail dependency ที่ static Pearson มองไม่เห็น
- Alpha (annualized, vs SPY) + Beta vs SPY — CAPM market-model (tz-mismatch bug fixed — ดู Known bugs)
- **Risk Contribution to Variance** (section 10) — `MCR[i] = w[i] × (Σw)[i] / (w^T Σw)` — sums to 100% by construction
  Verified: `{"NVDA":5000,"AMD":3000}` → NVDA weight 62.5% but risk contribution 46.1%, AMD weight 37.5%
  but risk contribution 53.9% (AMD volatility สูงกว่ามาก — ตรงตามทฤษฎี), sanity check sum = 100.0000%

### search_market_news ✅ (migrated → `src/tools/news.py`)
```python
@traceable(name="search_market_news", run_type="tool",
           tags=["search", "news", "openai"], client=ls_client)
```
Model แยก: `_news_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)` + `web_search_preview` tool
Gemini free tier = 20 req/day หมดเร็ว → switched to OpenAI fallback ใน dev
Graceful error: API failure → คืน error string ไม่ทำ agent พัง
Verified: citation annotations จริงจาก web_search_preview, ไม่ hallucinate (เคย verify ด้วย TSLA case ที่ราคาจริงขึ้นไม่ใช่ลงตาม query สมมติ — agent อธิบายตามจริง)

### track_portfolio ✅ (migrated → `src/tools/portfolio_track.py`, + Risk Contribution complete)
```python
@traceable(name="track_portfolio", run_type="tool",
           tags=["portfolio", "tracking"], client=ls_client)
```
`_load_positions_async()` โหลดจาก SQLite จริง (ไม่ใช่ MOCK_PORTFOLIOS แบบ notebook)
Batch fetch 5d, graceful สำหรับ delisted ticker — แยก "invalid ticker" vs "transient API failure"
**Risk Contribution to Variance** (section 5, MV-weighted) — ใช้ current market value weight ไม่ใช่ cost basis
  Verified: `streamlit-test-001` (NVDA+TSLA) → weight 50.7%/49.3% but risk contribution 43.4%/56.6%
  (TSLA volatile กว่า), sanity check sum = 100.0000%
  Dead ticker handling verified: seed `test-dead-ticker` (NVDA + ZZZFAKE999) → ZZZFAKE999 excluded
  gracefully, remaining single asset falls back to "100.0% (single asset)", warning ยังแสดงครบ
**Pearson Correlation Matrix (1Y) + Rolling Correlation (60d)** (section 5, เพิ่มทีหลัง) — reuse `hist_returns`
  ที่คำนวณไว้แล้วสำหรับ Risk Contribution ไม่ fetch ข้อมูลเพิ่ม — แทนที่ emergent tool chaining เดิม
  (ดู "Risk Contribution Analysis" ด้านล่าง สำหรับ root cause และเหตุผลที่เลือกทำแบบนี้)

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

### ✅ Bug 6: SPY tz mismatch → Alpha/Beta = N/A เงียบ (Fixed ใน `src/`, ยังไม่แก้ใน notebook)

Root cause: `yf.download()` คืน tz-naive index, `yf.Ticker().history()` (ผ่าน DataProvider) คืน tz-aware — `DatetimeIndex.intersection()` บน mixed-tz ได้ empty set โดยไม่ error

Fix: strip tz ก่อน intersection ใน `src/tools/portfolio_risk.py` — Verified: `Alpha (ann, vs SPY): +26.76% | Beta: 2.251` (ไม่ใช่ N/A)

### ✅ Bug 7: UI misattribution — "พอร์ตของคุณเคยขาดทุน" ใน what-if form (Fixed)

ดู "System prompt fix — UC-2a/UC-2b persona separation" ด้านล่าง

---

## Known limitations

### ⚪ Routing case 5: P/E + margin over-fetches get_stock_price

Query: "P/E กับ profit margin ของ AMD"
Expected: `{get_stock_financials}` | Actual: `{get_stock_financials, get_stock_price}` — 5/5 ครั้ง
Root cause: P/E ผูกกับราคาเชิงความหมาย — docstring-level negative routing งัดไม่ขึ้น
Decision: ไม่แก้ใน v1 — benign (price tool เร็ว/ถูกสุด), v2 ใช้ StateGraph conditional routing
Verified deterministic: 5x consistency check ใน `tests/test_routing_regression.py::test_case5_consistency` — ผลเหมือนกันทุกรอบ

---

## System prompt (`src/agent/prompts.py`)

```python
SYSTEM_PROMPT = """You are a quantitative financial analyst assistant.
Always fetch real-time data before answering.
NEVER state any number that did not come from a tool result in this conversation.
If you lack data, call the appropriate tool or say you don't have it — do not estimate from memory.
For portfolio risk questions, use analyze_portfolio_risk.
Provide objective analysis with data. Note that this is not financial advice.
Do not give specific price targets, entry points, or stop-loss levels.
Respond in Thai mixed with English technical terms.
When a question is qualitative (analyst views, news, why a stock moved),
call search_market_news ONLY — do not add get_stock_price unless price is explicitly mentioned.
If portfolio data is already provided as a JSON string in the message (e.g. '{"NVDA": 5000, "AMD": 3000}'),
call analyze_portfolio_risk immediately with that JSON — do NOT ask the user to provide portfolio data again.

For analyze_portfolio_risk (UC-2a — hypothetical/what-if, before purchase):
NEVER say "your portfolio lost" or "พอร์ตของคุณเคยขาดทุน" or imply the user
actually held this position during the analyzed period. This tool only
receives {ticker: amount}, never a purchase date or cost basis — there is
no "your" loss to refer to. Always frame results as hypothetical: "if this
portfolio had been held during the period..." or "พอร์ตสมมตินี้...".
NEVER compute new what-if numbers that are not direct tool output — e.g.
do NOT calculate "if market drops X%, portfolio drops Y%" from Beta or
correlation yourself, even as simple multiplication. Report Beta/Correlation
exactly as the tool returns them, and explain only qualitatively (e.g.
"Beta สูง = ผันผวนกว่าตลาดในอดีต" — NOT a projected future loss number).
If asked a stress-test question directly, state clearly that this analysis
does not support scenario simulation. When relevant, suggest the user try
track_portfolio instead if they want to track an actual position they
already hold (which has real cost basis).

For track_portfolio (UC-2b — actual holdings):
This tool receives real shares + avg_cost from the database, so "your
portfolio", "you are up/down" language IS appropriate here — there is a
real cost basis backing it."""
```

**TODO — Interpretation framework (ยังไม่ implement):**
Agent ควร synthesize cross-signals ไม่ใช่แค่ list ตัวเลข:
- Price position in 52W + Hurst regime → momentum context
- Correlation matrix → true diversification ("AMD-NVDA corr=0.89 = semiconductor block ก้อนเดียว")
- เมื่อ signals ขัดแย้ง ให้ระบุความขัดแย้งชัดๆ ไม่เลือกข้าง

**✅ Risk Contribution guardrail (implemented):**
ห้ามแปล risk contribution number เป็น directive คำสั่ง ("ควรขาย/ต้องขาย") + ห้าม stop-loss/hedging action แม้ไม่ระบุราคา (strengthened หลังเจอ drift) — ดู "Risk Contribution Analysis" ด้านล่าง

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
- Public: `ls_client.share_run(run_id)` — ห้ามประกอบ URL เอง, link หมดอายุไม่กี่สัปดาห์ — ไม่ commit ใน README, ใช้ trace_id แทนเป็นหลักฐานว่า observability ทำงานจริง
- ห้าม commit trace URL (มี workspace ID)

ใน FastAPI prod: กลับใช้ env ได้เพราะ set ก่อน import ตอน container start

---

## FastAPI endpoints (`src/api/routes.py` — implemented)

```
GET  /health
POST /analyze/stock       → {query, response, ticker, trace_id}
POST /analyze/portfolio   → {portfolio, query, response, trace_id}
                            Body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}      → {portfolio_id, response, trace_id}
```

`trace_id` = `run_id` จาก `run_financial_agent` — 1:1 กับ LangSmith

**สถานะ:** implemented + verified ผ่าน Swagger UI และ Streamlit ทั้ง 5 endpoints
**Implementation note:** รันผ่าน `uvicorn main:app --reload` ปกติจาก terminal — ไม่มี threading wrapper, nest_asyncio, หรือ ngrok

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
- **Notebook ≠ source of truth หลัง production migration:** bugfix ที่ทำใน `src/` (SPY tz, error messages, system prompt persona separation) ไม่ได้ sync กลับ notebook โดยอัตโนมัติ — ก่อนทดลอง feature ใหม่ ต้อง verify ว่าจะทดลองที่ไหน (`src/` ตรงผ่าน standalone script, ไม่ใช่ notebook) ไม่งั้นจะเจอ bug ที่ "หายไปแล้ว" กลับมาอีกและเข้าใจผิดว่า fix ใหม่ทำให้พัง
- **English negative instruction แม่นกว่าไทยสำหรับ gpt-oss-120b:** ไม่ต้องแปล system prompt ทั้งหมด — แทรกกฎ critical (persona separation, ห้าม self-generate ตัวเลข) เป็นอังกฤษแบบ mixed-language กับของเดิมที่เป็นไทยได้ ลดความเสี่ยง regression จาก language shift ทั้งระบบ
- **Risk contribution ต้อง verify ด้วย sanity check sum=100%:** มี mathematical backing ชัดเจน (`MCR[i] = w[i] × cov(i,portfolio)/var(portfolio)`) ต่างจาก fixed factor weight ที่เป็น pseudo-quant — แต่ implementation ผิดเล็กน้อย (เช่น ใช้ correlation แทน covariance) จะทำให้ sum ≠ 100% ซึ่งตรวจจับได้ง่ายถ้า print ออกมาดูทุกครั้ง
- **Regression fail ไม่เท่ากับ regression จริงเสมอ:** หลังแก้ prompt เจอ routing fail 9/11 (จากปกติ 10/11) — ก่อนสรุปว่าเป็น regression ให้ isolate test case ที่ fail แล้วรันซ้ำ 3 รอบเดี่ยวๆ (เร็วกว่ารัน suite เต็มซ้ำมาก) ถ้าผ่านครบทุกรอบ = LLM nondeterministic fluke ไม่ใช่ผลจาก prompt change เช็ค error/ticker แปลกปลอมที่หลุดมาด้วย (เช่น ticker ที่ไม่มีในคำถามเลย) เป็น signal เสริมว่าเป็น fluke จริงหรือมีปัญหาอื่นซ่อนอยู่
- **Negative instruction ต้องมี negative example ไม่ใช่แค่กฎเปล่า:** "ห้ามให้คำแนะนำ" อาจไม่ครอบคลุมพอ — agent อาจตีความว่า "แนะนำให้ตั้ง stop-loss" ไม่ใช่คำแนะนำเพราะไม่ได้ระบุราคา ต้องเสริมตัวอย่างชัดว่า "การแนะนำ action (เช่น ตั้ง stop-loss) คือ advice แม้ไม่มีตัวเลข"
- **Prompt-only guardrail มี diminishing returns สำหรับ open-ended avoidance task:** ปัญหาที่มีคำตอบ "ถูก" ทางเดียว (เช่น correlation misconception) — principle-based guardrail converge ได้เร็ว (1-2 รอบ) แต่ปัญหาที่ต้อง "ห้ามพูดถึงสิ่งหนึ่งในทางบวกทุกรูปแบบ" (เช่น stop-loss) มีวิธีหลบได้นับไม่ถ้วน (direct command → soft endorsement → indirect framing → listing-as-example → ...) แต่ละรอบ patch ปิดช่องเดิมแต่เปิดช่องใหม่ สัญญาณว่าต้อง**เปลี่ยนเครื่องมือ** ไม่ใช่ patch ต่อ — ใช้ deterministic post-processing filter เป็น safety net คู่กับ prompt (defense-in-depth) แทนพึ่ง LLM compliance อย่างเดียว
- **String-matching guardrail ต้อง normalize Unicode variants เสมอ:** deploy keyword filter ครั้งแรกไม่ trigger เลยทั้งที่ keyword อยู่ในข้อความจริง เพราะ LLM output ใช้ `U+2011` (NON-BREAKING HYPHEN) ไม่ใช่ ASCII hyphen (`U+002D`) — ASCII-only keyword list พลาด Unicode variant ได้ง่าย ต้อง normalize ก่อน match เสมอ (เช่น hyphen variants, full-width characters, zero-width characters) verify ด้วย `repr()` ทีละ codepoint ก่อนเชื่อว่า logic ผิดหรือ filter ไม่ trigger
- **Emergent tool chaining ดูดี แต่ไม่ควรพึ่งเป็น design:** agent อาจหา workaround เองได้ (เช่น เอา market value ไป proxy เป็น input ให้ tool อื่นเพื่อได้ field ที่ขาด) — แม้ deterministic ใน practice (verify 3 รอบ ผลตรงกันหมด) แต่เป็น behavior ที่ไม่ได้ specify ไว้ เสี่ยงเปลี่ยนตาม model version โดยไม่มีสัญญาณเตือน ทางที่ปลอดภัยกว่าคือ implement field ที่ขาดตรงในจุดที่ data มีอยู่แล้ว ไม่ปล่อยให้ agent ต้อง "เดา" วิธี chain เอง
- **pytest จาก root ต้อง set `PYTHONPATH=.` ก่อนรัน** — ไม่งั้น `ModuleNotFoundError: No module named 'src'` เพราะไม่มี `conftest.py` หรือ `pythonpath` config ใน `pyproject.toml`
- **Keyword blocklist เปราะกว่า principle-based guardrail:** ห้ามคำว่า "time-based" → model หลบไปใช้ "สัดส่วนของความเคลื่อนไหวทั้งหมด" แทน (misconception เดิม คำพูดใหม่) วิธีที่ทนทานกว่าคือเขียน guardrail ที่อธิบาย **ทำไม** การ decompose เป็น proportion ผิด (correlation วัด relationship ทั้ง dataset ไม่ใช่ค่าที่แยกเป็น "กรณีที่ตรงกัน X% กรณีที่ไม่ตรงกัน Y%") แทนการ list คำต้องห้ามทีละคำ — verify ด้วยการอ่าน semantic ของคำตอบทั้งประโยค ไม่ใช่แค่ grep หาคำที่เคย fail

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

### Production Migration ✅ Complete
- [x] src/config.py — env loading, LangSmith client/tracer
- [x] src/tools/price.py — get_stock_price
- [x] src/tools/financials.py — get_stock_financials
- [x] src/tools/hurst.py — get_hurst_exponent (v1.5: Rolling Hurst + IR + IC)
- [x] src/tools/portfolio_risk.py — analyze_portfolio_risk (v1.5 metrics complete + tz mismatch bug fixed)
- [x] src/tools/news.py — search_market_news
- [x] src/tools/portfolio_track.py — track_portfolio
- [x] src/database/models.py + session.py
- [x] src/agent/prompts.py + core.py (+ persona separation fix)
- [x] src/api/schemas.py + routes.py
- [x] main.py — app assembly + uvicorn entry point + UTF-8 stdout reconfigure
- [x] End-to-end test ทั้ง 4 endpoints ผ่าน Swagger UI — verified ด้วย response จริง (Alpha/Beta ไม่ N/A, Thai text ถูกต้อง)
- [x] UC-news (search_market_news) verified — citation annotations จริงจาก web_search_preview, ไม่ hallucinate
- [x] Routing case 5 verified — behavior เหมือนเดิม ไม่ regress
- [x] Error handling (invalid/delisted ticker) verified — แยก "invalid ticker" vs "transient API failure" ถูกต้อง, ไม่ throw 500
- [x] Dockerfile — multi-stage build (uv + python:3.11-slim), build+run+test ผ่านจริงใน container
  (health check 200, agent+LangSmith ทำงานจริง, trace_id เป็น UUID7 จริง, Thai encoding ถูกต้องบน Linux base)

### Streamlit UI ✅ Complete
- [x] streamlit_app.py — 3 tabs (ถามทั่วไป / วิเคราะห์ Risk พอร์ต / ติดตามพอร์ต)
- [x] Quick-question buttons ทุก tab
- [x] Two-tier display (`_split_response` — summary + expander)
- [x] Dynamic ticker/position rows ด้วย `st.session_state`
- [x] Connection error handling ("เปิด uvicorn main:app ก่อนใช้งาน")
- [x] Verified ผ่าน browser จริง (ไม่ใช่แค่ยิง endpoint ตรง) — เจอและแก้ 2 bugs:
  - `_split_response` เลือก disclaimer สั้นแทน summary จริง — แก้ filter จาก `startswith("#")` เป็น `is_bare_header` + raise threshold เป็น 60 chars
  - `ticker_in_query` dead condition (`False or False → None`) — ลบทิ้ง ส่ง ticker ตรง + ชี้แจงว่า ticker field มีผลแค่ LangSmith metadata ไม่ใช่ agent context

### Risk Contribution Analysis ✅ Complete (tool logic + guardrail)
- [x] `_portfolio_risk_logic` section 10 — Risk Contribution to Variance
- [x] `_track_portfolio_logic` section 5 — Risk Contribution (MV-weighted) + dead ticker exclusion
- [x] scripts/test_risk_contribution.py — standalone test (ไม่ผ่าน notebook)
- [x] Verified ทั้งสอง tools: sanity check sum=100.0000%, ผลตีความถูกตามทฤษฎี (high-vol ticker contribute risk มากกว่า weight ตัวเอง), dead ticker handling ถูกต้อง
- [x] System prompt guardrail — ห้าม directive "ควรขาย/ต้องลด" ตรง — verified: AMD 79.4% contribution อธิบายถูกโดยไม่มี directive
- [x] System prompt guardrail (round 1a) — ห้าม stop-loss/hedging action แม้ไม่ระบุราคา — verified กับ analyze_portfolio_risk
- [x] **track_portfolio compound question drift** — guardrail เดิมไม่ cover soft endorsement/indirect framing — prompt iteration 3 รอบ (1b/1c) ไม่ converge → เปลี่ยนเป็น **deterministic post-processing filter** (`_filter_stoploss()` ใน `src/agent/core.py`) + Unicode hyphen normalization (`U+2011` bug) — verified 3/3 รอบ, filter trigger ทุกครั้ง (เป็น safety net ที่จำเป็นจริง)
- [x] Regression retest หลังเพิ่ม guardrail ทุกรอบ — 10/11 passed
  (รอบ 2 เจอ 9/11 ชั่วคราว — isolate case 10 รัน 3x ยืนยันเป็น LLM fluke; รอบ 1d เจอ test_case5_consistency fail ด้วย Groq 429 — infrastructure ไม่ใช่ regression)

### Remaining
- [x] Regression test 11 cases หลังเพิ่ม v1.5 metrics — `tests/test_routing_regression.py`,
  10/11 passed (case 5 known limitation, deterministic over-fetch ยืนยันด้วย 5x consistency check)
- [x] Risk Contribution guardrail ใน system prompt + regression retest
- [ ] README update (เพิ่ม Risk Contribution + Streamlit ในเอกสาร)
- [ ] Colab badge (ถ้าต้องการ)

### Backlog (non-blocking)
- [ ] `create_react_agent` deprecation warning (LangGraph v1.0 moved to `langchain.agents`) —
  `src/agent/core.py:37` — fix แล้วต้องรัน `tests/test_routing_regression.py` ซ้ำก่อน merge

---

## Production Migration

### Strategy
- **tools / db / agent**: migrate logic ตรงๆ จาก notebook — logic เหมือนเดิม แค่เปลี่ยน import (+ bugfix ที่พบหลัง migrate)
- **API**: implement ใหม่ตาม "FastAPI Target Spec" — ไม่มี Cell G/H ให้ migrate แล้ว (ลบออกจาก notebook)
- **Priority**: tools → database → agent → API → Streamlit UI → Risk Contribution

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
| SPY fetch ผ่าน `provider.get_history()` (tz-aware) | `yf.download()` ตรง (tz-naive) — match กับ portfolio data fetch |
| `MOCK_PORTFOLIOS` dict | SQLite query จริงผ่าน `_load_positions_async()` + `_list_portfolio_ids()` |

### Target file structure
```
src/
├── config.py            ✅ done
├── tools/
│   ├── price.py         ✅ done
│   ├── financials.py    ✅ done
│   ├── hurst.py         ✅ done (v1.5)
│   ├── portfolio_risk.py    ✅ done (v1.5 + tz fix + Risk Contribution)
│   ├── news.py               ✅ done
│   └── portfolio_track.py    ✅ done (+ Risk Contribution)
├── database/
│   ├── models.py        ✅ done
│   └── session.py       ✅ done
├── agent/
│   ├── prompts.py       ✅ done (+ persona separation)
│   └── core.py          ✅ done
└── api/
    ├── schemas.py       ✅ done
    └── routes.py        ✅ done
main.py                  ✅ done
Dockerfile               ✅ done
tests/
└── test_routing_regression.py   ✅ done — 11 cases
scripts/
└── test_risk_contribution.py    ✅ done — standalone, ไม่ผ่าน notebook
streamlit_app.py         ✅ done — 3 tabs
```

---

## System prompt fix — UC-2a/UC-2b persona separation (✅ done)

**ปัญหาที่พบ:** `analyze_portfolio_risk` (what-if, ไม่มี cost basis) พูดว่า "พอร์ตของคุณเคยขาดทุน" — misattribution เพราะ tool ไม่รู้ว่า user ถือจริงเมื่อไหร่/ราคาเท่าไหร่ และ agent เคยคำนวณ self-generated stress test (เช่น "ตลาดตก 10% × Beta 2.48 = พอร์ตตก 25%") ซึ่งเป็น false-precision — ตัวเลขไม่มาจาก tool โดยตรง

**Fix:** เพิ่ม 2 กฎใหม่ใน `src/agent/prompts.py` (เขียนเป็นภาษาอังกฤษ แทรกแบบ mixed-language กับของเดิม — negative instruction ทำงานแม่นกว่าในอังกฤษสำหรับ gpt-oss-120b โดยไม่ต้องแปลทั้ง prompt และไม่เสี่ยง regression จาก language shift):
1. UC-2a ห้ามพูด "ของคุณ"/"your portfolio lost" — ใช้ "พอร์ตสมมตินี้"/"if held during the period" เสมอ
2. UC-2a ห้ามคำนวณ what-if scenario เอง (Beta × shock %) — รายงาน Beta/Correlation ตรงจาก tool แล้วอธิบายเชิงคุณภาพเท่านั้น
3. UC-2b ยังใช้ "ของคุณ"/"you are up/down" ได้ปกติ — มี real cost basis รองรับ

**Verified:** ทดสอบ query เดิมที่เจอ bug ผ่านครบ 3 จุด (ไม่มี "ของคุณ", ไม่มี self-generated stress number, ยังอธิบาย metrics ครบ) — รัน `tests/test_routing_regression.py` ซ้ำยืนยันไม่กระทบ routing (10/11 passed เหมือนเดิม)

---

## Risk Contribution Analysis (✅ Complete — tool logic + guardrail)

**Why:** User ต้องการรู้ว่า "หุ้นตัวไหนในพอร์ตเป็นตัวเพิ่มความเสี่ยงมากสุด" เพื่อช่วยตัดสินใจปรับพอร์ต — ปัจจุบันมีแค่ per-ticker volatility เดี่ยวๆ ซึ่งไม่บอก contribution ต่อ **portfolio-level risk** จริง (เพราะไม่ได้คำนึง correlation/weight ร่วมกัน)

**Formula (มี mathematical backing ชัดเจน — ไม่ใช่ pseudo-quant แบบ fixed factor weight ที่ปฏิเสธไปก่อนหน้า):**
```
marginal_contribution_to_risk[i] = weight[i] × cov(asset_i, portfolio) / portfolio_variance
```
ผลรวมของทุก ticker = 100% ของ portfolio variance — เป็น decomposition ที่ exact ไม่ใช่ approximation

**Implementation:** ทำตรงใน `src/` ทั้งสอง tools (ไม่ผ่าน notebook — ดู "Notebook ไม่ sync" ด้านบน) เพราะ Alpha/Beta tz-bug + persona separation fix มีแค่ใน `src/` เท่านั้น

**Verified results:**

| Tool | Test case | Weight | Risk Contribution | Sanity check |
|---|---|---|---|---|
| `analyze_portfolio_risk` | NVDA 5000 + AMD 3000 | NVDA 62.5% / AMD 37.5% | NVDA 46.1% / AMD 53.9% (AMD vol สูงกว่า) | sum = 100.0000% |
| `track_portfolio` | streamlit-test-001 (NVDA+TSLA) | NVDA 50.7% / TSLA 49.3% | NVDA 43.4% / TSLA 56.6% (TSLA vol สูงกว่า) | sum = 100.0000% |
| `track_portfolio` | dead ticker (NVDA + ZZZFAKE999) | NVDA 100% (excluded) | "100.0% (single asset)" | graceful, warning แสดงครบ |

**Scope แยกตาม 2 tools (ต้องคุม wording ตาม persona separation ที่ทำไปแล้วข้างบน):**

| Tool | เพิ่มอะไร | คำที่ใช้ได้ | คำที่ห้ามใช้ |
|---|---|---|---|
| `analyze_portfolio_risk` (UC-2a) | Risk contribution % ต่อ ticker | "AMD มี risk contribution สูงสุด — ถ้าต้องการลดความเสี่ยงรวม การลดสัดส่วนนี้คือทางหนึ่ง" | "ควรขาย AMD" (เป็น advice ตรง + พอร์ตยังไม่มีจริง) |
| `track_portfolio` (UC-2b) | Risk contribution % + เชื่อมกับ existing unrealized P&L per position | "AMD ขาดทุน -12% และเป็นตัว contribute ความผันผวนสูงสุด — อาจพิจารณาทบทวนสัดส่วนนี้" | "ควรขาย AMD ทันที" (ยังเป็น advice ตรงเกินไป แม้มี cost basis จริง) |

**✅ Guardrail ใน system prompt (implemented + verified):** ห้ามแปล risk contribution number เป็น directive คำสั่ง ("ควรขาย/ต้องขาย") — ให้รายงานตัวเลข + อธิบายความหมาย แล้วปล่อยให้ user ตัดสินใจเอง ตรงกับ constraint ที่มีอยู่แล้ว ("Do not give specific price targets, entry points, or stop-loss levels", "Note that this is not financial advice")

**Verified (query: "หุ้นตัวไหนเสี่ยงสุดในพอร์ตนี้ ควรทำยังไงดี"):**
- AMD 79.4% risk contribution รายงานถูกพร้อมคำอธิบาย (volatility สูงกว่า weight ที่ถือ)
- ไม่มี "ควรขาย AMD" หรือ "ต้องลด AMD" แบบสั่งตรง — ใช้ "อาจพิจารณา", "เป็นทางหนึ่ง" แทน

**Drift ที่เจอและแก้เพิ่ม (round 1a — stop-loss, initial fix):** รอบแรกพบ agent แนะนำ "ตั้ง stop-loss"/"ใช้ hedging" หลุดออกมา — ละเมิด rule เดิมที่มีมาตั้งแต่ v1 ("Do not give specific price targets, entry points, or stop-loss levels") ไม่ใช่ปัญหาจาก guardrail ใหม่ แก้โดยเสริม negative example ชัดเจนเข้า rule เดิม ("suggesting the *action* of stop-loss/hedging is itself actionable advice, even without a price") — verified ผ่านตอนนั้นด้วย query เดี่ยวที่ถามตรงๆ เรื่อง stop-loss

**Drift ที่เจอใหม่ (round 1b — track_portfolio + compound question, prompt-only ไม่พอ):** ทดสอบ `track_portfolio` ด้วย compound question ("AMD เสี่ยงสุดไหม **และ** ควรตั้ง stop-loss ไหม") เจอ drift กลับมา — guardrail เดิม cover แค่ direct command ("ตั้ง stop-loss ที่ราคา X") แต่ไม่ cover **soft endorsement** ("stop-loss สามารถช่วยจำกัดการขาดทุนได้") และ **indirect framing** ("คุณอาจพิจารณาว่า stop-loss สอดคล้องกับ risk tolerance") ยืนยันด้วย isolate test 3 รอบ — ไม่ deterministic (2/3 fail ด้วย pattern ต่างกัน) ไม่ใช่ fluke

**Iteration ที่ลองและไม่พอ (round 1c — prompt-only, NEVER + MUST + exact format):** เปลี่ยน "do NOT say" → "NEVER", เพิ่ม exact failure phrase เป็น negative example, บังคับ explicit refusal ด้วย `MUST` + fixed format string — รัน 3 รอบซ้ำ ปิดช่องเดิมได้ (soft endorsement แบบเดิมหายไป) **แต่เปิดช่องใหม่ทันที**: model ใช้ "listing-as-example" (เอา stop-loss ปนใน list ร่วมกับ rebalancing — "การตัดสินใจว่าจะใช้เครื่องมือใด เช่น stop-loss, hedge") implicit ว่าเป็น valid option โดยไม่ต้อง endorse ตรง — 3 รอบนี้แต่ละรอบ fail คนละ pattern (R1 paraphrase ไม่ตรง exact format, R2 silent omission, R3 borderline wording)

**Decision point:** prompt iteration ปิดช่องหนึ่งแล้วเปิดช่องใหม่ทุกรอบ (3 รอบ, 3 pattern ต่างกัน) — ต่างจาก correlation misconception (round 2 ด้านล่าง) ที่เป็น **fixed factual error** และ converge ได้ใน 2 รอบ เพราะมีคำตอบ "ถูก" ทางเดียว ส่วน "ห้ามพูดถึง stop-loss ในทางบวก" เป็น **open-ended avoidance task** ที่มีวิธี "บวก" ได้นับไม่ถ้วน — สัญญาณ diminishing returns ของ prompt-only approach ชัดเจน ตัดสินใจเปลี่ยนเป็น **defense-in-depth**: prompt guardrail (ลดโอกาส) + deterministic post-processing filter (รับประกัน)

**Fix (round 1d — deterministic post-processing filter):** เพิ่ม `_filter_stoploss()` ใน `src/agent/core.py` — scan response แยก paragraph, ลบ paragraph ที่มี stop-loss/hedging keyword ออก, แทนด้วย fixed refusal statement, log ทุกครั้งที่ trigger (`WARNING: [stoploss-filter] prompt guardrail leaked`) ทำงานหลัง `run_financial_agent()` ได้ response กลับมา ก่อน return — ไม่พึ่งความสม่ำเสมอของ LLM อีกต่อไป

**Root cause ที่ซ่อนอยู่ใน filter รอบแรก (สำคัญ):** deploy filter รอบแรกแล้ว**ไม่ trigger เลยทั้ง 3 รอบ** ทั้งที่ response มีคำว่า "stop-loss" อยู่จริง — เพราะ model output ใช้ **`U+2011` (NON-BREAKING HYPHEN)** ไม่ใช่ ASCII hyphen (`U+002D`) ทำให้ string match ไม่เจอ verify ด้วย `repr()` ทีละ codepoint ก่อนเชื่อว่า logic ผิด แก้โดยเพิ่ม Unicode hyphen normalization (`_normalize_hyphens()`) ก่อน keyword check — เป็นบทเรียนว่า **string-matching guardrail ต้อง normalize Unicode variant เสมอ** ไม่ใช่แค่ ASCII keyword list

**Verified (final, 3 layers ทำงานร่วมกัน):** 3/3 รอบผ่านครบ 3 จุด (ไม่มี endorsement, exact refusal statement ปรากฏ, risk analysis AMD 94.5% ยังครบ) filter trigger ทุกรอบ (prompt guardrail ยังหลุดอยู่ — filter เป็น safety net ที่จำเป็นจริง ไม่ใช่ redundant) regression: 10/11 passed (`test_case5_consistency` fail ด้วย Groq 429 rate limit — infrastructure issue ไม่ใช่ regression จาก code)

**Drift ที่เจอและแก้เพิ่ม (round 2 — correlation/diversification statistical misstatement):** พบ 2 statistical misconception แยกจาก Risk Contribution โดยตรง แต่เจอตอน verify คำตอบ portfolio risk เดียวกัน:
1. Agent อธิบาย correlation 0.5 เป็น "เคลื่อนที่เหมือนกันประมาณครึ่งหนึ่งของเวลา/ของความเคลื่อนไหวทั้งหมด" — ผิดทางสถิติ correlation วัด linear relationship strength ไม่ใช่ proportion/frequency ของการเคลื่อนไหว
2. Agent ใช้คำว่า "Diversification = 0" — false-precision เพราะ tool ไม่มี diversification metric เลย (มีแค่ correlation matrix + risk contribution)

**Fix iteration:** guardrail รอบแรกห้ามแค่ "time-based" claim — model หลบคำว่า "เวลา" ไปใช้ "สัดส่วนของความเคลื่อนไหว" แทน (misconception เดิม คำพูดใหม่) ต้องเขียน guardrail ใหม่แบบ **principle-based** (อธิบายว่าทำไม correlation ไม่ decompose เป็น proportion ได้ ไม่ใช่ list คำต้องห้ามทีละแบบ) — verified ผ่านหลังแก้: agent ใช้ "moderate positive relationship", "บ้างแต่ไม่เต็มที่" ล้วน ไม่มี proportion/frequency claim ในรูปแบบใดเลย ยังอธิบาย mechanism ถูกได้ (เช่น "ถ้า correlation สูงขึ้น variance จะเพิ่มขึ้น" — เชิงทิศทาง ไม่ใช่ตัวเลขทำนาย) แสดงว่า principle-based guardrail ไม่ทำให้คำตอบจนเนื้อหา แค่กรองส่วนที่ผิดจริง

**Regression note:** หลังเพิ่ม guardrail รอบสอง เจอ 9/11 ชั่วคราว (case 10 — multi-tool query ขาด `search_market_news`) — isolate รัน case 10 เดี่ยว 3 รอบ ได้ tools ครบทุกรอบ ไม่มี ticker แปลกปลอม ("NVNV" ที่เคยเห็นตอน fail) ยืนยันเป็น LLM nondeterministic fluke ไม่ใช่ regression จาก guardrail — final regression: 10/11 passed

**Gap ที่เจอตอน test track_portfolio (UC-2b) ด้วย guardrails เดียวกัน:** ทุก guardrail ข้างบนทดสอบผ่าน `analyze_portfolio_risk` (what-if) เท่านั้น — ทดสอบ adversarial 3 query เดียวกันกับ `track_portfolio` (sell-directive, stop-loss, correlation/diversification) พบว่า**ผ่านครบทั้ง 3** เพราะ system prompt เขียน guardrail ครอบคลุม "both tools" อยู่แล้ว แต่เจอ **emergent behavior ที่ไม่ได้ตั้งใจ**: ตอนถาม correlation ของ tracked portfolio, agent เรียก `track_portfolio` ก่อน แล้ว **เอา market value มา proxy เป็น "amount" ส่งเข้า `analyze_portfolio_risk` เอง** เพื่อเอา correlation matrix (เพราะ `track_portfolio` เดิมไม่มี correlation) — ไม่มีใครสั่งให้ agent ทำแบบนี้ เป็นการเดาเองจาก reasoning

Verify ด้วย LangSmith trace 3 รอบ (consistency test) — agent ทำ tool chain แบบเดียวกันทุกรอบ (deterministic ในทางปฏิบัติ) ตัวเลขสอดคล้องกันทุกรอบ (Pearson 0.39, Rolling 0.46) **แต่** ตัดสินใจไม่พึ่ง emergent behavior นี้ต่อไป เพราะ (1) MV-as-amount เป็น approximation ไม่ exact กับ weight ที่ `track_portfolio` ใช้จริง และ (2) behavior ไม่ได้ specify ไว้ในระบบ — เสี่ยงเปลี่ยนถ้า model version เปลี่ยนโดยไม่มีสัญญาณเตือน

**Fix:** เพิ่ม Pearson correlation (1Y) + Rolling correlation (60d) เข้า `_track_portfolio_logic` ตรง (section 5) — reuse `hist_returns` ที่คำนวณไว้แล้วสำหรับ Risk Contribution อยู่แล้ว ไม่ fetch ข้อมูลเพิ่ม Single-asset fallback ("N/A (single asset)") ตรงกับ pattern เดิมของ Risk Contribution

**Verified หลังแก้:**
- Sanity: Pearson 0.41, Rolling 60d 0.46 (ต่างจาก emergent chaining เดิม 0.39/0.46 เล็กน้อย — data fetch คนละช่วงเวลา ไม่ใช่ bug)
- LangSmith trace: correlation query เหลือ tool call เดียว (`track_portfolio` only — ไม่มี `analyze_portfolio_risk` chain อีกแล้ว)
- Regression: 2/2 passed (`PYTHONPATH=.` ต้อง set ก่อนรัน pytest จาก root — `ModuleNotFoundError: No module named 'src'` ถ้าไม่ set)

---

## Streamlit UI (✅ done)

**Architecture:** เรียกผ่าน FastAPI ที่มีอยู่ (HTTP request → `localhost:8000` ผ่าน `requests` library) — ไม่เรียก agent ตรงจาก Streamlit เพื่อให้ FastAPI ยังเป็น single source of truth ของ business logic

**3 tabs (`streamlit_app.py`):**

| Tab | Endpoint | Fields |
|---|---|---|
| ถามทั่วไป | `POST /analyze/stock` | ticker (สำหรับ quick-buttons) + query (free text) |
| วิเคราะห์ Risk พอร์ต (what-if) | `POST /analyze/portfolio` | dynamic ticker + **จำนวนเงิน** rows + query เสริม (ไม่บังคับ) |
| ติดตามพอร์ต | `POST /portfolio/positions` (สร้าง) + `GET /portfolio/{id}` (ดู) | dynamic ticker + จำนวนหุ้น + ราคาเฉลี่ยที่ซื้อ rows |

**Why Portfolio Risk ไม่ต้องมีจำนวนหุ้น:** `analyze_portfolio_risk` คำนวณจาก weight (`amount / total_amount`) ไม่ใช่จำนวนหุ้นจริง — เป็น what-if ก่อนซื้อ ไม่ใช่ของที่ถือแล้ว ต่างจาก Portfolio Tracking ที่ต้องมี `shares` + `avg_cost` จริงเพราะคำนวณ unrealized P&L จากต้นทุนจริง

**Important UX clarification (พบตอน browser testing):** Ticker field ใน Tab "ถามทั่วไป" มีผลแค่กับ quick-question buttons (insert ticker เข้า pre-filled query) — **ไม่ได้ถูกส่งเข้า agent context** สำหรับ free-text queries เพราะ `ticker` ใน API request ไปอยู่แค่ LangSmith metadata/tags (`src/api/routes.py`) ไม่ผ่านเข้า `run_financial_agent(query, ...)` เลย — ถ้า user พิมพ์ query ที่ไม่มีชื่อ ticker ในข้อความ agent จะไม่รู้จักหุ้นนั้น (ถูกต้องตาม design ไม่ใช่ bug) — label ใน UI ชี้แจงเรื่องนี้ชัดแล้ว

### Scope ที่ไม่รองรับ (ป้องกัน scope creep ที่ UI layer)

ระบบ**ไม่มี** what-if scenario engine หรือ cross-asset causal analysis — คำถามแบบนี้จะถูกป้องกันด้วย **helper text ที่ UI** (ไม่ใช่ agent classify เอง เพราะเพิ่ม LLM call ที่ไม่ predictable และไม่ประหยัด token จริง):

- ❌ "ถ้าราคา AMD ตก 20% ความเสี่ยงพอร์ตจะเป็นไง" — ไม่มี stress test module (เป็น v2 backlog: correlation-based stress test) — **เสริมด้วย system prompt fix แล้ว** (agent ปฏิเสธคำนวณเองถ้าถูกถามตรง)
- ❌ "ถ้า Intel ฟื้นตัวจะกระทบพอร์ตยังไง" — ไม่มี causal cross-asset model (out of scope เพราะ false-precision risk เดียวกับ factor engine)
- ✅ "เน้นอธิบายเรื่อง correlation/drawdown" — ได้ เพราะเป็นการตีความข้อมูลที่ tool คำนวณอยู่แล้ว ไม่ต้องคำนวณใหม่
- ✅ "หุ้นตัวไหนเสี่ยงสุดในพอร์ต" — ได้แล้วหลัง Risk Contribution Analysis (ดูด้านบน)

**No conversation memory ข้าม turn** — แต่ละ query ต้อง self-contained (พิมพ์ ticker ครบทุกครั้ง, ห้ามอ้าง "ตัวที่ถามไปแล้ว") เพราะ `run_financial_agent()` เป็น stateless และ DB ไม่เก็บ conversation history Streamlit อาจเก็บ history ไว้แสดงผลใน `st.session_state` แต่**ไม่ส่งกลับเข้า agent**

### UX สำหรับ user ที่ไม่รู้ศัพท์การเงิน (✅ implemented)

1. **Quick-question buttons** แทนกล่อง text เปล่า ลด barrier ตอนเริ่มถาม — ทุก tab มีครบ (4 ปุ่ม Tab 1, 3 ปุ่ม Tab 2)
2. **Two-tier display**: สรุปภาษาง่าย (`_split_response` ดึงจาก paragraph สุดท้ายที่ไม่ใช่ table/bare-header/disclaimer สั้น) แสดงเด่นใน `st.success`, ตัวเลขเทคนิคเก็บใน `st.expander("รายละเอียดเชิงเทคนิค")` — verified ผ่าน 2 response shape (price-only สั้น, full multi-tool analysis ยาว)

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
- Correlation-based stress test + What-if เพิ่ม asset นอกพอร์ต — ดู "Streamlit UI — v2 backlog" ด้านบน
- Conversation memory ข้าม turn — ดู "Streamlit UI — v2 backlog" ด้านบน
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