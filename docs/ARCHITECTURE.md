# Architecture

## Tech stack {#tech-stack}

| Layer | Technology | หมายเหตุ |
|---|---|---|
| LLM (dev) | Groq — `openai/gpt-oss-120b` | `reasoning_effort="low"`, tool orchestration เสถียร |
| LLM (prod) | Gemini 2.5 Flash | swap ตอน deploy |
| LLM (news) | OpenAI `gpt-4o-mini` | `_news_model` ใน `search_market_news` — Gemini free tier 20 req/day หมดเร็ว |
| Agent framework | LangGraph custom `StateGraph` (planner → agent → tools) | replaces deprecated `create_react_agent` — ดู `#agent-framework` |
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

## System diagram

```
User query
    ↓
run_financial_agent() @traceable → return trace_id (run_id)
    ↓
StateGraph: planner → agent ⇄ tools (loop) → END
    │
    ├── planner node: _plan_override(query) — deterministic portfolio_id match
    │     match  → plan = ["track_portfolio"]                (case 12 fix, skips LLM)
    │     no match → LLM planner (temp=0, structured output) classifies plan
    │
    ├── agent node: model (ALL tools bound) → reconcile tool_calls to plan
    │     • upper bound (every turn): drop off-plan calls      (case 5 fix)
    │     • lower bound (first turn): synthesize omitted planned calls (case 10 fix)
    │
    └── tools node (ToolNode) — executes only reconciled calls
          ├── get_stock_price            → YFinanceProvider → 5d history
          ├── get_stock_financials       → YFinanceProvider → .info
          ├── get_hurst_exponent         → YFinanceProvider 1Y + numpy R/S + Rolling Hurst + IC + IR
          ├── analyze_portfolio_risk     → YFinanceProvider 1Y + numpy/pandas (amount-based) + Risk Contribution
          ├── track_portfolio            → SQLite + YFinanceProvider 5d + Risk Contribution
          └── search_market_news         → OpenAI gpt-4o-mini (Gemini quota: 20 req/day)
    ↓
_filter_stoploss() — deterministic post-processing safety net
    ↓
LangSmith (traces ทุก step)
    ↓
FastAPI (5 endpoints)
    ↓
Streamlit UI (3 tabs) — calls FastAPI via requests
```

---

## Agent framework — StateGraph routing (v2 Phase 1) {#agent-framework}

**เหตุผลที่เปลี่ยนจาก `create_react_agent`:** case 5 (query "P/E กับ profit margin ของ AMD"
over-fetch `get_stock_price`) เป็น model-level semantic association ที่ docstring/prompt-level
negative routing แก้ไม่ได้ (ดู `docs/POSTMORTEMS.md#docstring-routing`) ต้อง structural fix แทน
พ่วงแก้ `create_react_agent` deprecation warning ไปด้วยเพราะแตะไฟล์เดียวกันอยู่แล้ว

**Design — planner → agent → tools loop** (`src/agent/core.py::build_agent`):

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]        # planned tool names for this turn
```

1. **planner node** — `_plan_override(query)` ก่อนเสมอ: word-boundary match query กับ
   `_list_portfolio_ids()` (DB) → เจอ id จริง = force `plan = ["track_portfolio"]`, ข้าม LLM
   planner ทั้งหมด (deterministic, ฟรี token) — แก้ **case 12**: "พอร์ต {id} + risk phrasing"
   เคย misroute ไปทาง `analyze_portfolio_risk` เพราะ risk/stop-loss wording เป็น semantic signal
   ที่แรงกว่า id ในสายตา LLM planner ไม่ match → LLM planner (temp=0, `.with_structured_output`)
   classify tool subset จาก `PLANNER_PROMPT` (แยกจาก `SYSTEM_PROMPT` โดยสิ้นเชิง — ไม่ถือว่าแก้
   SYSTEM_PROMPT) เขียนแค่ `state["plan"]` ไม่แตะ `messages` (กัน structured-output tool call
   หลุดเข้า message count ที่ test นับ tool จาก)
2. **agent node** — bind **ALL** tools เสมอ (bind subset ทำให้ Groq 400 ตอน model พยายามเรียก
   tool นอก request — ดู journey ด้านล่าง) แล้ว reconcile tool_calls ให้เท่ากับ plan:
   - **upper bound (ทุก turn)** — drop tool_calls นอก plan → แก้ over-fetch (case 5) ต้องรันทุก
     turn ไม่ใช่แค่ turn แรก เพราะ P/E semantic pull กลับมา re-request `get_stock_price` บน
     synthesis turn ได้เช่นกัน (case-5 gap ที่เจอตอน verify)
   - **lower bound (first turn เท่านั้น)** — synthesize tool_call ที่ planned ไว้แต่ model ข้าม
     ไป โดยยืม args จาก sibling call/query (deterministic, ไม่มี LLM call เพิ่ม) → แก้ under-call
     (case 10: `search_market_news` หลุดใน 4-tool query)
3. **tools node** — `ToolNode(ALL_tools)` execute เฉพาะ tool_calls ที่ผ่าน reconcile

**Fallback:** planner exception หรือคืน list ว่าง → `plan = ALL tools` (revert เป็น greedy เดิม —
ปลอดภัยกว่า under-trigger)

**Journey — approach ที่ลองแล้วพัง (สรุปสั้น, เต็มอยู่ `docs/POSTMORTEMS.md#docstring-routing`
และ git history ของ `docs/v2-stategraph-routing.md` ตอนยังไม่ลบ):**
1. bind เฉพาะ planned tool → Groq 400 (model ยังพยายามเรียก tool นอก request จริง) → ต้อง bind ครบ
2. prompt/param nudges (directive, temp, reasoning_effort) ไม่พอสำหรับ under-call → ต้อง
   deterministic synthesis
3. coverage-gated loop (loop กลับจน plan ครบ) → เพิ่ม LLM call ต่อ query จนชน Groq TPD limit
   ระหว่าง test → ตัดทิ้ง เปลี่ยนเป็น deterministic construction (final design ข้างบน)

**PLANNER_PROMPT** (`src/agent/core.py`) — แก้ต้องรัน `tests/test_routing_regression.py`
(13 cases) ซ้ำเหมือนกับ `SYSTEM_PROMPT`:

```python
PLANNER_PROMPT = """You are a routing planner for a financial analysis agent.
Given the user's query, decide EXACTLY which tools are required to answer it.
...
Critical routing rules:
1. P/E ratio, valuation multiples, and profit margin are FUNDAMENTALS →
   get_stock_financials ONLY. Do NOT add get_stock_price.
...
"""
```

ดูฉบับเต็มใน `src/agent/core.py` — เก็บสำเนาเต็มไว้ที่นี่จะซ้ำซ้อนกับ `#system-prompt` section
ด้านล่าง เพราะเป็น prompt คนละก้อนที่ไม่ควรสับสนกัน (`PLANNER_PROMPT` ไม่เคยถูกส่งเข้า agent node)

**Interface ที่ต้องคงไว้ (caller contract):** `build_agent()` ต้องรองรับ
`.stream(inputs, config, stream_mode="values")` และ emit `AIMessage.tool_calls` ลง
`state["messages"]` เพื่อให้ `tests/test_routing_regression.py` และ `run_financial_agent()`
ทำงานเหมือนเดิม — ไม่เปลี่ยน public interface ทั้งสอง

**Token cost trade-off:** planner เพิ่ม LLM call ต่อ query (~2x เทียบกับ ReAct เดิม) ยกเว้น query
ที่ `_plan_override` match (ไม่มี extra call เลย) — ยอมรับ trade-off นี้เพื่อแลก routing ที่
deterministic — ดู `docs/DECISIONS.md#routing`

---

## Data layer

### DataProvider Protocol (Cell 3.5 — v1.5)

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
- Metrics: Annualized Return/Volatility, Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%, Max Drawdown,
  Ulcer Index, Drawdown Duration, Rolling Correlation (60d), Benchmark Alpha/Beta vs SPY,
  Risk Contribution to Variance
- Tools: `analyze_portfolio_risk` (Cell F — canonical)
- **Persona:** what-if/hypothetical เท่านั้น — ห้ามพูด "ของคุณ", ห้าม self-generated stress test
  (ดู `docs/POSTMORTEMS.md#persona-separation`)

### UC-2b: ติดตาม portfolio หลังซื้อ ✅
- Input: `portfolio_id` string → load จาก SQLite
- Output: unrealized P&L per position, total MV, total P&L, current weights,
  Risk Contribution to Variance (MV-weighted)
- Tools: `track_portfolio`
- **Persona:** actual holdings — มี real cost basis รองรับ ใช้ "ของคุณ"/"you are up/down" ได้

### UC-news: ข่าว + analyst commentary ✅
- Tools: `search_market_news` → OpenAI gpt-4o-mini (Gemini free tier 20 req/day หมดเร็ว)
- Routing: 13/13 ผ่าน (ดู `docs/DECISIONS.md#routing`)

---

## Database {#database}

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

**Design principle:** เก็บเฉพาะ source-of-truth user input ที่ API ไหนก็ให้ไม่ได้ (ticker, shares,
avg_cost) — derive metric ทุกตัว on-the-fly ไม่เก็บ derived data ลง DB

---

## Tools {#tools} (สถานะจริง v1.5)

> **Source of truth:** ทุก tool migrate ไป `src/tools/*.py` แล้ว (logic เหมือนกับ notebook cell
> เดิม แต่มี bugfix เพิ่มที่ notebook ไม่มี — ดู `docs/DECISIONS.md#notebook-vs-src` สำหรับ diff)
> notebook cell ยังเก็บไว้สำหรับอ้างอิงประวัติเท่านั้น

### get_stock_price ✅ `src/tools/price.py`
```python
@traceable(name="fetch_stock_price", run_type="tool",
           tags=["market-data", "yfinance"], client=ls_client)
```
Period 5d เผื่อ market ปิด
Output: price, change%, 52W range, P/E TTM + Forward, market cap, position in 52W range

### get_stock_financials ✅ `src/tools/financials.py`
```python
@traceable(name="fetch_financials", run_type="tool",
           tags=["fundamentals", "yfinance"], client=ls_client)
```
Output: revenue, net income, profit margin, revenue growth YoY, EPS, D/E

### get_hurst_exponent ✅ `src/tools/hurst.py` (v1.5 complete)
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

### analyze_portfolio_risk ✅ `src/tools/portfolio_risk.py` (v1.5 + Risk Contribution complete)
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
- Alpha (annualized, vs SPY) + Beta vs SPY — CAPM market-model (tz-mismatch bug fixed —
  ดู `docs/POSTMORTEMS.md#bug-6-spy-tz`)
- **Risk Contribution to Variance** (section 10) — `MCR[i] = w[i] × (Σw)[i] / (w^T Σw)` —
  sums to 100% by construction — รายละเอียด verification: `docs/POSTMORTEMS.md#risk-contribution`

### search_market_news ✅ `src/tools/news.py`
```python
@traceable(name="search_market_news", run_type="tool",
           tags=["search", "news", "openai"], client=ls_client)
```
Model แยก: `_news_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)` + `web_search_preview` tool
Gemini free tier = 20 req/day หมดเร็ว → switched to OpenAI fallback ใน dev
Graceful error: API failure → คืน error string ไม่ทำ agent พัง
Verified: citation annotations จริงจาก web_search_preview, ไม่ hallucinate

### track_portfolio ✅ `src/tools/portfolio_track.py` (+ Risk Contribution complete)
```python
@traceable(name="track_portfolio", run_type="tool",
           tags=["portfolio", "tracking"], client=ls_client)
```
`_load_positions_async()` โหลดจาก SQLite จริง (ไม่ใช่ MOCK_PORTFOLIOS แบบ notebook)
Batch fetch 5d, graceful สำหรับ delisted ticker — แยก "invalid ticker" vs "transient API failure"
**Risk Contribution to Variance** (section 5, MV-weighted) — ใช้ current market value weight
**Pearson Correlation Matrix (1Y) + Rolling Correlation (60d)** (section 5) — reuse `hist_returns`
ที่คำนวณไว้แล้วสำหรับ Risk Contribution ไม่ fetch ข้อมูลเพิ่ม — รายละเอียด root cause ทำไมต้องเพิ่ม
ตรงนี้แทนพึ่ง emergent tool chaining: `docs/POSTMORTEMS.md#emergent-chaining`

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

## LangSmith tracing pattern — explicit binding {#langsmith-binding}

**ทำไมไม่ใช้ env vars:** `langsmith.utils.get_env_var` ถูก cache ด้วย `lru_cache` ตอน import ครั้งแรก
— ถ้า import ก่อน set env ค่าค้าง disabled ตลอด session

**Scope ของ "explicit binding":** หมายถึง **credential/client** เท่านั้น (`client=ls_client` ใน
`@traceable`, ไม่ใช้ env var สำหรับ API key) — ไม่ได้ครอบคลุม **tracing on/off switch** ซึ่งเป็นคนละ
concern กัน `LANGCHAIN_TRACING_V2` ยังต้องตั้งใน env เพราะเป็น switch ระดับ SDK ที่ LangGraph internal
เช็คจาก env เสมอ ไม่มี param ให้ผ่านตรง

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

**สัญญาณว่าผิดลำดับ:** `run_id` ใน `run_financial_agent()` คืน `None` แม้ agent ทำงานได้ปกติ —
ตรวจสอบด้วย end-to-end test แล้วดูค่า `run_id` ใน return value โดยตรง ไม่พึ่งแค่ "import ผ่าน"

```python
# config.py
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
- Public: `ls_client.share_run(run_id)` — ห้ามประกอบ URL เอง, link หมดอายุไม่กี่สัปดาห์ — ไม่ commit
  ใน README, ใช้ trace_id แทนเป็นหลักฐานว่า observability ทำงานจริง
- ห้าม commit trace URL (มี workspace ID)

ใน FastAPI prod: กลับใช้ env ได้เพราะ set ก่อน import ตอน container start

---

## System prompt {#system-prompt} (`src/agent/prompts.py`)

> **Source of truth คือโค้ดใน `src/agent/prompts.py`** — สำเนานี้เก็บไว้เป็น reference สำหรับ review
> การแก้ prompt โดยไม่ต้องเปิดโค้ด ถ้าแก้ prompt ในโค้ดแล้วต้อง sync สำเนานี้ด้วย
> **แก้ prompt ทุกครั้ง (SYSTEM_PROMPT หรือ PLANNER_PROMPT — ดู `#agent-framework`) → รัน
> `tests/test_routing_regression.py` (13 cases) ก่อน merge**
> ทุกบรรทัดในนี้ผ่านการ iterate/verify มาแล้ว — ก่อนแก้บรรทัดใดอ่าน `docs/POSTMORTEMS.md#guardrails`
> และ `#persona-separation` ว่าบรรทัดนั้นเกิดจาก drift อะไร

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

**Guardrail annotation (บรรทัดไหนมาจาก drift อะไร):**

| ส่วนใน prompt | ที่มา |
|---|---|
| "NEVER state any number that did not come from a tool result" | tool hallucination learning (tool หายจาก list → agent fabricate) |
| "Do not give specific price targets, entry points, or stop-loss levels" | base rule ตั้งแต่ v1 — เสริมด้วย `_filter_stoploss()` deterministic filter แล้ว (prompt อย่างเดียวไม่พอ — ดู `docs/POSTMORTEMS.md#guardrails`) |
| "call search_market_news ONLY — do not add get_stock_price" | news routing over-fetch fix |
| "If portfolio data is already provided as a JSON string..." | portfolio JSON guardrail — agent เคยถาม user ซ้ำทั้งที่ data อยู่ใน message แล้ว |
| UC-2a persona block ทั้งก้อน | Bug 7 misattribution + self-generated stress test — ดู `docs/POSTMORTEMS.md#persona-separation` |
| UC-2b persona block | คู่กันกับ UC-2a — แยกให้ชัดว่าฝั่งไหนใช้ "ของคุณ" ได้ |

หมายเหตุ: guardrail เรื่อง risk contribution directive ("ควรขาย/ต้องลด") และ correlation
misconception (principle-based) อยู่ในโค้ดจริงด้วย — สำเนาด้านบนคือ core prompt ณ จุดที่บันทึก
ถ้า diff กับโค้ดไม่ตรง ให้ยึดโค้ดแล้วอัปเดตสำเนานี้

---

## FastAPI endpoints {#fastapi-spec} (`src/api/routes.py` — implemented)

```
GET  /health
POST /analyze/stock       → {query, response, ticker, trace_id}
POST /analyze/portfolio   → {portfolio, query, response, trace_id}
                            Body: {"portfolio": {"NVDA": 5000, "AMD": 3000}}
POST /portfolio/positions → {portfolio_id, name, positions_saved}
GET  /portfolio/{id}      → {portfolio_id, response, trace_id}
```

`trace_id` = `run_id` จาก `run_financial_agent` — 1:1 กับ LangSmith

`POST /portfolio/positions` — สร้าง portfolio ใหม่ (id ยังไม่มีใน DB) ต้องผ่าน
`validate_new_portfolio_id()` (`src/api/schemas.py`): ≥5 ตัวอักษร, `[a-z0-9-]+` เท่านั้น,
ห้ามเป็นคำทั่วไปล้วน (`demo`/`test`/`port`/`portfolio`) → ไม่ผ่าน = `422`. id ที่มีอยู่แล้วใน DB
(append position เข้าพอร์ตเดิม) ไม่ต้องผ่าน validation นี้ซ้ำ (grandfathered)

**สถานะ:** implemented + verified ผ่าน Swagger UI และ Streamlit ทั้ง 5 endpoints
**Implementation note:** รันผ่าน `uvicorn main:app --reload` ปกติจาก terminal — ไม่มี threading
wrapper, nest_asyncio, หรือ ngrok

---

## Target file structure

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
└── test_routing_regression.py   ✅ done — 13 cases
scripts/
└── test_risk_contribution.py    ✅ done — standalone, ไม่ผ่าน notebook
streamlit_app.py         ✅ done — 3 tabs
```

---

## Streamlit UI

**Architecture:** เรียกผ่าน FastAPI ที่มีอยู่ (HTTP request → `localhost:8000` ผ่าน `requests`
library) — ไม่เรียก agent ตรงจาก Streamlit เพื่อให้ FastAPI ยังเป็น single source of truth ของ
business logic

**3 tabs (`streamlit_app.py`):**

| Tab | Endpoint | Fields |
|---|---|---|
| ถามทั่วไป | `POST /analyze/stock` | ticker (สำหรับ quick-buttons) + query (free text) |
| วิเคราะห์ Risk พอร์ต (what-if) | `POST /analyze/portfolio` | dynamic ticker + จำนวนเงิน rows + query เสริม |
| ติดตามพอร์ต | `POST /portfolio/positions` + `GET /portfolio/{id}` | dynamic ticker + จำนวนหุ้น + ราคาเฉลี่ยที่ซื้อ |

**Why Portfolio Risk ไม่ต้องมีจำนวนหุ้น:** `analyze_portfolio_risk` คำนวณจาก weight
(`amount / total_amount`) ไม่ใช่จำนวนหุ้นจริง — เป็น what-if ก่อนซื้อ ต่างจาก Portfolio Tracking
ที่ต้องมี `shares` + `avg_cost` จริงเพราะคำนวณ unrealized P&L จากต้นทุนจริง

**No conversation memory ข้าม turn** — แต่ละ query ต้อง self-contained เพราะ
`run_financial_agent()` เป็น stateless และ DB ไม่เก็บ conversation history

### Scope ที่ไม่รองรับ (ป้องกัน scope creep ที่ UI layer)

- ❌ stress test ("ถ้าราคา AMD ตก 20%...") — ไม่มี module นี้ (v2 backlog) — เสริมด้วย system
  prompt fix แล้ว (agent ปฏิเสธคำนวณเองถ้าถูกถามตรง)
- ❌ causal cross-asset impact (เช่น "ถ้า Intel ฟื้นตัวจะกระทบพอร์ตยังไง") — out of scope
  false-precision risk เดียวกับ factor engine
- ✅ "เน้นอธิบายเรื่อง correlation/drawdown" — ได้ เพราะตีความข้อมูลที่ tool คำนวณอยู่แล้ว
- ✅ "หุ้นตัวไหนเสี่ยงสุดในพอร์ต" — ได้แล้วหลัง Risk Contribution Analysis

ดู v2 backlog เต็มที่ `docs/DECISIONS.md#v2-backlog`

---

## Out of scope v1 → v2

ย้ายไปเป็น single source of truth ที่ `CLAUDE.md#out-of-scope` แล้ว (กันไฟล์ sync ไม่ตรงกัน) —
มีเหตุผลเต็มพร้อม note จาก external review 2 รอบที่ `docs/DECISIONS.md#external-review`