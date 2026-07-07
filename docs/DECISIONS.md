# Decisions

ไฟล์นี้เก็บ "ทำไมเลือก X ไม่เลือก Y" แบบสรุป — รายละเอียดการ debug/iterate เต็มอยู่ที่
`docs/POSTMORTEMS.md`

---

## Notebook vs src/ {#notebook-vs-src}

**Decision:** notebook คงไว้เป็น **historical reference เท่านั้น** ไม่ sync ตามทุกครั้งที่แก้ `src/`
— maintenance cost ไม่คุ้มกับประโยชน์ `src/` คือ source of truth เพียงที่เดียว

**Verified diff** (เช็คตอน implement Risk Contribution Analysis):

| จุด | Notebook | `src/` |
|---|---|---|
| `_portfolio_risk_logic` Alpha/Beta | ไม่มี tz fix — อาจได้ N/A เงียบๆ | มี tz-naive strip ก่อน intersection |
| `_track_portfolio_logic` error message | แสดง `MOCK_PORTFOLIOS.keys()` (hardcoded) | query DB จริงผ่าน `_list_portfolio_ids()` |
| `SYSTEM_PROMPT` | Base prompt เท่านั้น | + portfolio JSON guardrail + UC-2a/UC-2b persona separation |

**Migration pattern:**

| Notebook pattern | src/ replacement |
|---|---|
| `_get_secret(key)` Colab/dotenv hybrid | `os.getenv(key)` via `load_dotenv()` ใน `src/config.py` |
| `ls_client`, `tracer` (global) | `from src.config import ls_client, tracer` |
| `nest_asyncio.apply()` | ลบออก — ไม่ต้องการนอก notebook |
| `ngrok` tunnel | ลบออก — ใช้ reverse proxy / Railway จริง |
| `DB_PATH = "/content/..." if Colab` | `DB_PATH = os.getenv("DB_PATH", "portfolio.db")` |
| Cell G/H (FastAPI in Jupyter, threading+ngrok) | ลบทิ้ง — เขียนใหม่เป็น `src/api/routes.py` รันด้วย `uvicorn main:app` |
| SPY fetch ผ่าน `provider.get_history()` (tz-aware) | `yf.download()` ตรง (tz-naive) — match กับ portfolio data fetch |
| `MOCK_PORTFOLIOS` dict | SQLite query จริงผ่าน `_load_positions_async()` + `_list_portfolio_ids()` |

**กฎการรัน notebook:** รันจากบนลงล่างเสมอ ถ้า Cell 2 (LangSmith assert gate) ไม่ผ่านห้ามรันต่อ
หลัง refactor: Runtime → Restart → Run all ต้องผ่านครบก่อน push

**Notebook cell map** (สำหรับหา logic เดิมเวลาต้องเทียบกับ `src/` — API cells ถูกลบแล้ว):

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

⚠️ REMOVED: Cell G (FastAPI app) + Cell H (endpoint tests) — Bug 5 resolved by removal
```

---

## Routing {#routing}

### Case 5 — P/E + margin over-fetches get_stock_price (known limitation, accepted)

Query: "P/E กับ profit margin ของ AMD"
Expected: `{get_stock_financials}` | Actual: `{get_stock_financials, get_stock_price}` — 5/5 ครั้ง

Root cause: P/E ผูกกับราคาเชิงความหมาย — docstring-level negative routing งัดไม่ขึ้น
(LLM attends to `@tool` docstring เป็น structured schema field ไม่ใช่ human comment — ดู
`docs/POSTMORTEMS.md#docstring-routing`)

**Decision:** ไม่แก้ใน v1 — benign (price tool เร็ว/ถูกสุด) v2 ใช้ StateGraph conditional routing
แทน docstring engineering ต่อ เพราะนี่คือ model-level semantic association ที่ patch ที่ prompt
ไม่ได้แล้ว

Verified deterministic: 5x consistency check ใน `tests/test_routing_regression.py::test_case5_consistency`

### Emergent tool chaining — ปฏิเสธเป็น design pattern {#emergent-chaining}

พบว่า agent หา workaround เองได้ (เอา market value จาก `track_portfolio` มา proxy เป็น "amount"
ส่งเข้า `analyze_portfolio_risk` เพื่อเอา correlation matrix ที่ `track_portfolio` เดิมไม่มี)
deterministic ในทางปฏิบัติ (verify 3 รอบ ผลตรงกันหมด, Pearson 0.39, Rolling 0.46)

**Decision:** ไม่พึ่ง behavior นี้ต่อ เพราะ (1) MV-as-amount เป็น approximation ไม่ exact กับ
weight ที่ `track_portfolio` ใช้จริง และ (2) behavior ไม่ได้ specify ไว้ในระบบ — เสี่ยงเปลี่ยนถ้า
model version เปลี่ยนโดยไม่มีสัญญาณเตือน แก้โดย implement correlation ตรงใน `_track_portfolio_logic`
แทน (reuse `hist_returns` ที่มีอยู่แล้ว ไม่ fetch เพิ่ม)

---

## Deployment {#deployment} (open — ยังไม่ตัดสินใจสุดท้าย)

- ❌ PythonAnywhere — ตกไปแล้ว: internet access whitelist restrictions, ไม่รองรับ Docker
- 🔲 Current lean: presenting via repo + README + screen recording แทน live deployment
- 🔲 พิจารณา: Next.js portfolio website โชว์ FinAgent คู่กับ AstroSage (อีกโปรเจกต์แยก)

ยังไม่ final — ต้องตัดสินก่อนเขียน README ส่วน deployment

---

## QuantaAlpha integration (arXiv 2602.07085)

**Decision:** เอา concept ไม่ใช่ codebase

ที่เอา:
- Calmar Ratio → benchmark = 3.48 (ARR=27.75%, MDD=7.98%), rule of thumb > 1.0
- IC Score (Rank IC / Spearman) → signal quality, baseline IC=0.1501 (CSI300 weekly)
- IR = mean(IC)/std(IC) → consistency: > 0.5 usable, > 1.0 strong, > 2.0 exceptional
  (NVDA result: IC=-0.1065 (p=0.112) — magnitude strong แต่ contrarian, p ยังไม่ significant)

ที่ไม่เอา: Qlib pipeline, evolutionary loop, Next.js frontend, fixed factor weights
(pseudo-quant ถ้าไม่มี IC validation per-factor — ดู learning เรื่อง factor weights ใน POSTMORTEMS.md)

---

## v2 backlog {#v2-backlog}

เกิดจาก design discussion ตอนทำ Streamlit UI + Risk Contribution:

- **Correlation-based stress test** — รับ shock input เช่น `{"AMD": -0.20}` ใช้ correlation matrix
  + volatility ที่มีอยู่แล้วประมาณผลกระทบแบบ linear (ไม่ใช่ Monte Carlo) ต้องระบุขอบเขตความแม่นยำ
  ชัดในผลลัพธ์ (เป็น linear approximation จาก correlation ในอดีต ไม่ใช่การพยากรณ์ — correlation
  breakdown ตอนตลาดเครียดจริงเป็นความเสี่ยงที่ต้องบอกตรงๆ)
- **What-if เพิ่ม asset นอกพอร์ต** (เช่นเพิ่ม INTC เป็น hypothetical position) — ได้แค่
  correlation/diversification effect เชิงตัวเลข ไม่ใช่ causal impact จากข่าว
- **Conversation memory ข้ามturn** — ต้องเพิ่ม `ConversationTurn` table (`session_id`, `role`,
  `content`, `trace_id`) + pass message history เข้า `run_financial_agent()` แทนสร้าง
  `[HumanMessage(...)]` ใหม่ทุกครั้ง
- **StateGraph conditional routing** — แก้ case 5 อย่างถูกทาง (structural fix แทน docstring patch)
- **IC Score architecture debt** — ปัจจุบันเป็น standalone tool (`get_ic_score`, deprecated) ที่จริง
  ควร merge เข้า `get_hurst_exponent` output แบบที่ทำใน v1.5 แล้วบางส่วน — ตรวจสอบว่า merge ครบหรือยัง
- **Interpretation framework** (TODO เดิมจาก system prompt work — ยังไม่ implement) — agent ควร
  synthesize cross-signals ไม่ใช่แค่ list ตัวเลข:
  - Price position in 52W + Hurst regime → momentum context
  - Correlation matrix → true diversification ("AMD-NVDA corr=0.89 = semiconductor block ก้อนเดียว")
  - เมื่อ signals ขัดแย้งกัน ให้ระบุความขัดแย้งชัดๆ ไม่เลือกข้าง
  เพิ่มได้ใน system prompt แต่ต้อง regression test 11 cases ทุกครั้งหลังแก้ — และควรทำหลัง
  StateGraph routing เสร็จ เพราะ synthesis behavior จะขึ้นกับว่า tools ไหนถูกเรียกร่วมกันบ้าง

---

## v2 priority order {#v2-priority}

**Decision:** ทำตามลำดับ StateGraph routing → Conversation memory → deprecation fix →
Correlation-based stress test → What-if เพิ่ม asset

**เหตุผล:**
1. StateGraph routing กับ Conversation memory ทั้งคู่แก้ `src/agent/core.py` — รวบทำติดกันลด
   context switching และลด merge conflict กับตัวเอง ถ้าทำแยกกันจะต้อง refactor ไฟล์เดียวกันสองรอบ
2. `create_react_agent` deprecation fix (เดิมเป็น backlog แยก) ย้ายมาทำใน Phase 1 เพราะ "อยู่ตรงนั้น
   แล้ว" ไม่ต้องเปิด `core.py` ซ้ำอีกรอบเปล่าๆ
3. Feature ใหม่ (stress test, what-if) ทำหลัง agent infra นิ่งแล้ว เพราะเป็น tool ใหม่ที่ agent
   ต้อง route ถูก — ถ้า routing ยังไม่นิ่งจาก StateGraph work จะ debug ยากขึ้นเวลาเจอปัญหาใหม่
   (แยกไม่ออกว่าเป็น routing bug เก่าหรือ tool ใหม่ทำงานผิด)
4. What-if เพิ่ม asset ทำหลังสุดเพราะเล็กสุดและต่อยอดจาก `analyze_portfolio_risk` ที่มีอยู่แล้ว
   ไม่มีความเสี่ยง infra

**Conversation memory คือความเสี่ยง regression สูงสุดในกลุ่ม v2** — stateful behavior ทำให้
guardrail เดิม (stop-loss filter, persona separation, correlation misconception guard) ต้อง
retest ทั้งหมดใหม่ เพราะอาจ behave ต่างกันถ้า context มี conversation history (pattern เดียวกับ
drift ที่เจอตอน v1 guardrail saga — ดู `docs/POSTMORTEMS.md#guardrails`) — เพิ่ม multi-turn test
cases ใหม่ ไม่ใช่แค่รัน 11 cases เดิมซ้ำ (เดิมเป็น single-turn ทั้งหมด ไม่ cover stateful case)

---

## External review — scope validation {#external-review}

โปรเจกต์นี้ได้ external review 2 รอบจาก quant/product reviewer ภายนอก ก่อนเริ่ม v2 — เก็บสรุปไว้
เพราะมีคนอาจเสนอ feature เดิมซ้ำในอนาคต ให้ตรวจสอบ reasoning นี้ก่อน re-open

### รอบ 1 — ภาพรวม architecture + roadmap เสนอ

Reviewer ให้คะแนนสูง (product design 9/10, tool separation 8.5/10) และชม 2 จุดหลัก: การแยก
what-if vs tracked portfolio persona ชัดเจน และ Hurst+IC+IR ที่ลึกกว่า retail signal ทั่วไป

เสนอ roadmap 4 tier (A: robustness, B: predictive analytics [regime detection, factor exposure],
C: portfolio optimization [Efficient Frontier, Black-Litterman, HRP], D: explainability)

**Decision — รับบางส่วน ปฏิเสธส่วนใหญ่:**
- ✅ รับ: IC/IR ยังไม่มี train/validation split หรือ significance testing — เป็นช่องโหว่จริง
  log เป็น known limitation (ดู `docs/POSTMORTEMS.md` — ยังไม่ fix เพราะ scope v1 ปิดแล้ว)
- ✅ รับ: Query Planner / Execution Graph ที่เสนอ = StateGraph routing ที่มีอยู่ใน backlog แล้ว
  — confirm priority เดิม ไม่ใช่ feature ใหม่
- ❌ ปฏิเสธ Tier B/C/D ส่วนใหญ่ — ย้ายเข้า "Out of scope ถาวร" ใน `CLAUDE.md` เพราะขัดกับ
  positioning ตรงๆ ไม่ใช่แค่ effort สูง:
  - Factor exposure / regime detection (HMM) → pseudo-quant ถ้าไม่มี IC validation, หรือ
    interpretability cost > value
  - Portfolio optimization (Efficient Frontier, Black-Litterman, HRP) → คนละ layer จาก risk
    analysis ที่ทำอยู่ ข้ามไปทำจะกลายเป็น advisory engine ขัดกับ "not financial advice"
  - Bayesian expected return → การทำนาย return คือ advice โดยตรง เสี่ยงกว่า factor engine อีก
- 🔲 Tier A (data quality, cache, stale detection) → valid แต่ priority ต่ำกว่า v2 ที่เพิ่ม
  explainability/demo value โดยตรง เก็บเป็น v3 backlog

### รอบ 2 — reviewer ตอบกลับหลังเห็น decision รอบ 1

Reviewer เห็นด้วยกับการตัด factor/optimization ("รักษา identity" ไม่ใช่ "ตัด feature") และชม
guardrail philosophy (deterministic filter คุม stop-loss = "tool output → policy → language
layer" ใกล้ production มากกว่า prototype)

มี 2 จุดที่ challenge เพิ่ม:

**1. Planner priority ควรสูงกว่าที่คิด** — ไม่ใช่แค่ optimization แต่คือ **correctness** ถ้าระบบ
โตไป 10-15 tools semantic over-fetch/tool coupling จะโตแบบ nonlinear ไม่ใช่ linear cost

→ **Decision:** confirm ตรงกับแผนเดิมพอดี (StateGraph อยู่ Phase 1 อยู่แล้ว) ไม่ต้องเปลี่ยนอะไร
เป็นการ validate ไม่ใช่ scope ใหม่

**2. หลักการ "Explainable ≠ Simple"** — คำเตือนสำคัญที่สุดจากรอบนี้:

> "ระวังอย่าให้ 'Explainable' กลายเป็น 'Exclude anything difficult'; ใช้ explainability เป็น
> constraint ของ interface ไม่จำเป็นต้องเป็น constraint ของ analytics layer เสมอไป"

ตัวอย่างที่ให้: HMM อาจ explainable ได้ถ้า expose evidence-based output (realized vol ↑,
drawdown persistence ↑, Hurst ↓) แทนที่จะ expose latent state ตรงๆ — โมเดลซับซ้อนกับการอธิบาย
output ได้ เป็นคนละเรื่องกัน

→ **Decision:** ไม่เปลี่ยน scope ตอนนี้ (HMM ยัง out-of-scope เหมือนเดิม ด้วยเหตุผลอื่นร่วมด้วย —
scope/effort/interview timeline ไม่ใช่แค่ explainability) **แต่บันทึกหลักการนี้ไว้เป็น check
ก่อน reject feature ในอนาคต:** ก่อนบอกว่า "สิ่งนี้ไม่ explainable เลยไม่ทำ" ให้ถามก่อนว่า
สิ่งนี้ยากเพราะ *อธิบายไม่ได้จริงๆ* หรือยากเพราะ *ยังไม่ได้ลองหาวิธีอธิบาย*?
ใช้ตรวจสอบทุกครั้งที่จะ reject feature ด้วยเหตุผล "explainability" ล้วนๆ ไม่ใช่แค่ตอนนี้ครั้งเดียว