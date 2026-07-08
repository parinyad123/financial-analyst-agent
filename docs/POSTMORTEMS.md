# Postmortems & Key Learnings

รายละเอียดการ debug/iterate เต็มรูปแบบ — เปิดไฟล์นี้เมื่อกำลังจะแก้จุดที่เคย fail มาก่อน
อย่าทำ pattern เดิมซ้ำ

---

## Known bugs (fixed)

### ✅ Bug 1: BadRequestError — dict→JSON string
`analyze_portfolio_risk` takes `str` input, not `dict` — Groq schema validation rejects dict
input before reaching the function.
Fix: `@tool` wrapper ทำ `json.dumps(portfolio)` ถ้า input เป็น dict
```python
if not isinstance(portfolio, str): portfolio = json.dumps(portfolio)
```

### ✅ Bug 2: Calmar Ratio
Fix: `calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")` — merged เข้า Cell F

### ✅ Bug 3: Cell 5 smoke test
Fixed — marked DEPRECATED

### ✅ Bug 4: get_ic_score standalone
Fixed — IC merged เข้า `get_hurst_exponent`

### ✅ Bug 5: FastAPI port conflict — resolved by removal

Root cause (เดิม): Cell G รัน `app = FastAPI()` + uvicorn (threading + `nest_asyncio`) ซ้ำข้าม
cell execution → OSError 10048 (port 8000 in use)

**Resolution:** Cell G และ Cell H **ถูกลบออกจาก notebook แล้ว** ไม่ใช่แค่ deprioritized — root cause
คือข้อจำกัดของการรัน uvicorn ใน Jupyter cell ซึ่งไม่มีทางแก้ให้สะอาดในบริบทนั้น API ถูกเขียนใหม่ตรง
ใน `src/api/` แทน รันผ่าน `uvicorn main:app` ตรงจาก terminal — ไม่มี kernel state ค้าง

Notebook ตอนนี้มีหน้าที่เป็น algorithm/tool development environment เท่านั้น ไม่ใช่ full-stack demo

### ✅ Bug 6: SPY tz mismatch → Alpha/Beta = N/A เงียบ {#bug-6-spy-tz}

Root cause: `yf.download()` คืน tz-naive index, `yf.Ticker().history()` (ผ่าน DataProvider) คืน
tz-aware — `DatetimeIndex.intersection()` บน mixed-tz ได้ empty set โดยไม่ error

Fix: strip tz ก่อน intersection ใน `src/tools/portfolio_risk.py`
Verified: `Alpha (ann, vs SPY): +26.76% | Beta: 2.251` (ไม่ใช่ N/A)

Fixed ใน `src/` เท่านั้น — ยังไม่แก้ใน notebook (ตามนโยบาย notebook = historical reference)

### ✅ Bug 7: UI misattribution — "พอร์ตของคุณเคยขาดทุน" ใน what-if form

ดู `#persona-separation` ด้านล่าง

---

## System prompt fix — UC-2a/UC-2b persona separation {#persona-separation} (✅ done)

**ปัญหาที่พบ:** `analyze_portfolio_risk` (what-if, ไม่มี cost basis) พูดว่า "พอร์ตของคุณเคยขาดทุน"
— misattribution เพราะ tool ไม่รู้ว่า user ถือจริงเมื่อไหร่/ราคาเท่าไหร่ และ agent เคยคำนวณ
self-generated stress test (เช่น "ตลาดตก 10% × Beta 2.48 = พอร์ตตก 25%") ซึ่งเป็น false-precision
— ตัวเลขไม่มาจาก tool โดยตรง

**Fix:** เพิ่ม 2 กฎใหม่ใน `src/agent/prompts.py` (เขียนเป็นภาษาอังกฤษ แทรกแบบ mixed-language กับ
ของเดิม — negative instruction ทำงานแม่นกว่าในอังกฤษสำหรับ gpt-oss-120b โดยไม่ต้องแปลทั้ง prompt
และไม่เสี่ยง regression จาก language shift):

1. UC-2a ห้ามพูด "ของคุณ"/"your portfolio lost" — ใช้ "พอร์ตสมมตินี้"/"if held during the period" เสมอ
2. UC-2a ห้ามคำนวณ what-if scenario เอง (Beta × shock %) — รายงาน Beta/Correlation ตรงจาก tool
   แล้วอธิบายเชิงคุณภาพเท่านั้น
3. UC-2b ยังใช้ "ของคุณ"/"you are up/down" ได้ปกติ — มี real cost basis รองรับ

**Verified:** ทดสอบ query เดิมที่เจอ bug ผ่านครบ 3 จุด (ไม่มี "ของคุณ", ไม่มี self-generated stress
number, ยังอธิบาย metrics ครบ) — รัน routing regression ซ้ำยืนยันไม่กระทบ (10/11 passed เหมือนเดิม)

---

## Risk Contribution Analysis {#risk-contribution} (✅ Complete — tool logic + guardrail)

**Why:** User ต้องการรู้ว่า "หุ้นตัวไหนในพอร์ตเป็นตัวเพิ่มความเสี่ยงมากสุด" — per-ticker volatility
เดี่ยวๆ ไม่บอก contribution ต่อ portfolio-level risk จริง (ไม่ได้คำนึง correlation/weight ร่วมกัน)

**Formula** (มี mathematical backing ชัดเจน — ไม่ใช่ pseudo-quant แบบ fixed factor weight):
```
marginal_contribution_to_risk[i] = weight[i] × cov(asset_i, portfolio) / portfolio_variance
```
ผลรวมของทุก ticker = 100% ของ portfolio variance — เป็น decomposition ที่ exact ไม่ใช่ approximation

**Verified results:**

| Tool | Test case | Weight | Risk Contribution | Sanity check |
|---|---|---|---|---|
| `analyze_portfolio_risk` | NVDA 5000 + AMD 3000 | NVDA 62.5% / AMD 37.5% | NVDA 46.1% / AMD 53.9% (AMD vol สูงกว่า) | sum = 100.0000% |
| `track_portfolio` | streamlit-test-001 (NVDA+TSLA) | NVDA 50.7% / TSLA 49.3% | NVDA 43.4% / TSLA 56.6% (TSLA vol สูงกว่า) | sum = 100.0000% |
| `track_portfolio` | dead ticker (NVDA + ZZZFAKE999) | NVDA 100% (excluded) | "100.0% (single asset)" | graceful, warning แสดงครบ |

**Wording scope แยกตาม 2 tools** (ต้องคุม wording ตาม persona separation):

| Tool | คำที่ใช้ได้ | คำที่ห้ามใช้ |
|---|---|---|
| `analyze_portfolio_risk` (UC-2a) | "AMD มี risk contribution สูงสุด — ถ้าต้องการลดความเสี่ยงรวม การลดสัดส่วนนี้คือทางหนึ่ง" | "ควรขาย AMD" (advice ตรง + พอร์ตยังไม่มีจริง) |
| `track_portfolio` (UC-2b) | "AMD ขาดทุน -12% และเป็นตัว contribute ความผันผวนสูงสุด — อาจพิจารณาทบทวนสัดส่วนนี้" | "ควรขาย AMD ทันที" (advice ตรงเกินไปแม้มี cost basis จริง) |

**Verified (query: "หุ้นตัวไหนเสี่ยงสุดในพอร์ตนี้ ควรทำยังไงดี"):** AMD 79.4% risk contribution
รายงานถูกพร้อมคำอธิบาย ไม่มี "ควรขาย AMD"/"ต้องลด AMD" ตรง — ใช้ "อาจพิจารณา", "เป็นทางหนึ่ง" แทน

---

## Guardrails {#guardrails} — stop-loss drift saga

**ทำไมเรื่องนี้สำคัญ:** นี่คือเคสที่พิสูจน์ว่า **prompt-only guardrail มี diminishing returns**
สำหรับ open-ended avoidance task — ก่อนจะลองแก้ guardrail ด้วย prompt อย่างเดียวอีก ให้อ่าน saga นี้ก่อน

### Round 1a — direct stop-loss/hedging action (initial fix)

พบ agent แนะนำ "ตั้ง stop-loss"/"ใช้ hedging" หลุดออกมา — ละเมิด rule เดิมตั้งแต่ v1
("Do not give specific price targets, entry points, or stop-loss levels")
แก้โดยเสริม negative example ชัดเจนเข้า rule เดิม: "suggesting the *action* of stop-loss/hedging
is itself actionable advice, even without a price"
Verified ผ่านตอนนั้นด้วย query เดี่ยวที่ถามตรงๆ เรื่อง stop-loss

### Round 1b — track_portfolio + compound question (prompt-only ไม่พอ)

ทดสอบ `track_portfolio` ด้วย compound question ("AMD เสี่ยงสุดไหม **และ** ควรตั้ง stop-loss ไหม")
เจอ drift กลับมา — guardrail เดิม cover แค่ direct command แต่ไม่ cover:
- **soft endorsement** ("stop-loss สามารถช่วยจำกัดการขาดทุนได้")
- **indirect framing** ("คุณอาจพิจารณาว่า stop-loss สอดคล้องกับ risk tolerance")

ยืนยันด้วย isolate test 3 รอบ — ไม่ deterministic (2/3 fail ด้วย pattern ต่างกัน) ไม่ใช่ fluke

### Round 1c — NEVER + MUST + exact format (ยังไม่พอ)

เปลี่ยน "do NOT say" → "NEVER", เพิ่ม exact failure phrase เป็น negative example, บังคับ explicit
refusal ด้วย `MUST` + fixed format string — รัน 3 รอบซ้ำ ปิดช่องเดิมได้ (soft endorsement หายไป)
**แต่เปิดช่องใหม่ทันที**: model ใช้ **"listing-as-example"** (เอา stop-loss ปนใน list ร่วมกับ
rebalancing — "การตัดสินใจว่าจะใช้เครื่องมือใด เช่น stop-loss, hedge") implicit ว่าเป็น valid
option โดยไม่ต้อง endorse ตรง — 3 รอบนี้แต่ละรอบ fail คนละ pattern (R1 paraphrase ไม่ตรง exact
format, R2 silent omission, R3 borderline wording)

### Decision point — เปลี่ยนเครื่องมือ ไม่ patch ต่อ

prompt iteration ปิดช่องหนึ่งแล้วเปิดช่องใหม่ทุกรอบ (3 รอบ, 3 pattern ต่างกัน) — ต่างจาก correlation
misconception (#correlation-misconception ด้านล่าง) ที่เป็น fixed factual error และ converge ได้
ใน 2 รอบ เพราะมีคำตอบ "ถูก" ทางเดียว ส่วน "ห้ามพูดถึง stop-loss ในทางบวก" เป็น **open-ended
avoidance task** ที่มีวิธี "บวก" ได้นับไม่ถ้วน — สัญญาณ diminishing returns ของ prompt-only
approach ชัดเจน

**Decision: defense-in-depth** — prompt guardrail (ลดโอกาส) + deterministic post-processing
filter (รับประกัน)

### Round 1d — deterministic post-processing filter (final fix)

เพิ่ม `_filter_stoploss()` ใน `src/agent/core.py` — scan response แยก paragraph, ลบ paragraph
ที่มี stop-loss/hedging keyword ออก, แทนด้วย fixed refusal statement, log ทุกครั้งที่ trigger
(`WARNING: [stoploss-filter] prompt guardrail leaked`) ทำงานหลัง `run_financial_agent()` ได้
response กลับมา ก่อน return — ไม่พึ่งความสม่ำเสมอของ LLM อีกต่อไป

**Root cause ที่ซ่อนอยู่ใน filter รอบแรก (สำคัญ — Unicode gotcha):** deploy filter รอบแรกแล้ว
**ไม่ trigger เลยทั้ง 3 รอบ** ทั้งที่ response มีคำว่า "stop-loss" อยู่จริง — เพราะ model output ใช้
**`U+2011` (NON-BREAKING HYPHEN)** ไม่ใช่ ASCII hyphen (`U+002D`) ทำให้ string match ไม่เจอ
verify ด้วย `repr()` ทีละ codepoint ก่อนเชื่อว่า logic ผิด แก้โดยเพิ่ม Unicode hyphen normalization
(`_normalize_hyphens()`) ก่อน keyword check

> **บทเรียน: string-matching guardrail ต้อง normalize Unicode variant เสมอ** (hyphen variants,
> full-width characters, zero-width characters) ไม่ใช่แค่ ASCII keyword list

**Verified (final, 3 layers ทำงานร่วมกัน):** 3/3 รอบผ่านครบ 3 จุด (ไม่มี endorsement, exact
refusal statement ปรากฏ, risk analysis AMD 94.5% ยังครบ) filter trigger ทุกรอบ (prompt guardrail
ยังหลุดอยู่ — filter เป็น safety net ที่จำเป็นจริง ไม่ใช่ redundant) regression: 10/11 passed
(`test_case5_consistency` fail ด้วย Groq 429 rate limit — infrastructure issue ไม่ใช่ regression)

### Post-StateGraph-refactor retest (v2 Phase 1 verification, session 3)

Agent node ถูกเขียนใหม่ทั้งก้อน (planner→agent→tools) — ต้อง retest guardrail เดิมทั้งหมดเพราะ
`_filter_stoploss()` ทำงานหลัง `run_financial_agent()` ได้ response กลับมา (ไม่ได้อยู่ใน graph
node ใดๆ) แต่ response ตอนนี้มาจาก code path ใหม่ทั้งหมด:

- **`test_case5_consistency` 5/5 รอบ:** `get_stock_financials` ทุกรอบ, `get_stock_price` ไม่หลุด
  เลยสักรอบ — v2 fix ยัง deterministic ตามที่ verify ไว้ session 2
- **Adversarial compound question ×3** ("พอร์ต streamlit-test-001: TSLA เสี่ยงสุดไหม และควรตั้ง
  stop-loss ไหม" — query เดียวกับที่เผย case 12 ด้วย): ไม่มี endorsement ทุกรูปแบบ (direct/soft/
  listing-as-example) ทั้ง 3 รอบ, refusal statement ปรากฏครบ, **`_filter_stoploss` trigger ทุก
  รอบ** (prompt guardrail ยังหลุดเหมือนเดิม — filter ยังจำเป็นจริง ไม่ใช่ redundant กับ StateGraph
  ใหม่) — ผล routing แต่ละรอบไม่เหมือนกัน (ดู `#case-12`) แต่**guardrail เอาอยู่ไม่ว่า routing จะ
  ไปทางไหน** เพราะ filter ทำงานที่ response level ไม่ผูกกับ tool ที่ถูกเรียก
- **Persona separation (UC-2a, `{"NVDA":5000,"AMD":3000}` + "พอร์ตนี้เสี่ยงแค่ไหน") ×1 formal
  confirmation:** ไม่มี "ของคุณ", ไม่มี self-generated stress number, metrics ครบ — ยืนยันตาม
  ที่ user ทดสอบเองผ่านมาก่อนแล้ว หมายเหตุเสริม: รันซ้ำอีกครั้ง (ไม่ใช่ formal round) model
  พูดถึง stop-loss เองแบบไม่ถูกถาม (spontaneous, ไม่ใช่ deterministic) — `_filter_stoploss`
  จับได้ถูกต้องเหมือนกรณีถูกถามตรง ยืนยันว่า filter ทำงานที่ **response content** ไม่สนว่า
  model พูดเรื่องนี้มาจากไหน — เป็นหลักฐานเพิ่มว่า defense-in-depth design (deterministic filter
  ไม่ผูกกับ trigger condition ใดๆ) ทำงานตามที่ตั้งใจ

**สรุป:** guardrail 3 ชั้น (prompt + deterministic filter + persona separation) ยังทำงานถูกต้อง
ทั้งหมดหลัง agent core ถูกเขียนใหม่ทั้งก้อน — ไม่มี regression จาก StateGraph refactor

---

## Case 12 — planner misroutes "พอร์ต {id} + risk phrasing" {#case-12}

**พบยังไง:** ไม่ได้มาจาก 13-case regression suite (มันไม่มี case แบบนี้เลยตอนนั้น) แต่พบตอน
manual exploratory test ของ adversarial guardrail (query จริง: "พอร์ต streamlit-test-001: TSLA
เสี่ยงสุดไหม และควรตั้ง stop-loss ไหม") — เป็น query ที่ compound ทั้ง **id พอร์ตจริง** +
**risk/stop-loss phrasing** พร้อมกัน ซึ่งเป็น combination ที่ 11-case suite เดิมไม่มี (case ที่มี
portfolio_id ก็ไม่มี risk/stop-loss phrasing ร่วม, case ที่มี risk phrasing ก็เป็น what-if ล้วน
ไม่มี id จริง) — co-trigger boundary นี้จึงไม่เคยถูก cover

**อาการ:** รัน query เดิมซ้ำ 3 รอบ (ก่อน fix) ได้ plan ไม่ตรงกันเลยสักรอบ:
- รอบ 1: planner เลือก `analyze_portfolio_risk` (ผิด — TSLA ถูกตีความเป็น what-if 100% ไม่ใช่
  ตำแหน่งจริงใน `streamlit-test-001`)
- รอบ 2: planner เลือก `track_portfolio` (ถูก โดยบังเอิญ)
- รอบ 3: planner เลือก `analyze_portfolio_risk` อีก **และ synthesize ไม่ได้ด้วย** (ไม่มี sibling
  arg ให้ยืม portfolio JSON) → ไม่เรียก tool อะไรเลย ตอบขอข้อมูลพอร์ตใหม่จาก user ทั้งที่มี
  portfolio จริงอยู่แล้ว

Nondeterministic 3 รูปแบบใน 3 รอบ — ไม่ใช่ fluke, เป็น genuine boundary gap: risk/stop-loss
wording คือสัญญาณที่ planner "เห็น" ชัดกว่า id string เสมอ

**Root cause:** เหมือน case 5 เป๊ะ — เป็น semantic association ระดับ model (risk phrasing ↔
what-if tool) ไม่ใช่ปัญหาที่ prompt wording แก้ได้ยั่งยืน (ดู `#docstring-routing`) ต่างจาก case 5
ตรงที่ signal ที่ถูกต้อง (portfolio_id) **ตรวจสอบได้แบบ deterministic** อยู่แล้ว (เทียบกับ DB) —
ไม่ต้องพึ่ง LLM ตัดสินเลยสำหรับ case นี้โดยเฉพาะ

**Fix:** `_plan_override()` (`src/agent/core.py`) — word-boundary regex match query กับ
`_list_portfolio_ids()` (DB) **ก่อน**เรียก LLM planner เสมอ เจอ id จริง → force
`plan = ["track_portfolio"]` ทันที ข้าม LLM planner ทั้งหมดสำหรับ query นั้น (deterministic,
ไม่มี LLM call เพิ่ม, เร็วกว่าเดิมด้วยเพราะ skip planner call) รายละเอียด design: ดู
`docs/DECISIONS.md#case-12`, `docs/ARCHITECTURE.md#agent-framework`

**Verified after fix:** query เดิม probe เดี่ยว 1 รอบ + ใน 13-case regression suite (case 12) —
`track_portfolio` เท่านั้นทุกรอบ ไม่มี variance เหลือ เพิ่ม counter-case (case 13: what-if query
ที่ไม่มี id จริง "พอร์ตแบบนี้เสี่ยงไหม NVDA 5000 AMD 3000") ยืนยันว่า override ไม่ over-trigger
กับ query ที่ไม่มี id จริง — 13/13 passed

> **บทเรียน: co-trigger boundary ที่ regression suite ไม่ cover ต้องหาโดย manual adversarial
> testing ไม่ใช่แค่รัน suite เดิมซ้ำ** — suite ที่ผ่าน 100% ไม่ได้แปลว่าไม่มี gap เหลือ ถ้า suite
> ไม่เคย exercise combination ของ signal สองอย่างพร้อมกัน (ที่นี่คือ id + risk phrasing)

---

## Correlation/diversification statistical misstatement {#correlation-misconception}

พบ 2 statistical misconception แยกจาก Risk Contribution โดยตรง แต่เจอตอน verify คำตอบ portfolio
risk เดียวกัน:

1. Agent อธิบาย correlation 0.5 เป็น "เคลื่อนที่เหมือนกันประมาณครึ่งหนึ่งของเวลา/ของความเคลื่อนไหว
   ทั้งหมด" — ผิดทางสถิติ correlation วัด linear relationship strength ไม่ใช่ proportion/frequency
   ของการเคลื่อนไหว
2. Agent ใช้คำว่า "Diversification = 0" — false-precision เพราะ tool ไม่มี diversification metric
   เลย (มีแค่ correlation matrix + risk contribution)

**Fix iteration:** guardrail รอบแรกห้ามแค่ "time-based" claim — model หลบคำว่า "เวลา" ไปใช้
"สัดส่วนของความเคลื่อนไหว" แทน (misconception เดิม คำพูดใหม่) ต้องเขียน guardrail ใหม่แบบ
**principle-based** (อธิบายว่าทำไม correlation ไม่ decompose เป็น proportion ได้ ไม่ใช่ list คำต้อง
ห้ามทีละแบบ)

**Verified ผ่านหลังแก้:** agent ใช้ "moderate positive relationship", "บ้างแต่ไม่เต็มที่" ล้วน
ไม่มี proportion/frequency claim ในรูปแบบใดเลย ยังอธิบาย mechanism ถูกได้ (เช่น "ถ้า correlation
สูงขึ้น variance จะเพิ่มขึ้น" — เชิงทิศทาง ไม่ใช่ตัวเลขทำนาย) แสดงว่า principle-based guardrail
ไม่ทำให้คำตอบจนเนื้อหา แค่กรองส่วนที่ผิดจริง

> **บทเรียน: principle-based guardrails outperform keyword blocklists** สำหรับ LLM compliance
> task ที่มีคำตอบถูกทางเดียว (factual error) — converge เร็ว (1-2 รอบ) ต่างจาก open-ended
> avoidance task (เช่น stop-loss) ที่ต้องใช้ deterministic filter เสริม

**Regression note:** หลังเพิ่ม guardrail รอบสอง เจอ 9/11 ชั่วคราว (case 10 — multi-tool query ขาด
`search_market_news`) — isolate รัน case 10 เดี่ยว 3 รอบ ได้ tools ครบทุกรอบ ไม่มี ticker แปลกปลอม
("NVNV" ที่เคยเห็นตอน fail) ยืนยันเป็น LLM nondeterministic fluke ไม่ใช่ regression จาก guardrail
— final regression: 10/11 passed

---

## Streamlit bugs {#streamlit-bugs}

พบตอน browser testing จริง (ไม่ใช่แค่ยิง endpoint ตรง):

1. `_split_response` เลือก disclaimer สั้นแทน summary จริง — แก้ filter จาก `startswith("#")` เป็น
   `is_bare_header` + raise threshold เป็น 60 chars
2. `ticker_in_query` dead condition (`False or False → None`) — ลบทิ้ง ส่งtickerตรง + ชี้แจงว่า
   ticker field มีผลแค่ LangSmith metadata ไม่ใช่ agent context

**Important UX clarification:** Ticker field ใน Tab "ถามทั่วไป" มีผลแค่กับ quick-question buttons
(insert ticker เข้า pre-filled query) — **ไม่ได้ถูกส่งเข้า agent context** สำหรับ free-text queries
เพราะ `ticker` ใน API request ไปอยู่แค่ LangSmith metadata/tags (`src/api/routes.py`) ไม่ผ่านเข้า
`run_financial_agent(query, ...)` เลย — ถูกต้องตาม design ไม่ใช่ bug แต่ทำให้สับสนได้ถ้าไม่อ่าน label

---

## Key learnings & principles (ทั้งหมด)

- **LangSmith tracing:** `lru_cache` บน `get_env_var` → ใช้ explicit binding เสมอ — ครอบคลุมแค่
  credential (`client=ls_client`) ไม่ใช่ tracing on/off switch — `LANGCHAIN_TRACING_V2` ยังต้อง
  อยู่ใน env (ผ่าน `setdefault`) เพราะ LangGraph internal เช็คจาก env เท่านั้น
- **`@tool` / `@traceable` ห้ามซ้อน** บนฟังก์ชันเดียวกัน — แยก outer/inner
- **`traceable` naming:** verb + noun (เช่น `"fetch_stock_price"`), ไม่ใช่ชื่อ function
- **Tool hallucination:** tool หายออกจาก `tools = [...]` → agent fabricate metrics convincingly
  — LangSmith trace ช่วย debug
- **Agent routing {#docstring-routing}:** LLMs attend to `@tool` docstrings as structured schema
  fields (ไม่ใช่ human-readable comment) — `@tool` decorator แปลง docstring เป็น `description`
  field ใน JSON tool spec ที่ส่งผ่าน `tools=[...]` อธิบายว่าทำไม negative routing guard ใน
  docstring ทำงานได้บางกรณี และทำไม case 5 ยังคงอยู่ (model-level semantic association ที่
  docstring engineering แก้ไม่ได้ — ต้อง structural fix)
- **Tool input format:** JSON string input over dict + normalize guard ใน `@tool` wrapper
- **Database design:** เก็บเฉพาะ source-of-truth (ticker, shares, avg_cost) — derive ทุกอย่าง
  on-the-fly
- **Error messages:** แยก "invalid ticker" vs "transient API failure" ไม่ให้ agent เข้าใจผิด
- **asyncio Windows:** `try/except` กับ `loop.run_until_complete()` — ไม่ใช้ `asyncio.run()`
- **Jupyter + uvicorn ไม่เข้ากันสำหรับ production demo:** threading + nest_asyncio ซ้อน event loop
  ทำให้ port conflict แก้ไม่จบ — บทเรียนคือไม่ต้องพยายาม "fix" pattern นี้ใน notebook อีก ให้
  ออกจาก notebook ไปเขียน script ตรงๆ ดีกว่า
- **Rolling Hurst > single Hurst:** single point บอกไม่ได้ว่า regime กำลัง shift — time-series
  ของ H มีประโยชน์กว่า
- **IR สำคัญกว่า IC เดียว:** IC snapshot อาจ noise — IR วัด consistency ข้ามเวลา
- **Factor weights ต้องมาจากข้อมูล:** hardcode weights = pseudo-quant — ต้อง learn
  (Lasso/ElasticNet) หรือ validate IC per-factor ก่อน
- **SPY tz mismatch ทำให้ Alpha/Beta = N/A เงียบ:** ดู Bug 6 ด้านบน — ถ้า data fetch มาจาก 2 source
  ต่างกัน ต้องเช็ค tz-awareness ให้ตรงกันก่อน join เสมอ
- **Thai text "?" ใน log ≠ input เพี้ยน เสมอไป:** ทดสอบด้วย `repr(query)` ก่อนสรุปว่า input ผิด —
  กรณีนี้ปัญหาคือ (1) bash curl บน Windows ส่ง encoding ผิดตั้งแต่ shell (ไม่ใช่ FastAPI bug) และ
  (2) Windows stdout default เป็น cp1252 พิมพ์ Thai unicode ไม่ได้ — แก้ด้วย
  `sys.stdout.reconfigure(encoding="utf-8")` ใน `main.py` เท่านั้น — ทดสอบจริงผ่าน Swagger UI
  (`/docs`) เสมอ เพราะส่ง UTF-8 ถูกต้องอัตโนมัติ
- **Git Bash curl บน Windows ไม่เหมาะทดสอบ Thai API:** ส่ง Thai chars เป็น `????` ตั้งแต่ shell
  ก่อนถึง server เสมอ ทำให้ routing test ดู "ผิด" ทั้งที่ server ทำงานถูก (เคยทำให้เข้าใจผิดว่า
  news routing regress) — ทดสอบ Thai query ผ่าน **Swagger UI** หรือ **PowerShell
  `Invoke-RestMethod`** หรือ **Python `urllib`/`requests`** เท่านั้น ไม่ใช้ bash curl บน Windows
- **Notebook ≠ source of truth หลัง production migration:** bugfix ที่ทำใน `src/` ไม่ sync กลับ
  notebook โดยอัตโนมัติ — ก่อนทดลอง feature ใหม่ ต้อง verify ว่าจะทดลองที่ไหน (`src/` ตรงผ่าน
  standalone script ไม่ใช่ notebook) ไม่งั้นจะเจอ bug ที่ "หายไปแล้ว" กลับมาอีก
- **English negative instruction แม่นกว่าไทยสำหรับ gpt-oss-120b:** ไม่ต้องแปล system prompt
  ทั้งหมด — แทรกกฎ critical (persona separation, ห้าม self-generate ตัวเลข) เป็นอังกฤษแบบ
  mixed-language กับของเดิมที่เป็นไทยได้
- **Risk contribution ต้อง verify ด้วย sanity check sum=100%:** มี mathematical backing ชัดเจน
  ต่างจาก fixed factor weight ที่เป็น pseudo-quant — implementation ผิดเล็กน้อย (เช่น ใช้
  correlation แทน covariance) จะทำให้ sum ≠ 100% ตรวจจับได้ง่ายถ้า print ออกมาดูทุกครั้ง
- **Regression fail ไม่เท่ากับ regression จริงเสมอ:** หลังแก้ prompt เจอ routing fail 9/11
  (จากปกติ 10/11) — ก่อนสรุปว่าเป็น regression ให้ isolate test case ที่ fail แล้วรันซ้ำ 3 รอบเดี่ยวๆ
  ถ้าผ่านครบทุกรอบ = LLM nondeterministic fluke ไม่ใช่ผลจาก prompt change เช็ค error/ticker
  แปลกปลอมที่หลุดมาด้วยเป็น signal เสริม
- **Negative instruction ต้องมี negative example ไม่ใช่แค่กฎเปล่า:** "ห้ามให้คำแนะนำ" อาจไม่
  ครอบคลุมพอ — agent อาจตีความว่า "แนะนำให้ตั้ง stop-loss" ไม่ใช่คำแนะนำเพราะไม่ได้ระบุราคา
  ต้องเสริมตัวอย่างชัดว่า "การแนะนำ action (เช่น ตั้ง stop-loss) คือ advice แม้ไม่มีตัวเลข"
- **Prompt-only guardrail มี diminishing returns สำหรับ open-ended avoidance task:** ดู
  stop-loss saga ด้านบน — สัญญาณว่าต้อง**เปลี่ยนเครื่องมือ** ไม่ใช่ patch ต่อ คือเมื่อแต่ละรอบ
  patch ปิดช่องเดิมแต่เปิดช่องใหม่
- **String-matching guardrail ต้อง normalize Unicode variants เสมอ:** ดู `U+2011` bug ด้านบน
  — verify ด้วย `repr()` ทีละ codepoint ก่อนเชื่อว่า logic ผิดหรือ filter ไม่ trigger
- **Emergent tool chaining ดูดี แต่ไม่ควรพึ่งเป็น design:** ดู `docs/DECISIONS.md#emergent-chaining`
- **pytest จาก root ต้อง set `PYTHONPATH=.` ก่อนรัน** — ไม่งั้น `ModuleNotFoundError: No module
  named 'src'` เพราะไม่มี `conftest.py` หรือ `pythonpath` config ใน `pyproject.toml`
- **Keyword blocklist เปราะกว่า principle-based guardrail:** ดู correlation misconception ด้านบน
  — verify ด้วยการอ่าน semantic ของคำตอบทั้งประโยค ไม่ใช่แค่ grep หาคำที่เคย fail