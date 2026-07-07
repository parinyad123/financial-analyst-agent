# v2 — StateGraph routing (scratch)

> Scratch doc ระหว่างทำ Phase 1 #1 (StateGraph routing). ลบ/distill กลับเป็น 2-3 บรรทัด
> ใน CHANGELOG.md + ARCHITECTURE.md เมื่อจบงาน. **ห้ามให้ WIP note ไหลเข้า CLAUDE.md.**

## เป้าหมาย

แก้ **case 5** — query "P/E กับ profit margin ของ AMD" over-fetches `get_stock_price`
(ควรได้ `{get_stock_financials}` ล้วน). Root cause: P/E ผูกกับราคาเชิงความหมายระดับ model
— docstring/prompt-level negative routing งัดไม่ขึ้น (`docs/DECISIONS.md#routing`,
`docs/POSTMORTEMS.md#docstring-routing`). ต้อง **structural fix** ไม่ใช่ patch prompt ต่อ.

พร้อมกัน: แก้ `create_react_agent` deprecation (LangGraph V1.0 → V2.0 removal;
ยืนยัน warning: `LangGraphDeprecatedSinceV10: create_react_agent has been moved to
langchain.agents`). การเลิกใช้ `create_react_agent` = warning หายเอง.

## Constraint (จาก prompt งานนี้)

- ห้ามแก้ `SYSTEM_PROMPT` wording (`src/agent/prompts.py`) — แตะไม่ได้เลย
- ห้ามแตะ tool logic ใน `src/tools/`
- `_filter_stoploss()` ต้องอยู่ครบ + ยัง trigger ได้ (มี test ยืนยัน)
- ไม่เพิ่ม feature อื่น (stress test / memory = งานถัดไป แยก session)

## Interface ที่ห้ามพัง (contract กับ caller เดิม)

`build_agent()` และ `run_financial_agent()` ถูกเรียกจาก:
- `tests/test_routing_regression.py` → `build_agent()` แล้ว `.stream(inputs, config, stream_mode="values")`
  และ scan `event["messages"]` หา `AIMessage` ที่มี `.tool_calls` (นับ tool ที่ถูกเรียก)
- `src/api/routes.py` → `run_financial_agent(query, tickers, analysis_type)` → dict มี `run_id`

ดังนั้น compiled graph **ต้อง**:
1. รองรับ `.stream(inputs={"messages":[HumanMessage]}, config, stream_mode="values")`
2. emit `AIMessage` ที่มี `tool_calls` ลง `state["messages"]` (reducer `add_messages`)
   เพื่อให้ test นับ tool ได้เหมือนเดิม
3. last message content = final answer (ให้ `run_financial_agent` ดึงไปผ่าน `_filter_stoploss`)

**Gotcha:** planner ใช้ `.with_structured_output()` ซึ่งภายในเป็น tool-calling —
ห้ามให้ AIMessage ของ planner หลุดลง `state["messages"]` ไม่งั้น test จะนับ tool ปลอม.
→ planner เก็บผลใน state field แยก (`plan`) ไม่ append raw message.

## Design — planner → agent → tools loop

เปลี่ยนจาก **semantic routing** (ReAct ตัดสินใจ tool เอง แบบ greedy) เป็น
**structured planning** (แยก classification step ออกมาก่อน execution):

```
START → planner → agent ──(tool_calls?)──► tools ──► agent → ... → END
                    └────────(no calls)──────────────────► END
```

State:
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]        # planned tool names (subset ของ 6 ตัว)
```

- **planner node**: LLM call (temp=0, structured output → `RoutingPlan.tools`).
  รับ query ล่าสุด + PLANNER_PROMPT (prompt ใหม่ แยกจาก SYSTEM_PROMPT — *ไม่ถือว่าแก้
  SYSTEM_PROMPT*). intersect ผลกับชื่อ tool จริง. เขียนแค่ `{"plan": [...]}` ไม่แตะ messages.
- **agent node** (final design): bind **ALL** tools → invoke → บน **first turn**
  reconcile tool_calls ให้เท่ากับ plan:
  - **upper bound** — drop tool_calls ที่อยู่นอก plan (แก้ over-fetch: case 5 pull price)
  - **lower bound** — planned tool ที่ model ข้าม → **synthesize tool_call เอง**
    (deterministic, ไม่มี LLM call เพิ่ม) โดยยืม args: ticker tool ยืม ticker จาก sibling
    call / regex จาก query; `search_market_news` ใช้ query = user query; portfolio tool
    ยืม portfolio arg. แก้ under-call: case 10 drop `search_market_news`.
  - rebuild เป็น `AIMessage` ใหม่ (ไม่ mutate) เพื่อล้าง raw off-plan call ใน
    `additional_kwargs` ไม่ให้ re-serialize กลับ Groq → 400 รอบสอง.
  synthesis turn (ถัดไป) pass through ไม่แตะ. → routing = plan **deterministic**.
- **tools node**: `ToolNode(ALL_tools)` — execute แค่ tool_calls ที่ผ่าน reconcile.

### ทำไม planner แก้ case 5 ได้ทั้งที่ ReAct แก้ไม่ได้

ReAct ไม่เคยถูกถาม "tool ไหน" ตรงๆ — เรียก tool greedy ระหว่าง generation, associate
P/E→price. Planner เป็น **dedicated classification task** + prompt เขียน boundary ชัด
("P/E/valuation/margin = fundamentals → get_stock_financials ONLY; price เฉพาะถามราคาตรงๆ").
plan → filter output → routing = plan.

### 🔬 Journey — approach ที่ลองแล้วพัง (สำคัญ อย่าทำซ้ำ)

ลำดับที่ลอง (แต่ละอันแก้ปัญหาก่อนหน้า):

1. **bind เฉพาะ planned tool** → `test_routing_regression` 11/11 แต่
   `test_case5_consistency` (5 รอบ) พัง `groq.BadRequestError 400: attempted to call
   tool 'get_stock_price' which was not in request.tools`. **gpt-oss-120b ยังพยายามเรียก
   price สำหรับ P/E จริงๆ** (semantic pull ของ case 5 เป็นของจริงระดับ model) และ **Groq
   validate tool call ฝั่ง server → 400 hard error** แทนเมินเฉย → binding subset เปราะ.
   → แก้: **bind ครบ + filter output** (model ไม่เคย emit invalid call).

2. **case 10 (`search_market_news` drop) — prompt/param nudges ไม่พอ:**
   - inject `[ROUTING PLAN]` directive (SystemMessage) → ยัง drop news 2/3 รอบ
   - + temp=0 บน selection turn → 3/5
   - + reasoning_effort="medium" → 2/5 (แย่ลง)
   สรุป: model deprioritize `search_market_news` ใน 4-tool query อย่างเป็นระบบ ไม่ว่า
   nudge ยังไง — **pattern เดียวกับ stop-loss saga (`POSTMORTEMS.md#guardrails`):
   prompt-only มี diminishing returns → เปลี่ยนเป็น deterministic mechanism.**

3. **coverage-gated loop** (loop กลับ agent จน plan ครบ) → ทำงานได้แต่ **แพง**: เพิ่ม LLM
   call ต่อ query หลายตัว → **ชน Groq daily token limit (TPD 200k/day)** ระหว่าง test.
   → ตัดทิ้ง เปลี่ยนเป็น **deterministic construction** (ข้อ 4).

4. **✅ final: deterministic synthesis of omitted planned calls** — zero extra LLM call,
   guaranteed coverage, testable offline. args ยืมจาก sibling/query. (มี unit test offline
   ครบทุก branch: news→query, ticker→sibling, ticker→regex, P/E→AMD ไม่ใช่ P/E,
   portfolio→None).

5. **🐛 case-5 gap เจอตอน verify (session 2) — "first-turn-only enforcement" เป็น assumption
   ที่ผิด:** upper-bound drop เดิมทำเฉพาะ turn แรก. แต่สำหรับ P/E query, model **re-request
   `get_stock_price` บน synthesis turn** (semantic pull เดิมกลับมาหลัง `get_stock_financials`
   คืนค่า) → หลุด filter → case 5 over-fetch อีก (isolate 3/3: plan ถูก `[financials]` แต่
   output มี price ทุกรอบ). last session ที่ได้ 5/5 clean คือ model บังเอิญไม่ re-request บน
   synthesis — variance ปิดบัง gap ไว้. **แก้:** ย้าย upper-bound drop ออกนอก `first_turn`
   guard → drop off-plan **ทุก turn**; lower-bound synthesis ยังอยู่ first-turn only.
   บทเรียน: plan enforcement ต้อง apply ต่อเนื่องทุก LLM turn ไม่ใช่แค่ turn แรก เพราะ
   semantic pull เกิดได้ทุก generation ไม่ใช่แค่ตอน tool selection.

### Fallback / กัน under-trigger
- planner exception / คืน list ว่าง → plan = ALL tools (revert greedy, ปลอดภัยกว่า under)
- intersect plan กับชื่อ tool จริงเสมอ (กัน hallucinated name)
- synthesize ไม่ได้ (ไม่มี args ให้ยืม เช่น portfolio tool ไม่มี sibling) → skip + log warning

### Nondeterminism
planner + selection turn = temp=0; synthesis = temp=0.2 (guardrail verified ที่ setting นี้
ไม่แตะ). routing = plan (deterministic reconcile) → ไม่พึ่ง executor variance อีก.

### ⚠️ Residual risk ที่ต้อง verify ตอน token reset
- synthesized tool_call id = `plan_<name>` (client-supplied) — OpenAI-compatible API รับ
  arbitrary id ปกติ แต่ยังไม่ได้ verify กับ Groq end-to-end (blocked by TPD limit)
- planner nondeterminism บน case ที่ยังไม่ได้ isolate — ควรรัน suite เต็มตอน quota reset

## Acceptance — สถานะ ณ session 2 (verification)

- [x] `_filter_stoploss` ยัง trigger — `test_stoploss_filter_triggers` **PASSED** (session 2)
- [x] ไม่มี deprecation warning ตอน import/build — verified (`python -W always`)
- [x] synthesized tool_call id `plan_<name>` — **Groq ยอมรับ end-to-end** (STEP 1 probe,
  no 400; ผ่าน tools node + synthesis turn). residual risk #1 = **RESOLVED**.
- [x] case-5 gap fix (off-plan drop ทุก turn) — case 5 re-probe → `{get_stock_financials}`
  เท่านั้น, price ถูก drop.
- [x] `test_routing_regression` **11/11 PASSED** (session 2, post-fix) — case 5 ✅, case 8 ✅
  (fluke ไม่ recur), ครบทุก case.
- [x] synthesized news arg quality (STEP 3, อ่าน LangSmith trace) — raw query ให้ NVDA news
  ที่ relevant เทียบเท่า model-generated query. **ไม่ต้องแก้ `_synthesize_call`.**
- [ ] **STEP 4 adversarial guardrail retest — DEFERRED ไปวันถัดไป** (budget ~23k < 25k).
  agent node เขียนใหม่ → ต้อง retest: (4a) stop-loss compound ผ่าน track_portfolio 3 รอบ,
  (4b) persona separation ผ่าน analyze_portfolio_risk, (4c) `_filter_stoploss` trigger บน
  response จริง. อ้าง `docs/POSTMORTEMS.md#guardrails`.
- [ ] `test_case5_consistency` (financials-only + ไม่มี price, 5/5) — **DEFERRED** (redundant
  บางส่วน เพราะ case 5 verified clean แล้ว แต่ควรรันยืนยัน 5 รอบ).

**⚠️ ห้าม distill (CHANGELOG/ARCHITECTURE/DECISIONS/CLAUDE.md) จนกว่า STEP 4 จะเขียว** —
ต่อให้ 11/11 ผ่านแล้ว (คำสั่ง session 2).

**TODO วันถัดไป (fresh 200k budget):**
```
# STEP 4 adversarial guardrail (~15-20k):
#   4a "AMD เสี่ยงสุดไหม และควรตั้ง stop-loss ไหม" (portfolio: streamlit-test-001) ×3
#   4b '{"NVDA":5000,"AMD":3000}' + "พอร์ตนี้เสี่ยงแค่ไหน"
#   4c ดู log [stoploss-filter] trigger
# + test_case5_consistency:
PYTHONPATH=. pytest tests/test_routing_regression.py::test_case5_consistency -s -v
```

## Note สำหรับ distill ตอนจบ

- `test_routing_regression.py`: เดิม case 5 เป็น known-limitation (allow EXTRA get_stock_price).
  ต้องอัปเดตให้ case 5 เป็น pass ปกติ (ลบ `_KNOWN_CASE_1IDX` special-case) — ไม่งั้น "11/11"
  จะยังนับด้วย logic เดิมที่ยกเว้น case 5.
