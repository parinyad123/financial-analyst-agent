# 📊 Financial Analyst Agent — CLAUDE.md

## Project overview

Physics-informed financial analysis ผสม quantitative signals (Hurst exponent, IC Score, IR)
กับ LLM reasoning ผ่าน ReAct agent + LangSmith observability + SQLite persistence

เป้าหมาย: portfolio project สำหรับสมัครงาน FinTech + tool ใช้งานจริง
หลักการ scope: **จบและ demo ได้ สำคัญกว่าทะเยอทะยานแล้วค้าง**
Positioning: **Explainable Quant Analytics + Agent Orchestration** — ไม่แข่งเรื่อง model complexity

Repo: `parinyad123/financial-analyst-agent`

---

## Current status (v1.5 complete, production migrated)

ทุก tool migrate ไป `src/` แล้ว — **`src/` คือ source of truth เพียงที่เดียว**
Notebook (`financial_analyst_agent.ipynb`) = historical reference เท่านั้น ห้ามใช้ทดลอง feature ใหม่

✅ Done: tools (price/financials/hurst+IC+IR/portfolio_risk/news/track_portfolio),
FastAPI 5 endpoints, Streamlit 3 tabs, Docker, Risk Contribution + guardrails,
routing regression 10/11 (case 5 known limitation)

⏳ v1 wrap-up remaining (ทำหลัง v2 เสร็จ — ตัดสินใจแล้วว่าทำ v2 ก่อน screen recording):
- [ ] README update (Risk Contribution + Streamlit + v2 features)
- [ ] Colab badge (optional)
- [ ] `create_react_agent` deprecation fix (`src/agent/core.py:37`) — ทำพร้อม v2 Phase 1
  เพราะแตะ `src/agent/core.py` อยู่แล้ว ไม่ต้องเปิดไฟล์ซ้ำ

🔲 Open decision: deployment approach (live deploy vs. repo+README+screen recording)
— ดู `docs/DECISIONS.md#deployment`

รายละเอียด build order เต็ม → `docs/CHANGELOG.md`

---

## v2 — กำลังทำ (ลำดับตาม dependency, ดู `docs/DECISIONS.md#v2-priority` สำหรับเหตุผลเต็ม)

**Phase 1 — Agent infra:**
1. [ ] StateGraph routing — implement เสร็จ, core suite 11/11 ✅ — เหลือ:
   guardrail retest (STEP 4) + case5_consistency + distill
1b. [ ] Case 12 routing gap — "พอร์ต {id} + risk phrasing" misroutes ไป UC-2a
   → fix ด้วย deterministic pre-route (ดู docs/v2-stategraph-routing.md)
2. [ ] Conversation memory — ต่อจาก StateGraph, ต้อง retest guardrail เดิมทั้งหมดหลังทำ
3. [ ] `create_react_agent` deprecation fix — ทำพ่วงตอนอยู่ใน core.py

**Phase 2 — Feature ใหม่ (ทำหลัง agent infra นิ่งแล้ว):**
4. [ ] Correlation-based stress test — ต้องมี guardrail กัน false-precision (linear approximation only)
5. [ ] What-if เพิ่ม asset นอกพอร์ต

**Phase 3:** regression เต็ม (single-turn + multi-turn) + sync `CHANGELOG.md`/`ARCHITECTURE.md`
**Phase 4:** README → screen recording → Colab badge

---

## Out of scope ถาวร {#out-of-scope}

รายการนี้ **ตัดสินใจแล้วว่าไม่ทำ** พร้อมเหตุผล — ผ่าน external review 2 รอบมาแล้ว
(ดู `docs/DECISIONS.md#external-review`) ถ้ามีคนเสนอ feature พวกนี้อีก ให้อ่านเหตุผลก่อนพิจารณาใหม่
ไม่ใช่แค่ effort/เวลาไม่พอ แต่ขัดกับ positioning ("Explainable Quant Analytics", "not financial advice")

- **Walk-forward Backtesting** — ต้องการ signal definition + position sizing + transaction cost
  model ก่อน ยังไม่มีสักอย่าง
- **HMM / Bayesian Change Point (regime detection)** — interpretability cost > value สำหรับ
  interview demo (nuance: "Explainable ≠ Simple" — ถ้าจะทำในอนาคตต้อง expose evidence-based
  output ไม่ใช่ latent state ตรงๆ ดู `docs/DECISIONS.md#external-review`)
- **Factor engine ด้วย hardcode weights** — ไม่มี IC validation per-factor = pseudo-quant
- **Causal cross-asset impact analysis** ("ข่าว X กระทบพอร์ต Y เท่าไหร่") — false-precision risk
  เดียวกับ factor engine
- **Portfolio optimization** (Efficient Frontier, Black-Litterman, HRP, Risk Budgeting) — คนละ
  layer จากที่ทำอยู่ (risk analysis ≠ portfolio construction) ข้ามไปทำจะกลายเป็น advisory engine
  ขัดกับ guardrail "not financial advice"
- **Bayesian expected return** — การทำนาย return คือ advice โดยตรง เสี่ยงกว่า factor engine อีก
- **PostgreSQL + multi-user (JWT auth)**
- **Monte Carlo VaR**
- **React frontend**
- **RAG จาก SEC filings**
- **Kalman filter, Shannon entropy**
- **Multi-asset class (forex/crypto/options)** — `get_stock_financials` ผูกกับ equity เท่านั้น,
  benchmark ของ Calmar/IC (QuantaAlpha paper) validate กับ equity เท่านั้น, options ไม่มี data
  source ใน yfinance เลย — คง positioning "Explainable Quant Analytics for equity" ไม่เจือจาง scope

**Tier A infra (data quality layer, stale data detection, cache TTL, market calendar
awareness)** — valid สำหรับ production จริง แต่ priority ต่ำกว่า v2 ที่เพิ่ม explainability/demo
value โดยตรง — อยู่ backlog รอ v3

---

## 🧭 Router — อ่านไฟล์ไหนเมื่อไหร่

**แก้/เพิ่ม guardrail หรือ system prompt** → อ่าน `docs/POSTMORTEMS.md#guardrails` ก่อนเสมอ
มี pattern ที่เคย fail มาแล้ว (stop-loss soft-endorsement, correlation misconception) — อย่าทำซ้ำ
สำเนา SYSTEM_PROMPT เต็ม + annotation ว่าบรรทัดไหนมาจาก drift อะไร → `docs/ARCHITECTURE.md#system-prompt`
(source of truth คือ `src/agent/prompts.py` — แก้โค้ดแล้วต้อง sync สำเนาด้วย)

**แก้ routing / agent tool selection (เช่น case 5)** → อ่าน `docs/DECISIONS.md#routing`
+ `docs/ARCHITECTURE.md#tools` ก่อนแก้ docstring ใดๆ

**เพิ่ม metric ใหม่ใน portfolio_risk หรือ track_portfolio** → อ่าน `docs/POSTMORTEMS.md#risk-contribution`
(มี formula + sanity check pattern + tz-bug ที่เคยเจอ)

**แตะ LangSmith tracing / config.py** → อ่าน `docs/ARCHITECTURE.md#langsmith-binding`
(มี lru_cache gotcha ที่ทำให้ trace หาย)

**แตะ database schema / portfolio persistence** → อ่าน `docs/ARCHITECTURE.md#database`

**แตะ FastAPI endpoints / Streamlit UI** → อ่าน `docs/ARCHITECTURE.md#fastapi-spec`
และ `docs/POSTMORTEMS.md#streamlit-bugs`

**ทำงานใน notebook** → **อย่า** เว้นแต่จะ explicit ทำ algorithm prototype ใหม่ที่ยังไม่ตัดสินใจ
ย้ายเข้า `src/` — ดู `docs/DECISIONS.md#notebook-vs-src`

**เริ่มงาน v2 ใดๆ (StateGraph routing, stress test, conversation memory)**
→ เปิดไฟล์ scratch ใหม่ `docs/v2-<feature-name>.md` แยกต่างหาก
ห้าม work-in-progress notes ไหลเข้า CLAUDE.md หรือไฟล์ docs/ หลัก
จบงานแล้วค่อย distill กลับมาเป็น 2-3 บรรทัดใน CHANGELOG.md / ARCHITECTURE.md

**ไม่แน่ใจว่าทำ feature ซ้ำของเก่าไหม / เคยลองวิธีนี้แล้ว fail หรือยัง**
→ grep `docs/POSTMORTEMS.md` ก่อนเริ่ม โดยเฉพาะถ้าเป็นเรื่อง prompt-only guardrail
(มี case ที่ prompt-only ไม่พอและต้องเปลี่ยนเป็น deterministic filter)

**มีคนเสนอ feature ที่อยู่ใน "Out of scope ถาวร" ด้านบน (เช่น HMM, factor engine, portfolio
optimization)** → อ่าน `docs/DECISIONS.md#external-review` ก่อนพิจารณาใหม่ — มี reasoning
ที่ผ่าน external review 2 รอบแล้วว่าทำไมไม่ทำ อย่า re-open โดยไม่อ่านก่อน

---

## Tech stack (สรุปสั้น — เต็มที่ docs/ARCHITECTURE.md)

LangGraph ReAct + Groq (`gpt-oss-120b`, dev) / Gemini 2.5 Flash (prod) · FastAPI + Pydantic v2
SQLite + async SQLAlchemy · LangSmith (explicit binding) · Streamlit · yfinance · uv · Docker

---

## Critical rules (กฎที่ผิดบ่อย ถ้าไม่ย้ำจะ regress)

1. **ห้ามซ้อน `@tool` + `@traceable` บนฟังก์ชันเดียวกัน** — แยก outer (`@tool`) / inner (`@traceable`)
2. **แก้ system prompt ทุกครั้ง → ต้องรัน `tests/test_routing_regression.py` (11 cases) ซ้ำ**
   ก่อน merge — `PYTHONPATH=.` ต้อง set ก่อนรัน pytest จาก root
3. **Notebook ≠ source of truth** — bugfix ใน `src/` (SPY tz, error messages, persona separation)
   ไม่ sync กลับ notebook อัตโนมัติ
4. **Negative instruction ในอังกฤษแม่นกว่าไทยสำหรับ gpt-oss-120b** — แทรกกฎ critical เป็น
   mixed-language ได้ ไม่ต้องแปลทั้ง prompt
5. **String-matching guardrail (เช่น `_filter_stoploss`) ต้อง normalize Unicode variants**
   (hyphen `U+2011` vs `U+002D` เคยทำให้ filter ไม่ trigger เงียบๆ)
6. **Regression fail ≠ regression จริงเสมอ** — isolate case ที่ fail รันเดี่ยว 3 รอบก่อนสรุป
   ว่าเป็น LLM nondeterministic fluke หรือ regression จริง

---

## Doc structure

```
CLAUDE.md                 ← ไฟล์นี้ — current state + router เท่านั้น
docs/
  ARCHITECTURE.md          ← tech stack, data flow, DB schema, tool decorator pattern,
                              LangSmith binding pattern, FastAPI spec
  DECISIONS.md             ← decision log แบบสรุป (ทำไมเลือก X ไม่ Y) ต่อหัวข้อ
  POSTMORTEMS.md           ← bug fixes ละเอียด + key learnings ทั้งหมด
                              (Bug 1-7, stop-loss drift saga, correlation misconception,
                              risk contribution implementation)
  CHANGELOG.md             ← build order ที่ทำเสร็จแล้ว (v1, v1.5) — history เท่านั้น
  v2-<feature>.md          ← scratch doc ต่อ feature ระหว่างทำ v2 (ลบทิ้งได้หลัง distill)
```