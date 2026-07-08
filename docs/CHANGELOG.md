# Changelog / Build order history

ไฟล์นี้เก็บ build order ที่ทำเสร็จแล้ว — เป็น history เท่านั้น ไม่ใช่ active task tracker
(active remaining items อยู่ที่ `CLAUDE.md` ส่วน Current status)

---

## v1 Complete ✅
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

## v1.5 Complete ✅
- [x] DataProvider Protocol + YFinanceProvider (Cell 3.5)
- [x] Rolling Hurst (126d window, step 5d) ใน `_calc_hurst_logic`
- [x] IR = mean(IC_monthly) / std(IC_monthly) ใน `_calc_hurst_logic`
- [x] Rolling Correlation 60d ใน Cell F
- [x] Ulcer Index + Drawdown Duration ใน Cell F
- [x] Alpha/Beta vs SPY ใน Cell F
- [x] Bug 5 resolved by removal: Cell G/H ลบออกจาก notebook

## Production Migration ✅ Complete
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

## Streamlit UI ✅ Complete
- [x] streamlit_app.py — 3 tabs (ถามทั่วไป / วิเคราะห์ Risk พอร์ต / ติดตามพอร์ต)
- [x] Quick-question buttons ทุก tab
- [x] Two-tier display (`_split_response` — summary + expander)
- [x] Dynamic ticker/position rows ด้วย `st.session_state`
- [x] Connection error handling ("เปิด uvicorn main:app ก่อนใช้งาน")
- [x] Verified ผ่าน browser จริง — เจอและแก้ 2 bugs (ดู `docs/POSTMORTEMS.md#streamlit-bugs`)
- [x] UX สำหรับ user ที่ไม่รู้ศัพท์การเงิน: quick-question buttons + two-tier display, verified
  ผ่าน 2 response shape (price-only สั้น, full multi-tool analysis ยาว)

## Risk Contribution Analysis ✅ Complete (tool logic + guardrail)
- [x] `_portfolio_risk_logic` section 10 — Risk Contribution to Variance
- [x] `_track_portfolio_logic` section 5 — Risk Contribution (MV-weighted) + dead ticker exclusion
- [x] scripts/test_risk_contribution.py — standalone test (ไม่ผ่าน notebook)
- [x] Verified ทั้งสอง tools: sanity check sum=100.0000%, ผลตีความถูกตามทฤษฎี, dead ticker handling ถูกต้อง
- [x] System prompt guardrail — ห้าม directive "ควรขาย/ต้องลด" ตรง
- [x] System prompt guardrail (round 1a) — ห้าม stop-loss/hedging action แม้ไม่ระบุราคา
- [x] track_portfolio compound question drift → deterministic post-processing filter
  (`_filter_stoploss()`) + Unicode hyphen normalization — รายละเอียดเต็ม: `docs/POSTMORTEMS.md#guardrails`
- [x] Regression retest หลังเพิ่ม guardrail ทุกรอบ — 10/11 passed
- [x] correlation/diversification misstatement fix — principle-based guardrail —
  `docs/POSTMORTEMS.md#correlation-misconception`
- [x] emergent tool chaining → moved Pearson + Rolling correlation ตรงเข้า `track_portfolio`
  — `docs/DECISIONS.md#emergent-chaining`

## Remaining (ดู CLAUDE.md ส่วน Current status สำหรับ active list)
- [x] Regression test 11 cases หลังเพิ่ม v1.5 metrics — 10/11 passed
- [x] Risk Contribution guardrail ใน system prompt + regression retest
- [ ] README update (เพิ่ม Risk Contribution + Streamlit ในเอกสาร)
- [ ] Colab badge (ถ้าต้องการ)

## Backlog (non-blocking)
- [ ] `create_react_agent` deprecation warning (LangGraph v1.0 moved to `langchain.agents`) —
  `src/agent/core.py:37` — fix แล้วต้องรัน `tests/test_routing_regression.py` ซ้ำก่อน merge

## v2 Phase 1 — Agent infra ✅ Complete
- [x] StateGraph routing — replaced `create_react_agent` with custom planner→agent→tools graph
  (`src/agent/core.py::build_agent`) — planner classifies tool subset (structured output, temp=0),
  agent reconciles model tool_calls to the plan every turn (drop off-plan + synthesize omitted
  planned calls, zero extra LLM calls) — fixes case 5 (P/E over-fetching get_stock_price)
  structurally instead of via docstring/prompt patching. Also resolves the `create_react_agent`
  deprecation warning as a side effect (no longer imported). รายละเอียด design + journey ที่ลองแล้ว
  พัง: `docs/ARCHITECTURE.md#agent-framework`, `docs/POSTMORTEMS.md#docstring-routing`
- [x] Case 12 fix — deterministic pre-route (`_plan_override` ใน `src/agent/core.py`): query ที่มี
  portfolio_id จริง (word-boundary match กับ DB) → force `track_portfolio` ก่อนถึง LLM planner เลย,
  กัน risk/stop-loss phrasing ดึง planner ไปทาง `analyze_portfolio_risk` ผิด — `docs/POSTMORTEMS.md#case-12`
- [x] Portfolio naming validation (`src/api/schemas.py::validate_new_portfolio_id`) — บังคับใช้เฉพาะ
  ตอนสร้าง portfolio ใหม่ผ่าน `POST /portfolio/positions` (grandfather id เก่าใน DB)
- [x] Streamlit — helper text ใต้ช่อง Portfolio ID (Tab ติดตามพอร์ต) แนะนำใช้ id เดียวกันถามใน Tab ถามทั่วไป
- [x] Routing regression ขยายเป็น 13 cases (เพิ่ม case 12 + counter-case 13) — 13/13 passed
- [x] `test_case5_consistency` (5 รอบ) + adversarial guardrail retest (3 รอบ) + persona separation
  formal confirmation — ทั้งหมดผ่าน, รายละเอียด: `docs/POSTMORTEMS.md#guardrails`