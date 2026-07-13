"""
Routing regression tests — mirrors notebook Cell 27/29 logic.
Runs against the production agent in src/agent/core.py (not the notebook).

Run:
    uv run pytest tests/test_routing_regression.py -s -v
"""

import sys
import time
import uuid

import pytest
from groq import RateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import tracing_context

from src.agent.core import build_agent
from src.config import ls_client, tracer

# Windows UTF-8 stdout guard — Thai text needs this when running outside uvicorn
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROUTING_TESTS = [
    # ---- A) NEWS ควรถูกเรียก (under-trigger guard) ----
    {
        "query": "NVDA มีข่าวอะไรล่าสุดบ้าง",
        "expected": {"search_market_news"},
        "why": "ข่าวตรง ๆ — ถ้าไม่เรียก = under-trigger / ตอบจากความจำ (ผิด system prompt)",
    },
    {
        "query": "ทำไมหุ้น TSLA ร่วงช่วงนี้",
        "expected": {"search_market_news"},
        "why": "'ทำไม' = ต้องการ context เชิงเหตุการณ์ ไม่ใช่ตัวเลข",
    },
    {
        "query": "analyst มองหุ้น AMD ยังไงบ้างตอนนี้",
        "expected": {"search_market_news"},
        "why": "analyst commentary = qualitative → news",
    },
    # ---- B) NEWS ไม่ควรถูกเรียก (over-trigger guard) ----
    {
        "query": "ราคา NVDA ตอนนี้เท่าไหร่",
        "expected": {"get_stock_price"},
        "why": "ตัวเลขล้วน — news ไม่ควรโผล่ (news ช้าสุด เปลือง latency ฟรี)",
    },
    {
        "query": "P/E กับ profit margin ของ AMD",
        "expected": {"get_stock_financials"},
        "why": "fundamentals ล้วน → ห้าม news",
    },
    {
        "query": "ประเมิน risk พอร์ต NVDA 50% AMD 30% TSLA 20%",
        "expected": {"analyze_portfolio_risk"},
        "why": "risk metrics → ห้าม news, ห้ามแตกเป็น get_stock_price 3 ตัว",
    },
    {
        "query": "พอร์ต demo ตอนนี้กำไรขาดทุนเท่าไหร่",
        "expected": {"track_portfolio"},
        "why": "P&L พอร์ตที่มีอยู่ → ห้าม news, ห้าม analyze_portfolio_risk",
    },
    {
        "query": "ตอนนี้ NVDA เป็น trending หรือ mean-reverting",
        "expected": {"get_hurst_exponent"},
        "why": "regime = quant signal มีอยู่แล้ว → ห้ามไป search ข่าวแทน",
    },
    # ---- C) CO-TRIGGER: ต้องเรียกหลาย tool (จุดพังบ่อยสุด) ----
    {
        "query": "ราคา NVDA เท่าไหร่ และมีข่าวอะไรทำให้ขยับ",
        "expected": {"get_stock_price", "search_market_news"},
        "why": "ตัวเลข+ข่าว ต้องได้ทั้งคู่ — ขาดตัวใดตัวนึง = routing เพี้ยน",
    },
    {
        "query": "วิเคราะห์ NVDA แบบเต็ม: ราคา, fundamentals, regime, ข่าว",
        "expected": {"get_stock_price", "get_stock_financials", "get_hurst_exponent", "search_market_news"},
        "why": "UC-1 ขยาย — เคย route 3 tools ถูก ต้องไม่ตกตัวไหนหลัง add news",
    },
    # ---- D) REGRESSION: UC เดิมเป๊ะ ๆ (ก่อนมี news ต้องเหมือนเดิม) ----
    {
        "query": "วิเคราะห์ NVDA ให้หน่อย: ราคา, fundamentals, Hurst",
        "expected": {"get_stock_price", "get_stock_financials", "get_hurst_exponent"},
        "why": "UC-1 ตัวเดิมเป๊ะ — news ไม่ควรแทรกเพราะไม่ได้ขอข่าว",
    },
    # ---- E) CASE 12/13: deterministic pre-route (_plan_override) ----
    {
        "query": "พอร์ต streamlit-test-001: TSLA เสี่ยงสุดไหม และควรตั้ง stop-loss ไหม",
        "expected": {"track_portfolio"},
        "why": "มี portfolio_id จริงใน query — risk/stop-loss phrasing เคยดึง planner "
        "ไปทาง analyze_portfolio_risk ผิด (case 12 gap) _plan_override ต้อง force "
        "track_portfolio ก่อน LLM planner ทำงานเลย",
    },
    {
        "query": "พอร์ตแบบนี้เสี่ยงไหม NVDA 5000 AMD 3000",
        "expected": {"analyze_portfolio_risk"},
        "why": "counter-case กัน over-trigger — ไม่มี portfolio_id จริงใน query "
        "(เป็น what-if ล้วน) _plan_override ต้องไม่ force เข้า track_portfolio",
    },
]

# Build agent once for the whole test session (mirrors notebook agent_graph)
_agent_graph = build_agent()

# Case 5 (1-indexed): "P/E + profit margin" — formerly a known limitation where
# the ReAct agent over-fetched get_stock_price. v2 StateGraph structured planning
# fixes this structurally (planner routes fundamentals-only, executor binds only
# planned tools). It must now pass like any other case — no exemption.
_CASE5_1IDX = 5


def get_called_tools(query: str, max_retries: int = 3) -> set:
    """Run agent on one query and return the set of tool names called.
    Mirrors notebook Cell 27 exactly. Retries on Groq 429 (free-tier TPM limit).

    thread_id is REQUIRED now that the graph compiles with a SqliteSaver checkpointer
    (conversation memory). A fresh uuid per call keeps every routing case single-turn and
    isolated — history from one case must never influence the next one's routing."""
    for attempt in range(max_retries):
        try:
            called = set()
            inputs = {"messages": [HumanMessage(content=query)]}
            config = RunnableConfig(
                callbacks=[tracer],
                configurable={"thread_id": str(uuid.uuid4())},
            )
            for event in _agent_graph.stream(inputs, config=config, stream_mode="values"):
                for msg in event.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            called.add(tc["name"])
            return called
        except RateLimitError:
            wait = 8 * (attempt + 1)  # 8s → 16s → 24s
            print(f"     ⏳ 429 — รอ {wait}s แล้ว retry (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"ยัง 429 หลัง retry {max_retries} ครั้ง — query: {query[:40]}")


def test_routing_regression():
    """Run all 13 routing cases.

    Pass condition: ALL cases match expected exactly (13/13). Case 5 (P/E +
    margin) is no longer exempt — the v2 StateGraph planner fixes the historic
    get_stock_price over-fetch structurally. Cases 12/13 cover the
    deterministic _plan_override pre-route (case 12 gap fix).
    """
    results = []

    with tracing_context(enabled=True, client=ls_client):
        for i, t in enumerate(ROUTING_TESTS, 1):
            actual = get_called_tools(t["query"])
            ok = actual == t["expected"]
            missing = t["expected"] - actual
            extra = actual - t["expected"]
            results.append((i, t, actual, ok, missing, extra))

            status = "✅" if ok else "❌"
            print(f"\n{status} [{i}] {t['query'][:55]}")
            print(f"     expected: {t['expected']}")
            print(f"     actual:   {actual}")
            if missing:
                print(f"     ⚠️ MISSING (under-trigger): {missing}")
            if extra:
                print(f"     ⚠️ EXTRA (over-trigger):    {extra}")
            if not ok:
                print(f"     why this matters: {t['why']}")

            time.sleep(2)  # guard Groq rate limit between queries

    ls_client.flush()

    passed = sum(r[3] for r in results)
    failed = sum(not r[3] for r in results)
    print(f"\n{'=' * 55}")
    print(f"Routing: {passed} passed, {failed} failed / {len(ROUTING_TESTS)}")

    failures = []
    for i, t, actual, ok, missing, extra in results:
        if ok:
            continue
        failures.append(
            f"  Case {i}: {t['query'][:55]!r}\n"
            f"    MISSING={missing}  EXTRA={extra}\n"
            f"    why: {t['why']}"
        )

    if failures:
        pytest.fail("Routing failures:\n" + "\n".join(failures))


def test_case5_consistency():
    """Run case 5 query 5× to verify the v2 fix is deterministic.

    Asserts (all 5 rounds): get_stock_financials is called AND get_stock_price
    is NOT — i.e. the fundamentals-only route holds consistently, not by luck.
    """
    query = ROUTING_TESTS[_CASE5_1IDX - 1]["query"]  # "P/E กับ profit margin ของ AMD"
    print(f"\nเช็ค consistency case {_CASE5_1IDX} (P/E + margin) — expect financials-only:")

    results = []
    with tracing_context(enabled=True, client=ls_client):
        for i in range(1, 6):
            r = get_called_tools(query)
            print(f"  รอบ {i}: {r}")
            results.append(r)
            if i < 5:
                time.sleep(5)  # longer gap — 5 rapid calls stress free-tier quota

    ls_client.flush()

    for i, r in enumerate(results, 1):
        assert "get_stock_financials" in r, (
            f"รอบ {i}: get_stock_financials MISSING from {r} — under-trigger regression"
        )
        assert "get_stock_price" not in r, (
            f"รอบ {i}: get_stock_price present in {r} — over-fetch regression (case 5 fix broke)"
        )


def test_stoploss_filter_triggers():
    """Guardrail safety net must survive the StateGraph refactor — deterministic
    post-processing filter still strips stop-loss/hedging paragraphs and appends
    the fixed refusal. Pure string test, no LLM/network. See
    docs/POSTMORTEMS.md#guardrails (incl. U+2011 Unicode-hyphen gotcha)."""
    from src.agent.core import _REFUSAL_STATEMENT, _filter_stoploss

    # ASCII hyphen
    out = _filter_stoploss(
        "AMD volatility สูงมาก\n\nคุณอาจตั้ง stop-loss เพื่อจำกัดการขาดทุน"
    )
    assert "volatility" in out, "non-stoploss paragraph must be kept"
    assert "จำกัดการขาดทุน" not in out, "stop-loss paragraph must be removed"
    assert _REFUSAL_STATEMENT.strip() in out, "fixed refusal must be appended"

    # U+2011 non-breaking hyphen — must still trigger after normalization
    assert _REFUSAL_STATEMENT.strip() in _filter_stoploss("ควรใช้ stop‑loss ไหม")

    # Clean response passes through unchanged (no false positives)
    clean = "AMD มี P/E 30 และ margin 20%"
    assert _filter_stoploss(clean) == clean
