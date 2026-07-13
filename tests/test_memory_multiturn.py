"""
Conversation-memory tests (v2 Phase 1 #2) — multi-turn behaviour of the SqliteSaver
checkpointer added in src/agent/core.py.

Covers what single-turn routing regression cannot:
  1. pronoun / follow-up resolution  ("แล้วข่าวล่ะ" after asking NVDA's price)
  2. thread isolation                (a different thread_id must not see the history)
  3. stale-number guardrail          (re-asking a price must re-call the tool, never
                                      re-quote the figure fetched several turns ago)

Run:
    PYTHONPATH=. uv run pytest tests/test_memory_multiturn.py -s -v
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_agent_graph = build_agent()


def _turn(query: str, thread_id: str, max_retries: int = 3) -> tuple[set, str]:
    """Send one turn on `thread_id`; return (tools called this turn, final answer).

    Only tool_calls produced AFTER this turn's HumanMessage are counted — with memory the
    state also replays earlier turns' AIMessages, and counting those would make every turn
    look like it re-called the previous turn's tools."""
    for attempt in range(max_retries):
        try:
            inputs = {"messages": [HumanMessage(content=query)]}
            config = RunnableConfig(
                callbacks=[tracer],
                configurable={"thread_id": thread_id},
            )
            final_messages = []
            for event in _agent_graph.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event:
                    final_messages = event["messages"]

            # index of THIS turn's human message = last HumanMessage in the final state
            start = max(
                (i for i, m in enumerate(final_messages)
                 if isinstance(m, HumanMessage) and m.content == query),
                default=0,
            )
            called, answer = set(), ""
            for msg in final_messages[start:]:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        called.update(tc["name"] for tc in msg.tool_calls)
                    if isinstance(msg.content, str) and msg.content.strip():
                        answer = msg.content
            return called, answer
        except RateLimitError:
            wait = 8 * (attempt + 1)
            print(f"     ⏳ 429 — รอ {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"ยัง 429 หลัง retry {max_retries} ครั้ง — query: {query[:40]}")


def test_followup_resolves_from_history():
    """Turn 2 says only "แล้วข่าวล่ะ" — the ticker exists only in turn 1's history.
    Without memory the planner cannot know what "ข่าว" is about; with it, the follow-up
    must route to search_market_news (and must NOT re-fetch the price it already gave)."""
    thread = str(uuid.uuid4())
    with tracing_context(enabled=True, client=ls_client):
        t1, _ = _turn("ราคา NVDA ตอนนี้เท่าไหร่", thread)
        print(f"\n  turn1 'ราคา NVDA' → {t1}")
        assert "get_stock_price" in t1, f"turn 1 ต้องเรียก price — got {t1}"

        time.sleep(2)
        t2, ans2 = _turn("แล้วข่าวล่ะ", thread)
        print(f"  turn2 'แล้วข่าวล่ะ' → {t2}")

    assert "search_market_news" in t2, (
        f"follow-up ต้อง resolve เป็นข่าวของ NVDA จาก history — got {t2}"
    )
    assert "NVDA" in ans2.upper(), "คำตอบ turn 2 ควรอ้างถึง NVDA (resolve จาก history สำเร็จ)"


def test_threads_are_isolated():
    """Same follow-up phrasing on a FRESH thread must not inherit the other thread's
    ticker — proves thread_id actually scopes memory (Tab 2/3 rely on this to stop a
    previous portfolio's figures leaking into a new one)."""
    thread_a, thread_b = str(uuid.uuid4()), str(uuid.uuid4())
    with tracing_context(enabled=True, client=ls_client):
        _turn("ราคา TSLA ตอนนี้เท่าไหร่", thread_a)
        time.sleep(2)
        _, ans_b = _turn("แล้วข่าวล่ะ", thread_b)  # fresh thread — no TSLA anywhere

    print(f"\n  fresh-thread answer (first 120): {ans_b[:120]!r}")
    assert "TSLA" not in ans_b.upper(), (
        "thread ใหม่ต้องไม่เห็น history ของ thread เดิม — TSLA รั่วข้าม thread"
    )


def test_stale_number_forces_tool_recall():
    """The hazard conversation memory itself creates: SYSTEM_PROMPT's "every number must
    come from a tool call" was written for single-turn, so nothing stopped the model from
    re-quoting a price fetched several turns earlier as the current one.
    CONVERSATION_CONTEXT_NOTE (src/agent/core.py) exists to block this — asking for the
    price again must call get_stock_price again, not read it out of history."""
    thread = str(uuid.uuid4())
    with tracing_context(enabled=True, client=ls_client):
        t1, _ = _turn("ราคา AMD ตอนนี้เท่าไหร่", thread)
        assert "get_stock_price" in t1, f"turn 1 ต้องเรียก price — got {t1}"
        time.sleep(2)

        _turn("AMD เป็น trending หรือ mean-reverting", thread)  # intervening turn
        time.sleep(2)

        t3, _ = _turn("ราคา AMD ตอนนี้เท่าไหร่อีกที", thread)
        print(f"\n  turn3 (re-ask price) → {t3}")

    assert "get_stock_price" in t3, (
        f"ถามราคาซ้ำต้องเรียก tool ใหม่ ไม่ใช่ดึงตัวเลขเก่าจาก history — got {t3}"
    )


def test_persona_separation_survives_multiturn():
    """Guardrail retest under memory: a what-if portfolio stays HYPOTHETICAL on the
    follow-up turn too. The original bug ("your portfolio lost 23%") is the exact failure
    memory could reintroduce by carrying the portfolio across turns."""
    thread = str(uuid.uuid4())
    with tracing_context(enabled=True, client=ls_client):
        _turn("วิเคราะห์ความเสี่ยงพอร์ต: '{\"NVDA\": 5000, \"AMD\": 3000}'", thread)
        time.sleep(2)
        _, ans2 = _turn("แล้ว drawdown ล่ะ น่ากังวลไหม", thread)

    print(f"\n  follow-up answer (first 200): {ans2[:200]!r}")
    lowered = ans2.lower()
    banned = ["พอร์ตของคุณขาดทุน", "คุณขาดทุน", "you lost", "your portfolio lost"]
    hits = [b for b in banned if b.lower() in lowered]
    assert not hits, (
        f"persona drift — พอร์ต what-if ถูกพูดถึงเป็นพอร์ตจริงใน follow-up: {hits}"
    )
