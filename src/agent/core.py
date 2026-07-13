import logging
import re
import sqlite3
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from src.agent.prompts import SYSTEM_PROMPT
from src.config import CHECKPOINT_DB_PATH, GROQ_API_KEY, PROJECT_NAME, ls_client, tracer
from src.tools.financials import get_stock_financials
from src.tools.hurst import get_hurst_exponent
from src.tools.news import search_market_news
from src.tools.portfolio_risk import analyze_portfolio_risk
from src.tools.portfolio_track import _list_portfolio_ids, track_portfolio
from src.tools.price import get_stock_price

_logger = logging.getLogger(__name__)

_STOPLOSS_KEYWORDS = frozenset([
    "stop-loss", "stop loss", "stoploss",
    "hedge", "hedging", "เฮดจ์",
    "เครื่องมือป้องกัน",
])

_REFUSAL_STATEMENT = (
    "\n\nส่วนเรื่อง stop-loss/hedging — "
    "ฉันไม่สามารถแนะนำได้ เพราะเป็น actionable trading decision "
    "ที่อยู่นอกขอบเขตของระบบนี้"
)


_UNICODE_HYPHENS = "‐‑‒–—−"  # ‐‑‒–—−


def _normalize_hyphens(text: str) -> str:
    for ch in _UNICODE_HYPHENS:
        text = text.replace(ch, "-")
    return text


def _filter_stoploss(response: str) -> str:
    """Deterministic safety net — remove any paragraph that mentions
    stop-loss/hedging and replace with the fixed refusal statement.
    Normalizes Unicode hyphen variants before matching (model outputs U+2011).
    Logs each trigger so prompt guardrail leakage is visible in logs."""
    normalized_full = _normalize_hyphens(response.lower())
    if not any(kw in normalized_full for kw in _STOPLOSS_KEYWORDS):
        return response

    paragraphs = re.split(r"\n{2,}", response)
    clean, n_removed = [], 0
    for para in paragraphs:
        if any(kw in _normalize_hyphens(para.lower()) for kw in _STOPLOSS_KEYWORDS):
            n_removed += 1
            _logger.warning(
                "[stoploss-filter] removed paragraph | preview: %s",
                para[:120].replace("\n", " "),
            )
        else:
            clean.append(para)

    _logger.warning(
        "[stoploss-filter] prompt guardrail leaked — removed %d paragraph(s)",
        n_removed,
    )
    return "\n\n".join(clean) + _REFUSAL_STATEMENT


_tools = [
    get_stock_price,
    get_stock_financials,
    get_hurst_exponent,
    analyze_portfolio_risk,
    track_portfolio,
    search_market_news,
]
_TOOLS_BY_NAME = {t.name: t for t in _tools}
_ALL_TOOL_NAMES = set(_TOOLS_BY_NAME)
_TOOL_ORDER = [t.name for t in _tools]  # stable order for the routing directive


# ---------------------------------------------------------------------------
# StateGraph routing (v2) — structured planning replaces create_react_agent's
# semantic (greedy) tool selection. A dedicated planner classifies which tools
# are needed BEFORE execution; the agent node then binds ONLY those tools, so
# over-fetch (e.g. case 5: P/E query pulling get_stock_price) is structurally
# impossible, not merely discouraged by prompt/docstring.
# See docs/v2-stategraph-routing.md and docs/DECISIONS.md#routing.
# ---------------------------------------------------------------------------

# NOTE: separate from SYSTEM_PROMPT (which must not be edited) — this is a new
# classification prompt used only by the planner node, never sent to the agent.
PLANNER_PROMPT = """You are a routing planner for a financial analysis agent.
Given the user's query, decide EXACTLY which tools are required to answer it.
Return only the tool names — do not answer the question itself.

Available tools:
- get_stock_price: CURRENT price / quote, daily change %, 52-week range,
  market cap. Use ONLY when the user explicitly asks about current price or
  quote ("ราคา", "price", "เท่าไหร่", "quote").
- get_stock_financials: Fundamentals — P/E ratio (trailing & forward),
  profit margin, revenue, net income, EPS, revenue growth, debt-to-equity.
  Valuation ratios and margins are FUNDAMENTALS and live here.
- get_hurst_exponent: Regime / trend detection — trending vs mean-reverting,
  Hurst exponent, IC score, IR. Use for "trending", "mean-reverting", "regime".
- analyze_portfolio_risk: Hypothetical / what-if portfolio risk from a set of
  {ticker: amount or weight}. Use for portfolio risk given allocations.
- track_portfolio: P&L of an EXISTING saved portfolio referenced by name/id
  ("พอร์ต demo", "portfolio X"). Use for tracking already-held positions.
- search_market_news: News, analyst opinions/views, and qualitative
  "why did it move / ทำไม...ร่วง/ขึ้น" questions.

Critical routing rules:
1. P/E ratio, valuation multiples, and profit margin are FUNDAMENTALS →
   get_stock_financials ONLY. Do NOT add get_stock_price. Even though P/E is
   mathematically derived from price, get_stock_financials already returns P/E;
   the price tool is NOT needed and must NOT be included unless the user also
   explicitly asks for the current price/quote.
2. Include get_stock_price ONLY when the user explicitly asks for current
   price/quote — not merely because a price-derived ratio was mentioned.
3. Qualitative questions (news, analyst views, "ทำไม/why did it move") →
   search_market_news only. Do not add get_stock_price unless price is
   explicitly requested too.
4. Include EVERY tool the query explicitly requests. E.g.
   "ราคา, fundamentals, regime, ข่าว" → all four corresponding tools.
5. Portfolio risk given weights/amounts → analyze_portfolio_risk (never split
   into per-ticker get_stock_price calls).
6. Existing/tracked portfolio P&L → track_portfolio.
7. FOLLOW-UP QUESTIONS: if a [CONVERSATION HISTORY] block is provided and the query
   refers back to it ("ตัวนั้น", "มัน", "อันนี้", "แล้ว...ล่ะ", "เทียบกับ X หน่อย",
   "then what about..."), first resolve WHICH ticker/entity the query means from that
   history, then pick tools for that entity. Example: "ราคา NVDA เท่าไหร่" followed by
   "แล้วข่าวล่ะ" → the follow-up is about NVDA news → search_market_news (NOT get_stock_price
   again — the user already has the price).
When genuinely uncertain, prefer including a relevant tool over omitting it,
but NEVER include get_stock_price for a pure fundamentals question (rule 1)."""


# Injected as a runtime SystemMessage ONLY on turns that have prior history — same pattern
# as [ROUTING PLAN] below. SYSTEM_PROMPT is NOT edited (see docs/ARCHITECTURE.md#system-prompt).
# Guards a hazard that memory itself introduces: SYSTEM_PROMPT's "every number must come from a
# tool call" rule was written for single-turn, so nothing stops the model from re-quoting a price
# fetched 4 turns ago as if it were current. English negative instruction — more reliable for
# gpt-oss-120b than Thai (critical rule #4).
CONVERSATION_CONTEXT_NOTE = (
    "[CONVERSATION CONTEXT] Earlier messages in this thread are snapshots from the moment "
    "they were fetched — they are NOT current.\n"
    "ถ้าผู้ใช้ถามข้อมูลปัจจุบัน (ราคา / ข่าว / metrics) ต้องเรียก tool ใหม่เสมอ\n"
    "NEVER cite a number from a previous turn as if it were the current value. "
    "If you reference an earlier figure, say explicitly when it was fetched."
)


class RoutingPlan(BaseModel):
    """Which tools are required to answer the user's query."""

    tools: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of the allowed tool names required to answer the query: "
            "get_stock_price, get_stock_financials, get_hurst_exponent, "
            "analyze_portfolio_risk, track_portfolio, search_market_news."
        ),
    )


_TICKER_TOOLS = {"get_stock_price", "get_stock_financials", "get_hurst_exponent"}
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")  # NVDA/AMD/TSLA/SPY — skips single letters (P, E)


class AgentState(TypedDict):
    """messages carries the conversation (reduced by add_messages so agent
    AIMessages + tool_calls are visible to callers scanning event['messages']).
    plan holds the planner's chosen tool names for this turn."""

    messages: Annotated[list, add_messages]
    plan: list[str]


def _plan_override(query: str) -> list[str] | None:
    """Deterministic pre-route: if the query contains a word-boundary match of
    an EXISTING portfolio_id, force `track_portfolio` and skip the LLM planner
    entirely. Fixes case 12 — "พอร์ต <id>: ... เสี่ยงสุดไหม" pulls the planner
    toward analyze_portfolio_risk because the risk/stop-loss phrasing is the
    dominant semantic signal, even though a real saved portfolio is named
    explicitly. An exact id match is a stronger, unambiguous signal than any
    prompt wording could express. See docs/DECISIONS.md#routing."""
    try:
        portfolio_ids = _list_portfolio_ids()
    except Exception as exc:
        _logger.warning("[plan_override] portfolio id lookup failed (%s) — skip override", exc)
        return None
    for pid in portfolio_ids:
        if re.search(rf"(?<![\w-]){re.escape(pid)}(?![\w-])", query):
            _logger.info(
                "[plan_override] matched portfolio_id=%r in query → forcing track_portfolio", pid
            )
            return ["track_portfolio"]
    return None


def _last_human_query(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return messages[-1].content if messages else ""


def _last_human_index(messages: list) -> int:
    """Index of the current turn's HumanMessage (-1 if none). `> 0` ⇒ this thread has
    prior history, which is what gates CONVERSATION_CONTEXT_NOTE and the planner's
    history block."""
    return max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )


def _recent_dialogue(messages: list, max_turns: int = 6, clip: int = 300) -> str:
    """Compact Human/AI transcript of PRIOR turns for the planner.

    ToolMessages are excluded on purpose: they are large raw dumps (a track report is
    thousands of chars) and the planner only needs to know what was discussed, not the
    figures. The trailing HumanMessage is excluded too — that is the query being planned."""
    prior = messages[: _last_human_index(messages)] if _last_human_index(messages) > 0 else []
    lines = []
    for m in prior:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {str(m.content)[:clip]}")
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            lines.append(f"Assistant: {m.content[:clip]}")
    return "\n".join(lines[-max_turns:])


# Rough char→token ratio; exact counting is not worth an extra tokenizer dependency here.
_MAX_HISTORY_TOKENS = 6000


def _approx_tokens(msgs) -> int:
    return sum(len(str(getattr(m, "content", "") or "")) for m in msgs) // 4


def _trim_history(messages: list) -> list:
    """Cap conversation context so accumulated tool dumps cannot blow the window.

    start_on="human" is load-bearing: it guarantees the window never begins with an
    orphan ToolMessage whose parent AIMessage(tool_calls) was trimmed away — Groq
    rejects that pairing with a 400. If the budget cannot fit even one turn, fall back
    to the current turn intact rather than dropping the user's question."""
    trimmed = trim_messages(
        messages,
        token_counter=_approx_tokens,
        max_tokens=_MAX_HISTORY_TOKENS,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    if trimmed:
        return trimmed
    idx = _last_human_index(messages)
    return messages[idx:] if idx >= 0 else messages


def _extract_ticker(query: str, sibling_calls: list) -> str | None:
    """Ticker for a synthesized ticker-tool call: prefer one the model already
    used in a sibling call (authoritative), else fall back to the first
    uppercase 2-5 letter token in the query."""
    for tc in sibling_calls:
        t = (tc.get("args") or {}).get("ticker")
        if t:
            return t
    m = _TICKER_RE.search(query or "")
    return m.group(0) if m else None


def _synthesize_call(name: str, sibling_calls: list, query: str) -> dict | None:
    """Build a tool_call for a planned tool the executor omitted, borrowing
    arguments from sibling calls / the user query. Returns None when arguments
    cannot be safely inferred (then we skip rather than call with bad args).
    Deterministic (no LLM) → zero extra tokens, unlike a retry/loop."""
    if name in _TICKER_TOOLS:
        ticker = _extract_ticker(query, sibling_calls)
        args = {"ticker": ticker} if ticker else None
    elif name == "search_market_news":
        args = {"query": query} if query else None
    elif name == "analyze_portfolio_risk":
        p = next((tc["args"].get("portfolio") for tc in sibling_calls
                  if (tc.get("args") or {}).get("portfolio")), None)
        args = {"portfolio": p} if p else None
    elif name == "track_portfolio":
        p = next((tc["args"].get("portfolio_id") for tc in sibling_calls
                  if (tc.get("args") or {}).get("portfolio_id")), None)
        args = {"portfolio_id": p} if p else None
    else:
        args = None
    if args is None:
        return None
    # id must be unique & match the ToolMessage ToolNode emits; derived from name
    # so it is stable/deterministic within a single AIMessage (no random needed).
    return {"name": name, "args": args, "id": f"plan_{name}", "type": "tool_call"}


def build_agent():
    """สร้าง custom StateGraph agent — call ครั้งเดียวตอน startup.

    planner → agent → tools loop (replaces deprecated create_react_agent).
    planner (temp=0, structured output) classify tool subset; agent reconcile
    tool_calls ให้เท่ากับ plan บน first turn — drop off-plan (กัน over-fetch,
    case 5) + synthesize planned tool ที่ model ข้าม (กัน under-call, case 10).
    routing = plan แบบ deterministic, ไม่มี LLM call เพิ่ม. ดู
    docs/v2-stategraph-routing.md."""

    def _llm(temperature: float, reasoning_effort: str = "low") -> ChatGroq:
        return ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_key=GROQ_API_KEY,
        )

    # temp=0.2 for synthesis (answer quality); temp=0 for the deterministic
    # routing classifier and the first tool-selection turn.
    model = _llm(0.2)
    planner_model = _llm(0.0).with_structured_output(RoutingPlan)

    tool_node = ToolNode(_tools)

    def planner(state: AgentState) -> dict:
        """Classify which tools are needed. Writes only `plan` — never appends
        to `messages`, so the structured-output tool call does NOT leak into
        the tool set that callers count from event['messages']. A deterministic
        portfolio_id match (_plan_override) short-circuits the LLM call entirely
        when it applies — cheaper and immune to semantic drift (case 12).

        With conversation memory, prior turns are passed as a separate history block so
        follow-ups ("แล้วข่าวล่ะ") can be resolved to the right ticker. _plan_override
        still sees ONLY the latest human query — feeding it history would let a
        portfolio_id mentioned several turns ago force track_portfolio forever."""
        query = _last_human_query(state["messages"])
        override = _plan_override(query)
        if override:
            return {"plan": override}
        history = _recent_dialogue(state["messages"])
        planner_msgs = [SystemMessage(content=PLANNER_PROMPT)]
        if history:
            planner_msgs.append(
                SystemMessage(content=f"[CONVERSATION HISTORY]\n{history}")
            )
        planner_msgs.append(HumanMessage(content=query))
        try:
            plan = planner_model.invoke(planner_msgs)
            chosen = [n for n in (plan.tools or []) if n in _ALL_TOOL_NAMES]
        except Exception as exc:  # planner failure → safe fallback (see below)
            _logger.warning("[planner] failed (%s) — falling back to all tools", exc)
            chosen = []
        # Empty plan (failure or genuinely no match) → bind ALL tools. Reverting
        # to old greedy behavior is safer than under-triggering; case 5 relies on
        # a non-empty, price-free plan, which the planner produces normally.
        if not chosen:
            chosen = sorted(_ALL_TOOL_NAMES)
        _logger.info("[planner] query=%r → plan=%s", query[:80], chosen)
        return {"plan": chosen}

    # Bind ALL tools once — the model must see every schema, otherwise gpt-oss
    # sometimes emits a call to an *unbound* tool (e.g. get_stock_price for a P/E
    # query — the semantic pull behind case 5) and Groq rejects the whole request
    # with a 400 (tool_use_failed). We enforce the plan on the OUTPUT instead.
    # Turn 1 (tool selection) uses temp=0 for determinism; synthesis uses temp=0.2.
    select_model = _llm(0.0).bind_tools(_tools)
    synth_model = model.bind_tools(_tools)

    def agent(state: AgentState) -> dict:
        """Invoke the model (all tools bound), then reconcile its tool_calls to the
        planner's decision — making routing deterministically equal to the plan:
          • upper bound (EVERY turn): drop off-plan tool_calls → fixes over-fetch.
            Must run on every turn, not just the first: for a P/E query the model
            re-requests get_stock_price on the SYNTHESIS turn (semantic pull), so a
            first-turn-only filter lets it leak (the case-5 gap found in verify).
          • lower bound (first turn only): synthesize any planned tool the model
            omitted, borrowing args from sibling calls / the query → fixes
            intermittent under-call (case 10: search_market_news dropped on a
            4-tool query). Deterministic → ZERO extra LLM calls.
        Runtime-only — SYSTEM_PROMPT is unchanged. See docs/v2-stategraph-routing.md."""
        plan = set(state.get("plan", []))
        planned_list = [n for n in _TOOL_ORDER if n in plan]
        # First turn of THIS user question = no tool has run for it yet.
        first_turn = bool(state["messages"]) and isinstance(state["messages"][-1], HumanMessage)
        # Prior turns exist ⇒ the stale-number hazard applies (see CONVERSATION_CONTEXT_NOTE).
        has_history = _last_human_index(state["messages"]) > 0

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if has_history:
            messages.append(SystemMessage(content=CONVERSATION_CONTEXT_NOTE))
        if first_turn and planned_list:
            messages.append(
                SystemMessage(
                    content=(
                        "[ROUTING PLAN] For this query, call exactly these tools "
                        f"and no others: {', '.join(planned_list)}. Call every tool "
                        "in this list before answering; do not call any tool not in it."
                    )
                )
            )
        messages += _trim_history(state["messages"])

        response = (select_model if first_turn else synth_model).invoke(messages)

        if not (isinstance(response, AIMessage) and plan):
            return {"messages": [response]}

        raw = response.tool_calls or []
        kept = [tc for tc in raw if tc["name"] in plan]  # upper bound — every turn
        dropped = [tc["name"] for tc in raw if tc["name"] not in plan]
        if dropped:
            _logger.info("[agent] dropped off-plan tool calls: %s", dropped)

        if first_turn:  # lower bound — synthesize omitted planned tools (first turn only)
            called = {tc["name"] for tc in kept}
            query = _last_human_query(state["messages"])
            for name in planned_list:
                if name in called:
                    continue
                synth = _synthesize_call(name, kept, query)
                if synth:
                    _logger.info("[agent] synthesized omitted planned call: %s args=%s",
                                 name, synth["args"])
                    kept.append(synth)
                else:
                    _logger.warning("[agent] could not synthesize planned tool %r "
                                    "(no inferable args) — leaving uncovered", name)

        if kept != raw:
            # Rebuild cleanly so no raw off-plan call lingers in additional_kwargs
            # (would 400 when re-serialized to Groq). Kept + synthesized calls carry
            # ids → ToolNode produces matching ToolMessages.
            response = AIMessage(content=response.content, tool_calls=kept)
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    # check_same_thread=False: run_financial_agent is sync and callers run it via
    # asyncio.to_thread, so the connection is touched from worker threads.
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)


# สร้าง graph ครั้งเดียวระดับ module — reuse ทุก request
_agent_graph = build_agent()


def thread_has_history(session_id: str | None) -> bool:
    """True if this thread already holds messages — lets callers avoid re-injecting
    context they already put in the thread on turn 1 (see /portfolio/{id}/ask)."""
    if not session_id:
        return False
    try:
        snapshot = _agent_graph.get_state({"configurable": {"thread_id": session_id}})
    except Exception as exc:
        _logger.warning("[memory] get_state failed for %r (%s)", session_id, exc)
        return False
    return bool(snapshot and snapshot.values and snapshot.values.get("messages"))


@traceable(
    name="financial_analyst_agent",
    run_type="chain",
    tags=["agent", "financial-analysis"],
    project_name=PROJECT_NAME,
    client=ls_client,          # explicit binding — ไม่พึ่ง env var (lru_cache issue)
)
def run_financial_agent(
    query: str,
    tickers: list[str] | None = None,
    analysis_type: str = "general",
    session_id: str | None = None,
) -> dict:
    """Main entry point — groups all sub-runs (LLM calls + tool calls) under 1 parent trace.
    Returns run_id for 1:1 mapping to LangSmith trace URL.

    session_id = the checkpointer thread. Omitting it mints a fresh thread per call, which
    reproduces the pre-memory stateless behavior — so existing single-turn callers and the
    routing regression keep their semantics."""
    thread_id = session_id or str(uuid.uuid4())
    config = RunnableConfig(
        run_name=f"query_{analysis_type}_{datetime.now().strftime('%H%M%S')}",
        callbacks=[tracer],                        # explicit tracer — ไม่พึ่ง env
        tags=[analysis_type] + (tickers or []),
        metadata={
            "query": query,
            "tickers": tickers,
            "analysis_type": analysis_type,
            "session_id": thread_id,
        },
        configurable={"thread_id": thread_id},     # required once a checkpointer is compiled in
    )

    # diagnostic: repr() แสดง raw Python string — ถ้าเห็น \u0e?? = unicode ถูก, ถ้าเห็น ? literal = input เพี้ยนตั้งแต่ decode
    print(f"[agent] query repr: {repr(query[:120])}", flush=True)

    inputs = {"messages": [HumanMessage(content=query)]}
    final_response = ""

    for event in _agent_graph.stream(inputs, config=config, stream_mode="values"):
        if "messages" in event:
            last = event["messages"][-1]
            try:
                last.pretty_print()
            except (UnicodeEncodeError, UnicodeDecodeError):
                # fallback เมื่อ terminal encoding แคบ (cp1252/cp874)
                safe = repr(getattr(last, "content", ""))[:200]
                print(f"[{type(last).__name__}] {safe}", flush=True)
            if hasattr(last, "content") and last.content:
                final_response = last.content

    final_response = _filter_stoploss(final_response)
    rt = get_current_run_tree()
    return {
        "query": query,
        "response": final_response,
        "tickers": tickers,
        "analysis_type": analysis_type,
        "run_id": str(rt.id) if rt else None,
        "session_id": thread_id,
    }
