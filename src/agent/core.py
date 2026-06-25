import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from src.agent.prompts import SYSTEM_PROMPT
from src.config import GROQ_API_KEY, PROJECT_NAME, ls_client, tracer
from src.tools.financials import get_stock_financials
from src.tools.hurst import get_hurst_exponent
from src.tools.news import search_market_news
from src.tools.portfolio_risk import analyze_portfolio_risk
from src.tools.portfolio_track import track_portfolio
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


def build_agent():
    """สร้าง ReAct agent graph — call ครั้งเดียวตอน startup"""
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        reasoning_effort="low",
        api_key=GROQ_API_KEY,
    )
    return create_react_agent(model, _tools, prompt=SYSTEM_PROMPT)


# สร้าง graph ครั้งเดียวระดับ module — reuse ทุก request
_agent_graph = build_agent()


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
) -> dict:
    """Main entry point — groups all sub-runs (LLM calls + tool calls) under 1 parent trace.
    Returns run_id for 1:1 mapping to LangSmith trace URL."""
    config = RunnableConfig(
        run_name=f"query_{analysis_type}_{datetime.now().strftime('%H%M%S')}",
        callbacks=[tracer],                        # explicit tracer — ไม่พึ่ง env
        tags=[analysis_type] + (tickers or []),
        metadata={
            "query": query,
            "tickers": tickers,
            "analysis_type": analysis_type,
        },
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
    }
