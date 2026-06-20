from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config import ls_client, OPENAI_API_KEY

# สร้างครั้งเดียวระดับ module — Gemini free tier = 20 req/day หมดเร็ว → OpenAI fallback
_news_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=OPENAI_API_KEY,
).bind(tools=[{"type": "web_search_preview"}])


def _extract_grounding_sources(resp, max_sources: int = 5) -> list[str]:
    """ดึง URL citations จาก OpenAI web_search_preview annotations"""
    annotations = (getattr(resp, "additional_kwargs", None) or {}).get("annotations", [])
    seen, out = set(), []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        citation = ann.get("url_citation", {})
        title = citation.get("title") or citation.get("url", "")
        if title and title not in seen:
            seen.add(title)
            out.append(title)
        if len(out) >= max_sources:
            break
    return out


@tool
def search_market_news(query: str) -> str:
    """USE THIS TOOL when the question needs CURRENT EVENTS, news, analyst
    commentary, or qualitative context that is NOT a number
    (e.g. "มีข่าวอะไรเกี่ยวกับ NVDA", "ทำไมหุ้นร่วง", "analyst มองยังไง",
    earnings reactions, M&A, regulatory news).
    Do NOT call this for numeric data — use get_stock_price / get_stock_financials
    for price, P/E, revenue, margins. Do NOT call for risk metrics —
    use analyze_portfolio_risk. Input: a natural-language search query string."""
    return _search_news_logic(query)


@traceable(
    name="search_market_news",
    run_type="tool",
    tags=["search", "news", "openai"],
    client=ls_client,
)
def _search_news_logic(query: str) -> str:
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Today is {today}. Search for the most recent news, analyst commentary, "
            f"and market-moving events about: {query}\n"
            f"Summarize in 4-6 concise bullet points, prioritizing items from the "
            f"last 2 weeks and including dates. If nothing recent is found, say so "
            f"explicitly instead of returning old/generic info. "
            f"Reply in Thai mixed with English financial terms. "
            f"Do not give price targets or buy/sell recommendations."
        )
        resp = _news_model.invoke(prompt)

        content = resp.content
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        content = (content or "").strip()

        sources = _extract_grounding_sources(resp)
        if sources:
            content += "\n\nSources:\n" + "\n".join(f"  - {s}" for s in sources)

        return content or "No recent news found."

    except Exception as e:
        return f"Error: ดึงข่าวไม่สำเร็จ — {type(e).__name__}: {e}"
