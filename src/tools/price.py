import yfinance as yf
from langchain_core.tools import tool
from langsmith import traceable

from src.config import ls_client


@tool
def get_stock_price(ticker: str) -> str:
    """Fetch CURRENT PRICE and trading metrics: price, % change, 52W range,
    market cap, position in 52W range.
    Use ONLY when the user asks about price, % change, or where the stock
    trades in its range. Do NOT call this for fundamentals like revenue,
    net income, profit margin, EPS, or debt — use get_stock_financials for those.
    Do NOT call this when the question is about analyst opinions,
    news, commentary, or qualitative context — use search_market_news instead.
    Do NOT call this alongside get_stock_financials unless the user
    explicitly asks for BOTH price AND fundamentals."""
    return _fetch_stock_price_logic(ticker)


@traceable(
    name="fetch_stock_price",
    run_type="tool",
    tags=["market-data", "yfinance"],
    client=ls_client,
)
def _fetch_stock_price_logic(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="5d")
        if hist.empty:
            return f"No data for {ticker}"

        latest = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) > 1 else latest
        change_pct = ((latest - prev) / prev) * 100
        info = stock.info

        pos_in_range = (latest - info.get("fiftyTwoWeekLow", latest)) / (
            info.get("fiftyTwoWeekHigh", latest)
            - info.get("fiftyTwoWeekLow", latest)
            + 1e-9
        ) * 100

        return (
            f"Ticker: {ticker.upper()}\n"
            f"Price: ${latest:.2f} (Change: {change_pct:+.2f}%)\n"
            f"52W Range: ${info.get('fiftyTwoWeekLow', 'N/A')} – ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"P/E (TTM): {info.get('trailingPE', 'N/A')} | Forward P/E: {info.get('forwardPE', 'N/A')}\n"
            f"Market Cap: ${info.get('marketCap', 0) / 1e9:.1f}B\n"
            f"Position in 52W range: {pos_in_range:.0f}%"
        )
    except Exception as e:
        return f"Error: {str(e)}"
