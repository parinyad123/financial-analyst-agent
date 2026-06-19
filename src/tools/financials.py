import yfinance as yf
from langchain_core.tools import tool
from langsmith import traceable

from src.config import ls_client


@tool
def get_stock_financials(ticker: str) -> str:
    """Get fundamental financial metrics for analysis."""
    return _fetch_financials_logic(ticker)


@traceable(
    name="fetch_financials",
    run_type="tool",
    tags=["fundamentals", "yfinance"],
    client=ls_client,
)
def _fetch_financials_logic(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker.upper()).info
        return (
            f"Revenue (TTM): ${info.get('totalRevenue', 0) / 1e9:.1f}B\n"
            f"Net Income: ${info.get('netIncomeToCommon', 0) / 1e9:.1f}B\n"
            f"Profit Margin: {info.get('profitMargins', 0) * 100:.1f}%\n"
            f"Revenue Growth YoY: {info.get('revenueGrowth', 0) * 100:.1f}%\n"
            f"EPS (TTM): ${info.get('trailingEps', 'N/A')}\n"
            f"Debt/Equity: {info.get('debtToEquity', 'N/A')}"
        )
    except Exception as e:
        return f"Error: {str(e)}"
