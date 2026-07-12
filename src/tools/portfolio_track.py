import asyncio

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from langsmith import traceable
from sqlalchemy import select

from src.config import ls_client
from src.database.models import Portfolio, Position
from src.database.session import AsyncSessionLocal


async def _load_positions_async(portfolio_id: str):
    """Query DB — คืน (name, positions_list) | raise KeyError ถ้าไม่เจอ"""
    async with AsyncSessionLocal() as session:
        pf = await session.get(Portfolio, portfolio_id)
        if pf is None:
            raise KeyError(portfolio_id)
        result = await session.execute(
            select(Position).where(Position.portfolio_id == portfolio_id)
        )
        rows = result.scalars().all()
        positions = [
            {"ticker": r.ticker, "shares": r.shares, "avg_cost": r.avg_cost}
            for r in rows
        ]
        return pf.name, positions


async def _list_portfolio_ids_async() -> list[str]:
    """คืน list ของ portfolio_id ทั้งหมดใน DB — ใช้สำหรับ error message"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Portfolio.portfolio_id))
        return [row[0] for row in result.all()]


async def _list_portfolios_async() -> list[tuple[str, str, list[str]]]:
    """คืน (portfolio_id, name, [tickers]) ของทุกพอร์ต — ใช้สร้าง dropdown
    เลือกพอร์ตด้วยชื่อ (UI) + ส่ง tickers เป็น context ให้ /ask.
    Query เดียว join positions, group ticker ต่อพอร์ตใน Python (พอร์ตมีไม่กี่ตัว)."""
    async with AsyncSessionLocal() as session:
        pf_result = await session.execute(
            select(Portfolio.portfolio_id, Portfolio.name)
        )
        portfolios = pf_result.all()
        pos_result = await session.execute(
            select(Position.portfolio_id, Position.ticker)
        )
        tickers_by_pf: dict[str, list[str]] = {}
        for pid, ticker in pos_result.all():
            tickers_by_pf.setdefault(pid, []).append(ticker)
        return [
            (pid, name, sorted(set(tickers_by_pf.get(pid, []))))
            for pid, name in portfolios
        ]


def _load_positions(portfolio_id: str):
    return asyncio.run(_load_positions_async(portfolio_id))


def _list_portfolio_ids() -> list[str]:
    return asyncio.run(_list_portfolio_ids_async())


def _list_portfolios() -> list[tuple[str, str, list[str]]]:
    return asyncio.run(_list_portfolios_async())


@tool
def track_portfolio(portfolio_id: str) -> str:
    """USE THIS TOOL for tracking an EXISTING portfolio's performance
    (e.g. "พอร์ตของฉันกำไรเท่าไหร่", "track portfolio", "P&L ตอนนี้",
    "ขาดทุนอยู่เท่าไหร่"). Loads saved positions by portfolio_id,
    fetches current prices, and computes unrealized P&L per position,
    total market value, total P&L, and current weights.
    Do NOT call get_stock_price separately — this tool fetches all prices itself.
    Input: portfolio_id string, e.g. "demo".
    Note: this is different from analyze_portfolio_risk, which assesses
    risk of a HYPOTHETICAL weight allocation before investing."""
    return _track_portfolio_logic(portfolio_id)


@traceable(
    name="track_portfolio",
    run_type="tool",
    tags=["portfolio", "tracking"],
    client=ls_client,
)
def _track_portfolio_logic(portfolio_id: str) -> str:
    # ---------- 1) Load positions ----------
    try:
        portfolio_id = str(portfolio_id).strip()
        pf_name, positions = _load_positions(portfolio_id)
    except KeyError:
        available = ", ".join(_list_portfolio_ids()) or "none"
        return (
            f"Error: ไม่พบ portfolio_id '{portfolio_id}' — "
            f"portfolio ที่มีอยู่: {available}. ให้ถาม user ว่าหมายถึงอันไหน"
        )

    if not positions:
        return f"Portfolio '{pf_name}' ({portfolio_id}) ยังไม่มี position ใด ๆ"

    for p in positions:
        if p.get("shares", 0) <= 0 or p.get("avg_cost", 0) <= 0:
            return (
                f"Error: position {p.get('ticker', '?')} มี shares/avg_cost "
                f"ไม่ถูกต้อง ({p}) — ข้อมูลใน DB อาจเสียหาย"
            )

    # ---------- 2) Batch fetch ราคาปัจจุบัน ----------
    tickers = sorted({p["ticker"].upper() for p in positions})
    try:
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)["Close"]
    except Exception as e:
        return f"Error: ดึงราคาจาก yfinance ไม่สำเร็จ — {e}"

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    prices = {}
    for t in tickers:
        if t in data.columns and data[t].dropna().size > 0:
            prices[t] = float(data[t].dropna().iloc[-1])
        else:
            prices[t] = None

    # ---------- 3) P&L per position ----------
    lines, warnings_list = [], []
    total_mv = total_cost = 0.0

    for p in positions:
        t = p["ticker"].upper()
        shares, avg_cost = p["shares"], p["avg_cost"]
        cost_basis = shares * avg_cost

        if prices[t] is None:
            warnings_list.append(
                f"  ⚠️ {t}: price unavailable (delisted/ticker ผิด?) — "
                f"{shares} shares @ cost ${avg_cost:.2f} ไม่รวมในยอด"
            )
            continue

        mv = shares * prices[t]
        pnl = mv - cost_basis
        pnl_pct = pnl / cost_basis * 100
        total_mv += mv
        total_cost += cost_basis

        lines.append(
            f"  {t}: {shares:g} shares @ avg ${avg_cost:.2f} → "
            f"now ${prices[t]:.2f} | MV ${mv:,.2f} | "
            f"P&L {pnl:+,.2f} ({pnl_pct:+.1f}%)"
        )

    if total_mv == 0:
        return (
            f"Portfolio '{pf_name}': ดึงราคาไม่ได้สักตัวเดียว\n"
            + "\n".join(warnings_list)
        )

    # ---------- 4) Current weights ----------
    weight_lines = []
    for p in positions:
        t = p["ticker"].upper()
        if prices[t] is not None:
            w = (p["shares"] * prices[t]) / total_mv * 100
            weight_lines.append(f"  {t}: {w:.1f}%")

    total_pnl = total_mv - total_cost
    total_pnl_pct = total_pnl / total_cost * 100

    # ---------- 5) Risk Contribution & Correlation (1Y history, current MV weights) ----------
    # Uses market-value weights (not cost basis) — reflects actual current exposure
    active_tickers = [t for t in tickers if prices[t] is not None]
    risk_contrib_str = "N/A"
    corr_str = "N/A"
    rolling_corr_str = "N/A"
    if len(active_tickers) >= 2:
        try:
            hist = yf.download(active_tickers, period="1y", progress=False, auto_adjust=True)["Close"]
            if isinstance(hist, pd.Series):
                hist = hist.to_frame(name=active_tickers[0])
            hist = hist[active_tickers].dropna()
            if len(hist) >= 30:
                hist_returns = np.log(hist / hist.shift(1)).dropna()
                corr_str = (
                    hist_returns.corr().round(2)
                    .rename_axis(index=None, columns=None).to_string()
                )
                rolling_corr_str = (
                    "Rolling Corr (last 60d):\n"
                    + hist_returns.tail(60).corr().round(2)
                    .rename_axis(index=None, columns=None).to_string()
                )
                shares_map = {p["ticker"].upper(): p["shares"] for p in positions}
                mv_active = np.array([shares_map[t] * prices[t] for t in active_tickers])
                mv_weights = mv_active / mv_active.sum()
                cov_matrix = hist_returns.cov().values
                sigma_w = cov_matrix @ mv_weights
                port_var_rc = float(mv_weights @ sigma_w)
                if port_var_rc > 0:
                    risk_contrib = mv_weights * sigma_w / port_var_rc
                    rc_lines = "\n".join(
                        f"  {t}: {rc*100:.1f}%" for t, rc in zip(active_tickers, risk_contrib)
                    )
                    rc_sum = float(sum(risk_contrib))
                    risk_contrib_str = f"{rc_lines}\n  (sanity: sum = {rc_sum*100:.4f}%)"
                else:
                    risk_contrib_str = "N/A (zero portfolio variance)"
        except Exception as e:
            risk_contrib_str = f"N/A (error: {e})"
    elif len(active_tickers) == 1:
        risk_contrib_str = f"  {active_tickers[0]}: 100.0% (single asset)"
        corr_str = "N/A (single asset)"
        rolling_corr_str = "N/A (single asset)"

    out = (
        f"Portfolio: {pf_name} (id: {portfolio_id})\n"
        f"Positions:\n" + "\n".join(lines) + "\n"
        f"Total Market Value: ${total_mv:,.2f}\n"
        f"Total Cost Basis: ${total_cost:,.2f}\n"
        f"Total Unrealized P&L: {total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)\n"
        f"Current Weights (by market value):\n" + "\n".join(weight_lines) + "\n"
        f"Risk Contribution to Variance (1Y, MV-weighted):\n{risk_contrib_str}\n"
        f"Pearson Correlation Matrix (1Y):\n{corr_str}\n"
        f"{rolling_corr_str}"
    )
    if warnings_list:
        out += "\nWarnings:\n" + "\n".join(warnings_list)
    return out
