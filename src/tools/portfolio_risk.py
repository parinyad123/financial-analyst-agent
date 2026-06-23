import json
import time

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from langsmith import traceable

from src.config import ls_client
from src.tools.data_provider import provider

TRADING_DAYS = 252
RISK_FREE_RATE = 0.045
MIN_HISTORY_DAYS = 60


@tool
def analyze_portfolio_risk(portfolio: str) -> str:
    """USE THIS TOOL for any portfolio risk assessment question
    (e.g. "ประเมิน risk ของ portfolio", "วิเคราะห์ความเสี่ยงพอร์ต",
    "ถ้าลงทุน X บาทใน NVDA และ Y บาทใน AMD risk เป็นยังไง").
    Computes ALL risk metrics in one call: volatility, Sharpe, Sortino,
    VaR/CVaR 95%, max drawdown, correlation matrix.
    Do NOT call get_stock_price or get_stock_financials for portfolio risk —
    this tool fetches all required data itself.
    Input MUST be a JSON string of investment amounts (any currency, any positive number):
    '{"NVDA": 5000, "AMD": 3000, "TSLA": 2000}'.
    Tool will compute weights automatically. All values must be positive.
    Input MUST be a JSON string — NOT a JSON object.
    Correct:   '{"NVDA": 5000, "AMD": 3000}'
    Incorrect: {"NVDA": 5000, "AMD": 3000}"""
    if not isinstance(portfolio, str):
        portfolio = json.dumps(portfolio)
    return _portfolio_risk_logic(portfolio)


@traceable(
    name="portfolio_risk_analysis",
    run_type="tool",
    tags=["quant", "risk"],
    client=ls_client,
)
def _portfolio_risk_logic(portfolio) -> str:
    # ---------- 1) Parse input ----------
    if isinstance(portfolio, dict):
        raw = portfolio
    else:
        try:
            raw = json.loads(portfolio)
        except (json.JSONDecodeError, TypeError) as e:
            return f'Error: invalid input — ต้องเป็น JSON string เช่น {{"NVDA": 5000, "AMD": 3000}} ({e})'

    if not isinstance(raw, dict) or not raw:
        return "Error: portfolio ว่าง หรือ format ไม่ใช่ {ticker: amount}"

    try:
        raw = {str(t).upper().strip(): float(v) for t, v in raw.items()}
    except (ValueError, TypeError):
        return "Error: ค่าทุกตัวต้องเป็นตัวเลข เช่น 5000 ไม่ใช่ '5000 บาท'"

    if any(v <= 0 for v in raw.values()):
        return "Error: จำนวนเงินต้องเป็นบวกทุกตัว (v1 ยังไม่รองรับ short positions)"

    # ---------- 2) Normalize → weights ----------
    total_amount = sum(raw.values())
    weights_dict = {t: v / total_amount for t, v in raw.items()}
    weight_info = ", ".join(f"{t}: {w*100:.1f}%" for t, w in weights_dict.items())
    tickers = sorted(weights_dict)

    # ---------- 3) Download 1Y daily close (batch — DataProvider covers single-ticker only) ----------
    try:
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)["Close"]
    except Exception as e:
        return f"Error: ดึงข้อมูลจาก yfinance ไม่สำเร็จ — {e}"

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    dead = [t for t in tickers if t not in data.columns or data[t].isna().all()]
    if dead:
        time.sleep(1.5)
        retry = yf.download(dead, period="1y", progress=False, auto_adjust=True)["Close"]
        if isinstance(retry, pd.Series):
            retry = retry.to_frame(name=dead[0])
        for t in dead:
            if t in retry.columns and not retry[t].isna().all():
                data[t] = retry[t]
        dead = [t for t in tickers if t not in data.columns or data[t].isna().all()]

    if dead:
        return (
            f"Error: ไม่พบข้อมูลของ {dead} หลัง retry — "
            f"อาจเป็น ticker ผิด/delisted หรือ API ขัดข้องชั่วคราว"
        )

    data = data[tickers].dropna()
    if len(data) < MIN_HISTORY_DAYS:
        return (
            f"Error: ข้อมูลที่ทุก ticker มีร่วมกันมีแค่ {len(data)} วัน "
            f"(ขั้นต่ำ {MIN_HISTORY_DAYS})"
        )

    # ---------- 4) Returns + portfolio series ----------
    returns = np.log(data / data.shift(1)).dropna()
    weights = np.array([weights_dict[t] for t in data.columns])
    port_returns = returns.values @ weights

    # ---------- 5) Core metrics ----------
    ann_return = port_returns.mean() * TRADING_DAYS
    ann_vol    = port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe     = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else float("nan")

    downside = port_returns[port_returns < 0]
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = (ann_return - RISK_FREE_RATE) / (downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sortino_str = f"{sortino:.2f}"
    else:
        sortino_str = "N/A (แทบไม่มีวันติดลบใน 1Y)"

    var_95  = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()

    cum  = np.exp(np.cumsum(port_returns))
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak
    max_dd = dd.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")
    calmar_str = f"{calmar:.2f}" if calmar == calmar else "N/A"

    per_vol     = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    per_vol_str = "\n".join(f"  {t}: {v*100:.1f}%" for t, v in per_vol.items())

    corr_str = (
        returns.corr().round(2).rename_axis(index=None, columns=None).to_string()
        if len(tickers) >= 2 else "N/A (single asset)"
    )

    # ---------- 6) Ulcer Index — pain metric รวม severity + duration ----------
    ulcer = np.sqrt(np.mean(dd ** 2))

    # ---------- 7) Drawdown Duration ----------
    under_peak = (cum < peak).astype(int)
    episodes, count = [], 0
    for v in under_peak:
        if v == 1:
            count += 1
        else:
            if count > 0:
                episodes.append(count)
                count = 0
    if count > 0:
        episodes.append(count)

    dd_dur_str = (
        f"median={int(np.median(episodes))}d, max={max(episodes)}d"
        if episodes else "no drawdown periods in 1Y"
    )

    # ---------- 8) Rolling Correlation 60d — จับ tail dependency ----------
    rolling_corr_str = (
        "Rolling Corr (last 60d):\n"
        + returns.tail(60).corr().round(2).rename_axis(index=None, columns=None).to_string()
        if len(tickers) >= 2 else "N/A (single asset)"
    )

    # ---------- 10) Risk Contribution to Portfolio Variance ----------
    # MCR[i] = w[i] * (Σw)[i] / (w^T Σw) — sums to 1.0 by construction
    cov_matrix = returns.cov().values          # sample cov (ddof=1), order = data.columns
    sigma_w    = cov_matrix @ weights          # Σw: cov of each asset with portfolio
    port_var   = float(weights @ sigma_w)      # w^T Σw = portfolio variance (daily)

    if port_var > 0:
        risk_contrib = weights * sigma_w / port_var
        rc_lines = "\n".join(
            f"  {t}: {rc*100:.1f}%" for t, rc in zip(data.columns, risk_contrib)
        )
        rc_sum = float(sum(risk_contrib))
        risk_contrib_str = f"{rc_lines}\n  (sanity: sum = {rc_sum*100:.4f}%)"
    else:
        risk_contrib_str = "N/A (zero portfolio variance)"

    # ---------- 9) Alpha + Beta vs SPY — CAPM market-model ----------
    # yf.download() เพื่อให้ index tz-naive ตรงกับ portfolio data (yf.Ticker().history() คืน tz-aware)
    alpha_beta_str = "N/A (SPY fetch failed)"
    try:
        spy_raw  = yf.download("SPY", period="1y", progress=False, auto_adjust=True)["Close"]
        if isinstance(spy_raw, pd.DataFrame):
            spy_raw = spy_raw.squeeze()
        spy_ret  = np.log(spy_raw / spy_raw.shift(1)).dropna()
        port_s   = pd.Series(port_returns, index=returns.index)
        # strip tz if present on either side — safety net for yfinance version changes
        if getattr(port_s.index, "tz", None) is not None:
            port_s.index = pd.to_datetime(port_s.index.date)
        if getattr(spy_ret.index, "tz", None) is not None:
            spy_ret.index = pd.to_datetime(spy_ret.index.date)
        common   = port_s.index.intersection(spy_ret.index)
        if len(common) >= 30:
            p_arr = port_s.loc[common].values
            s_arr = spy_ret.loc[common].values
            beta_spy    = np.cov(p_arr, s_arr)[0, 1] / (np.var(s_arr) + 1e-9)
            alpha_daily = np.mean(p_arr) - beta_spy * np.mean(s_arr)
            alpha_ann   = alpha_daily * TRADING_DAYS
            alpha_beta_str = f"Alpha (ann, vs SPY): {alpha_ann*100:+.2f}% | Beta: {beta_spy:.3f}"
        else:
            alpha_beta_str = f"N/A (only {len(common)} common dates with SPY — need ≥30)"
    except Exception as e:
        alpha_beta_str = f"N/A (error: {e})"

    return (
        f"Investment amounts: {raw}\n"
        f"Computed weights: {weight_info}\n"
        f"Period: 1Y daily ({len(returns)} trading days)\n"
        f"Annualized Return: {ann_return*100:+.1f}%\n"
        f"Annualized Volatility: {ann_vol*100:.1f}%\n"
        f"Sharpe Ratio: {sharpe:.2f} (rf = {RISK_FREE_RATE*100:.1f}%)\n"
        f"Sortino Ratio: {sortino_str}\n"
        f"Calmar Ratio: {calmar_str}\n"
        f"VaR 95% (daily): {var_95*100:.2f}% | CVaR 95% (daily): {cvar_95*100:.2f}%\n"
        f"Max Drawdown: {max_dd*100:.1f}%\n"
        f"Ulcer Index: {ulcer*100:.2f}%\n"
        f"Drawdown Duration: {dd_dur_str}\n"
        f"{alpha_beta_str}\n"
        f"Per-Ticker Volatility (annualized):\n{per_vol_str}\n"
        f"Risk Contribution to Variance (1Y):\n{risk_contrib_str}\n"
        f"Pearson Correlation Matrix (1Y):\n{corr_str}\n"
        f"{rolling_corr_str}"
    )
