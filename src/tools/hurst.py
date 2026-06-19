import numpy as np
import yfinance as yf
from langchain_core.tools import tool
from langsmith import traceable

from src.config import ls_client


@tool
def get_hurst_exponent(ticker: str) -> str:
    """Calculate Hurst exponent to detect market regime.
    Use when asked about trend/momentum, market regime, mean-reversion potential,
    or quantitative signal quality (IC, IR). Do NOT use for current price or
    fundamentals — those have dedicated tools."""
    return _calc_hurst_logic(ticker)


@traceable(
    name="calc_hurst_exponent",
    run_type="tool",
    tags=["quant", "regime-detection"],
    client=ls_client,
)
def _calc_hurst_logic(ticker: str) -> str:
    try:
        hist = yf.Ticker(ticker.upper()).history(period="1y")["Close"]
        returns = np.log(hist / hist.shift(1)).dropna().values

        lags = range(2, 20)
        rs_values = []
        for lag in lags:
            segments = [returns[i : i + lag] for i in range(0, len(returns) - lag, lag)]
            rs_list = [
                (
                    np.max(np.cumsum(s - np.mean(s)))
                    - np.min(np.cumsum(s - np.mean(s)))
                )
                / np.std(s)
                for s in segments
                if np.std(s) > 0
            ]
            if rs_list:
                rs_values.append(np.mean(rs_list))

        hurst = np.polyfit(
            np.log(list(lags)[: len(rs_values)]), np.log(rs_values), 1
        )[0]

        if hurst > 0.55:
            regime = "Trending — momentum strategies work"
        elif hurst < 0.45:
            regime = "Mean-Reverting — RSI/Bollinger strategies work"
        else:
            regime = "Random Walk — harder to predict"

        def _rs_hurst(w):
            lgs = list(range(2, min(10, len(w) // 2)))
            rv = []
            for lg in lgs:
                segs = [
                    w[i : i + lg]
                    for i in range(0, len(w) - lg, lg)
                    if len(w[i : i + lg]) == lg
                ]
                if segs:
                    rv.append(
                        np.mean(
                            [
                                (
                                    np.max(np.cumsum(s - np.mean(s)))
                                    - np.min(np.cumsum(s - np.mean(s)))
                                )
                                / (np.std(s) + 1e-9)
                                for s in segs
                            ]
                        )
                    )
            return (
                np.polyfit(np.log(lgs[: len(rv)]), np.log(rv), 1)[0]
                if len(rv) >= 3
                else np.nan
            )

        ic_str = ""
        try:
            from scipy import stats as _stats

            hw, fd = 20, 5
            hs = [_rs_hurst(returns[j - hw : j]) for j in range(hw, len(returns) - fd)]
            fwd = [returns[j : j + fd].sum() for j in range(hw, len(returns) - fd)]
            hs_arr, fwd_arr = np.array(hs), np.array(fwd)
            mask = ~(np.isnan(hs_arr) | np.isnan(fwd_arr))
            if mask.sum() >= 20:
                ic, pv = _stats.spearmanr(hs_arr[mask], fwd_arr[mask])
                quality = (
                    "Strong"
                    if abs(ic) >= 0.10
                    else ("Usable" if abs(ic) >= 0.05 else "Weak")
                )
                ic_str = f"\nIC Score (Hurst→5d): {ic:.4f} (p={pv:.3f}) [{quality}]"

                ic_monthly = []
                mstep = 21
                for s in range(0, len(hs_arr) - mstep, mstep):
                    sub = ~(
                        np.isnan(hs_arr[s : s + mstep])
                        | np.isnan(fwd_arr[s : s + mstep])
                    )
                    if sub.sum() >= 5:
                        ic_sub, _ = _stats.spearmanr(
                            hs_arr[s : s + mstep][sub], fwd_arr[s : s + mstep][sub]
                        )
                        if not np.isnan(ic_sub):
                            ic_monthly.append(ic_sub)
                if len(ic_monthly) >= 3:
                    ir = np.mean(ic_monthly) / (np.std(ic_monthly) + 1e-9)
                    ir_q = (
                        "Strong"
                        if abs(ir) >= 1.0
                        else ("Usable" if abs(ir) >= 0.5 else "Weak")
                    )
                    ic_str += f"\nIR (IC/month consistency): {ir:.3f} [{ir_q}]"
        except Exception:
            pass

        rolling_str = ""
        try:
            rw, rstep = 126, 5
            rh = [
                _rs_hurst(returns[i : i + rw])
                for i in range(0, len(returns) - rw, rstep)
            ]
            rh = [h for h in rh if not np.isnan(h)]
            if len(rh) >= 2:
                trend = (
                    "Rising (more trending)"
                    if rh[-1] > rh[0]
                    else "Falling (more mean-reverting)"
                )
                rolling_str = (
                    f"\nRolling Hurst (126d window, step 5d): "
                    f"early={rh[0]:.3f} → recent={rh[-1]:.3f} [{trend}]"
                )
        except Exception:
            pass

        return (
            f"Hurst Exponent ({ticker.upper()}, 1Y): {hurst:.4f}\n"
            f"Regime: {regime}"
            f"{rolling_str}"
            f"{ic_str}"
        )
    except Exception as e:
        return f"Error: {str(e)}"
