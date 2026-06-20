import pandas as pd
import yfinance as yf
from typing import Protocol


class DataProvider(Protocol):
    def get_history(self, ticker: str, period: str) -> pd.DataFrame: ...
    def get_info(self, ticker: str) -> dict: ...


class YFinanceProvider:
    def get_history(self, ticker: str, period: str) -> pd.DataFrame:
        return yf.Ticker(ticker).history(period=period)

    def get_info(self, ticker: str) -> dict:
        return yf.Ticker(ticker).info


provider = YFinanceProvider()
