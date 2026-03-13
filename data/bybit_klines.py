"""
Bybit kline data for XAUT strategy.
Uses Bybit public REST API - no auth required.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"


def fetch_klines(
    symbol: str = "XAUTUSDT",
    interval: str = "5",
    limit: int = 200,
    end_ms: Optional[int] = None,
) -> list:
    """Fetch klines from Bybit USDT Perpetual (linear)."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_ms:
        params["end"] = end_ms

    resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)
    resp.raise_for_status()
    result = resp.json()
    if result.get("retCode") != 0:
        raise RuntimeError(f"Bybit API: {result.get('retMsg', 'Unknown error')}")

    return result.get("result", {}).get("list", [])


def fetch_klines_dataframe(
    symbol: str = "XAUTUSDT",
    interval: str = "5",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Fetch latest klines as OHLCV dataframe.
    Columns: open, high, low, close, volume
    """
    candles = fetch_klines(symbol, interval, limit)
    if not candles:
        raise ValueError(f"No kline data from Bybit for {symbol}")

    df = pd.DataFrame(
        candles,
        columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
    )
    df = df.sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_historical_bybit(
    symbol: str = "XAUTUSDT",
    interval: str = "5",
    days: int = 30,
) -> pd.DataFrame:
    """Fetch historical klines from Bybit (paginated)."""
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    current_end = end_ms

    while current_end > start_ms:
        candles = fetch_klines(symbol, interval, limit=1000, end_ms=current_end)
        if not candles:
            break
        all_candles.extend(candles)
        current_end = int(candles[-1][0]) - 1
        if int(candles[-1][0]) <= start_ms:
            break
        time.sleep(0.25)

    if not all_candles:
        raise ValueError(f"No data from Bybit for {symbol}")

    df = pd.DataFrame(
        all_candles,
        columns=["open_time", "open", "high", "low", "close", "volume", "turnover"],
    )
    df = df.sort_values("open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]
