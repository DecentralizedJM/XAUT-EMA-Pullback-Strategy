"""
Bybit kline data for XAUT strategy.
Uses Bybit public REST API - no auth required.
Rate limit: 600 req / 5s per IP (public). Retries on 10006 with backoff.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_RATE_LIMIT_CODE = 10006  # Too many visits
BYBIT_RATE_LIMIT_WAIT = 65  # seconds to wait on rate limit (5s window + buffer)


def fetch_klines(
    symbol: str = "XAUTUSDT",
    interval: str = "5",
    limit: int = 200,
    end_ms: Optional[int] = None,
    max_retries: int = 3,
) -> list:
    """Fetch klines from Bybit USDT Perpetual (linear). Retries on rate limit (10006)."""
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_ms:
        params["end"] = end_ms

    for attempt in range(max_retries):
        resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        ret_code = result.get("retCode")
        ret_msg = result.get("retMsg", "")

        if ret_code == 0:
            return result.get("result", {}).get("list", [])

        if ret_code == BYBIT_RATE_LIMIT_CODE or "rate limit" in ret_msg.lower() or "too many" in ret_msg.lower():
            wait = BYBIT_RATE_LIMIT_WAIT
            reset_ts = resp.headers.get("X-Bapi-Limit-Reset-Timestamp")
            if reset_ts:
                try:
                    wait = max(5, int(reset_ts) - int(time.time()))
                except (ValueError, TypeError):
                    pass
            if attempt < max_retries - 1:
                logger.warning("Bybit rate limit, waiting %ds before retry (%d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
        raise RuntimeError(f"Bybit API: {ret_msg or 'Unknown error'}")


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
