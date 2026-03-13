"""
XAUT EMA Pullback Strategy.
Uses Bybit klines for data, Mudrex for execution.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class TradeSignal:
    signal: Signal
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reason: str


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


class EMAPullbackStrategy:
    def __init__(
        self,
        ema_period: int = 21,
        tap_threshold_pct: float = 0.2,
        stop_loss_buffer_pct: float = 0.5,
        take_profit_rr: float = 2.0,
        risk_per_trade_pct: float = 1.0,
        use_rsi_filter: bool = True,
        use_macd_filter: bool = False,
        first_tap_only: bool = True,
        rsi_period: int = 14,
        rsi_long_min: float = 50.0,
        rsi_short_max: float = 50.0,
    ):
        self.ema_period = ema_period
        self.tap_threshold_pct = tap_threshold_pct
        self.stop_loss_buffer_pct = stop_loss_buffer_pct
        self.take_profit_rr = take_profit_rr
        self.risk_per_trade_pct = risk_per_trade_pct
        self.use_rsi_filter = use_rsi_filter
        self.use_macd_filter = use_macd_filter
        self.first_tap_only = first_tap_only
        self.rsi_period = rsi_period
        self.rsi_long_min = rsi_long_min
        self.rsi_short_max = rsi_short_max
        self._in_trend: bool = False

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema"] = ema(df["close"], self.ema_period)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        macd_line, signal_line, _ = macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        return df

    def _is_in_tap_zone_long(self, row: pd.Series) -> bool:
        ema_val = row["ema"]
        close = row["close"]
        tap_zone = ema_val * (self.tap_threshold_pct / 100)
        return close > ema_val and close <= ema_val + tap_zone

    def _is_in_tap_zone_short(self, row: pd.Series) -> bool:
        ema_val = row["ema"]
        close = row["close"]
        tap_zone = ema_val * (self.tap_threshold_pct / 100)
        return close < ema_val and close >= ema_val - tap_zone

    def evaluate(
        self,
        df: pd.DataFrame,
        equity: float = 10000.0,
        current_position: Optional[str] = None,
    ) -> Optional[TradeSignal]:
        if df.empty or len(df) < max(self.ema_period, 50):
            return None

        df = self._compute_indicators(df)
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else row

        trend_flip = (prev["close"] <= prev["ema"] and row["close"] > row["ema"]) or (
            prev["close"] >= prev["ema"] and row["close"] < row["ema"]
        )
        if trend_flip:
            self._in_trend = True
        if current_position and current_position != "none":
            self._in_trend = False

        long_tap = self._is_in_tap_zone_long(row) and (
            not self.first_tap_only or self._in_trend
        )
        short_tap = self._is_in_tap_zone_short(row) and (
            not self.first_tap_only or self._in_trend
        )

        rsi_ok_long = row["rsi"] > self.rsi_long_min if self.use_rsi_filter else True
        rsi_ok_short = row["rsi"] < self.rsi_short_max if self.use_rsi_filter else True
        macd_ok_long = (
            row["macd"] > row["macd_signal"] if self.use_macd_filter else True
        )
        macd_ok_short = (
            row["macd"] < row["macd_signal"] if self.use_macd_filter else True
        )

        long_signal = long_tap and rsi_ok_long and macd_ok_long
        short_signal = short_tap and rsi_ok_short and macd_ok_short

        close = float(row["close"])
        ema_val = float(row["ema"])

        if long_signal:
            sl = ema_val * (1 - self.stop_loss_buffer_pct / 100)
            risk = abs(close - sl)
            tp = close + risk * self.take_profit_rr
            risk_dollars = equity * (self.risk_per_trade_pct / 100)
            qty = risk_dollars / risk if risk > 0 else 0
            return TradeSignal(
                signal=Signal.LONG,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size=qty,
                reason="EMA pullback long",
            )

        if short_signal:
            sl = ema_val * (1 + self.stop_loss_buffer_pct / 100)
            risk = abs(sl - close)
            tp = close - risk * self.take_profit_rr
            risk_dollars = equity * (self.risk_per_trade_pct / 100)
            qty = risk_dollars / risk if risk > 0 else 0
            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                position_size=qty,
                reason="EMA pullback short",
            )

        return None
