"""
Configuration for XAUT EMA Pullback on Mudrex.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    symbol: str = "XAUTUSDT"
    timeframe: str = "5m"
    ema_period: int = 21
    tap_threshold_pct: float = 0.2
    stop_loss_buffer_pct: float = 0.5
    take_profit_rr: float = 2.0
    risk_per_trade_pct: float = 1.0
    use_rsi_filter: bool = True
    use_macd_filter: bool = False
    first_tap_only: bool = True
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0


@dataclass
class MudrexConfig:
    api_secret: str = ""
    leverage: int = 10
    margin_percent: float = 5.0
    margin_type: str = "ISOLATED"
    quantity_step: float = 0.001  # XAUT min step
    initial_equity: float = 10000.0  # Fallback if balance fetch fails
    min_order_value: float = 8.0  # Minimum notional $8


@dataclass
class Config:
    strategy: StrategyConfig
    mudrex: MudrexConfig
