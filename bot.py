"""
XAUT EMA Pullback - Production bot for Mudrex.
Mudrex executes trades (broker of Bybit); Bybit klines for prices.
Deployable to Railway.
"""

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from config import Config, MudrexConfig, StrategyConfig
from data.bybit_klines import fetch_klines_dataframe
from exchange.mudrex_client import MudrexClient, MudrexAPIError
from strategy.ema_pullback import EMAPullbackStrategy, Signal

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def round_quantity(qty: float, step: float) -> float:
    """Round quantity to exchange step."""
    if step <= 0:
        return max(0.0, qty)
    n = round(qty / step)
    return max(step, n * step) if n > 0 else 0.0


def compute_position_and_leverage(
    signal,
    equity: float,
    qty_step: float,
    min_order_value: float,
    max_leverage: int,
    risk_pct: float,
) -> tuple[float, int]:
    """
    Autoscale position size and leverage for low balances.
    Ensures: notional >= min_order_value, margin <= equity, leverage capped.
    Returns (quantity, leverage).
    """
    entry = signal.entry_price
    risk_per_unit = abs(signal.entry_price - signal.stop_loss)
    if risk_per_unit <= 0:
        return 0.0, 1

    risk_amount = equity * (risk_pct / 100)
    qty = risk_amount / risk_per_unit
    notional = qty * entry

    # Bump position to meet minimum order value
    if notional < min_order_value:
        qty = min_order_value / entry
        notional = qty * entry

    qty = round_quantity(qty, qty_step)
    if qty <= 0:
        return 0.0, 1

    notional = qty * entry
    if equity <= 0:
        return qty, 1

    # Leverage needed: margin = notional/leverage <= equity => leverage >= notional/equity
    min_lev = notional / equity
    leverage = max(1, min(max_leverage, math.ceil(min_lev)))
    return qty, leverage


def get_current_position(positions: list, symbol: str) -> Optional[str]:
    """Return 'long' | 'short' | None based on open position."""
    for p in positions or []:
        pos_symbol = (p.get("symbol") or p.get("asset_id") or "").upper()
        if symbol.upper() in pos_symbol or pos_symbol in symbol.upper():
            side = (p.get("side") or p.get("order_type") or "").upper()
            if "LONG" in side:
                return "long"
            if "SHORT" in side:
                return "short"
    return None


def run(config: Config, paper: bool = False) -> None:
    """Main trading loop."""
    api_secret = "" if paper else (os.getenv("MUDREX_API_SECRET") or config.mudrex.api_secret)
    if not paper and not api_secret:
        logger.error("Set MUDREX_API_SECRET in .env or config")
        sys.exit(1)

    client = MudrexClient(api_secret) if not paper else None
    strategy = EMAPullbackStrategy(
        ema_period=config.strategy.ema_period,
        tap_threshold_pct=config.strategy.tap_threshold_pct,
        stop_loss_buffer_pct=config.strategy.stop_loss_buffer_pct,
        take_profit_rr=config.strategy.take_profit_rr,
        risk_per_trade_pct=config.strategy.risk_per_trade_pct,
        use_rsi_filter=config.strategy.use_rsi_filter,
        use_macd_filter=config.strategy.use_macd_filter,
        first_tap_only=config.strategy.first_tap_only,
        rsi_period=config.strategy.rsi_period,
        rsi_long_min=config.strategy.rsi_long_min,
        rsi_short_max=config.strategy.rsi_short_max,
    )

    symbol = config.strategy.symbol
    leverage = config.mudrex.leverage
    qty_step = config.mudrex.quantity_step
    initial_equity = config.mudrex.initial_equity
    min_order_value = config.mudrex.min_order_value
    max_leverage = config.mudrex.max_leverage

    # Rate limits: Mudrex 2 req/s; Bybit 600/5s per IP (we do 1 kline/cycle)
    MUDREX_SPACING = 0.6  # seconds between Mudrex calls
    POLL_INTERVAL = 90  # seconds between loop cycles (Bybit + Mudrex friendly)

    logger.info(
        "XAUT EMA Pullback | Symbol=%s | Leverage=%dx (fixed) | MinOrder=$%.0f | Data=Bybit | Mode=%s",
        symbol, leverage, min_order_value, "PAPER" if paper else "LIVE",
    )

    if not paper:
        try:
            client._resolve_asset(symbol)
            time.sleep(1)  # Mudrex 2 req/s
            client.set_leverage(symbol, leverage, config.mudrex.margin_type)
            logger.info("Leverage set to %dx (fixed)", leverage)
            time.sleep(2)  # Mudrex rate limit recovery before loop
        except MudrexAPIError as e:
            logger.warning("Leverage set failed (may already be set): %s", e)

    poll_interval = POLL_INTERVAL
    last_bar_count = 0

    while True:
        try:
            # Fetch equity
            if paper:
                equity = initial_equity
            else:
                try:
                    equity = client.get_futures_balance()
                    if equity <= 0:
                        equity = initial_equity
                        logger.debug("Using initial_equity fallback: %.0f", equity)
                except Exception as e:
                    logger.warning("Balance fetch failed: %s; using initial_equity", e)
                    equity = initial_equity
                time.sleep(MUDREX_SPACING)  # Mudrex 2 req/s

            # Fetch Bybit klines (retries on rate limit inside fetch_klines)
            df = fetch_klines_dataframe(symbol, interval="5", limit=200)
            if len(df) < 60:
                logger.warning("Insufficient kline data: %d bars", len(df))
                time.sleep(poll_interval)
                continue

            if not paper:
                time.sleep(MUDREX_SPACING)  # Mudrex 2 req/s before positions
            positions = client.get_open_positions(symbol) if not paper else []
            position = get_current_position(positions, symbol)

            # Evaluate strategy
            signal = strategy.evaluate(df, equity=equity, current_position=position)
            last_close = float(df["close"].iloc[-1]) if not df.empty else 0
            if signal:
                qty, lev = compute_position_and_leverage(
                    signal,
                    equity,
                    qty_step,
                    min_order_value,
                    max_leverage,
                    config.strategy.risk_per_trade_pct,
                )
                if not config.mudrex.auto_leverage:
                    lev = leverage  # Fixed 25x
                if qty < qty_step:
                    logger.warning("Position size %.6f below min step, skipping", qty)
                else:
                    order_type = "LONG" if signal.signal == Signal.LONG else "SHORT"
                    if paper:
                        logger.info(
                            "[PAPER] SIGNAL: %s qty=%.4f lev=%dx entry=%.2f sl=%.2f tp=%.2f",
                            order_type, qty, lev, signal.entry_price, signal.stop_loss, signal.take_profit,
                        )
                    else:
                        try:
                            result = client.place_market_order(
                                symbol=symbol,
                                order_type=order_type,
                                quantity=qty,
                                leverage=lev,
                                order_price=signal.entry_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                reduce_only=False,
                            )
                            if result.get("success"):
                                logger.info(
                                    "ORDER: %s qty=%.4f lev=%dx entry=%.2f sl=%.2f tp=%.2f",
                                    order_type,
                                    qty,
                                    lev,
                                    signal.entry_price,
                                    signal.stop_loss,
                                    signal.take_profit,
                                )
                            else:
                                logger.error("Order failed: %s", result)
                        except MudrexAPIError as e:
                            logger.error("Order error: %s", e)
            else:
                logger.info(
                    "No signal | close=%.2f position=%s next in %ds",
                    last_close,
                    position or "flat",
                    poll_interval,
                )

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.exception("Loop error: %s", e)
            time.sleep(30)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Paper trade (no real orders)")
    args = parser.parse_args()

    config = Config(
        strategy=StrategyConfig(),
        mudrex=MudrexConfig(),
    )
    run(config, paper=args.paper)


if __name__ == "__main__":
    main()
