"""
XAUT EMA Pullback - INSTITUTIONAL ML BOT for Mudrex.
Uses the validated ML model and advanced filtering stack.
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
from datetime import datetime, timezone

from config import Config, MudrexConfig, StrategyConfig
from data.bybit_klines import fetch_klines_dataframe
from exchange.mudrex_client import MudrexClient, MudrexAPIError
from strategy.institutional_ml import InstitutionalMLStrategy, Signal

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def round_quantity(qty: float, step: float) -> float:
    if step <= 0: return max(0.0, qty)
    n = round(qty / step)
    return max(step, n * step) if n > 0 else 0.0

def compute_position_params(
    signal,
    equity: float,
    qty_step: float,
    min_order_value: float,
    margin_percent: float,
    leverage: int,
) -> tuple[float, int]:
    """
    Computes position size based on user-defined MARGIN_PERCENT and LEVERAGE.
    Skips the trade if the notional value is under the minimum $8 limit.
    """
    entry = signal.entry_price
    margin_used = equity * (margin_percent / 100.0)
    notional = margin_used * leverage
    
    if notional < min_order_value:
        logger.warning(
            "Skip %s: Notional value ($%.2f) < min_order_value ($%.2f) with %.1f%% margin",
            signal.signal.value.upper(), notional, min_order_value, margin_percent
        )
        return 0.0, leverage

    qty = notional / entry
    qty = round_quantity(qty, qty_step)
    
    if qty * entry < min_order_value:
        logger.warning(
            "Skip %s: Rounded notional ($%.2f) < min_order_value ($%.2f)",
            signal.signal.value.upper(), qty * entry, min_order_value
        )
        return 0.0, leverage

    return qty, leverage

def _log_signal_heartbeat(diag: dict) -> None:
    """Log signal check heartbeat with evaluation details."""
    reason = diag.get("reason", "unknown")
    if reason == "insufficient_data":
        logger.info(
            "[HEARTBEAT] Signal check | bars=%d | reason=%s",
            diag.get("bars", 0), reason
        )
        return
    logger.info(
        "[HEARTBEAT] Signal check | close=%.2f ema21=%.2f rsi=%.1f atr=%.2f | "
        "score_L=%d score_S=%d | tap_L=%s tap_S=%s | session=%s macro_L=%s macro_S=%s full_moon=%s | "
        "can_L=%s can_S=%s | reason=%s",
        diag.get("close", 0),
        diag.get("ema21", 0),
        diag.get("rsi", 0),
        diag.get("atr", 0),
        diag.get("score_long", 0),
        diag.get("score_short", 0),
        diag.get("tap_long", False),
        diag.get("tap_short", False),
        diag.get("valid_session", False),
        diag.get("macro_long", False),
        diag.get("macro_short", False),
        diag.get("full_moon_avoid", False),
        diag.get("can_long", False),
        diag.get("can_short", False),
        reason,
    )
    if "ml_prob" in diag:
        logger.info(
            "[HEARTBEAT] ML | candidate=%s prob=%.3f threshold=%.3f",
            diag.get("candidate", "-"),
            diag.get("ml_prob", 0),
            diag.get("ml_threshold", 0),
        )


def get_current_position(positions: list, symbol: str) -> Optional[str]:
    for p in positions or []:
        pos_symbol = (p.get("symbol") or p.get("asset_id") or "").upper()
        if symbol.upper() in pos_symbol or pos_symbol in symbol.upper():
            side = (p.get("side") or p.get("order_type") or "").upper()
            if "LONG" in side: return "long"
            if "SHORT" in side: return "short"
    return None

def run(config: Config, paper: bool = False) -> None:
    api_secret = "" if paper else (os.getenv("MUDREX_API_SECRET") or config.mudrex.api_secret)
    if not paper and not api_secret:
        logger.error("Set MUDREX_API_SECRET in .env or config")
        sys.exit(1)

    client = MudrexClient(api_secret) if not paper else None
    strategy = InstitutionalMLStrategy(model_dir="saved_model")

    try:
        config.mudrex.margin_percent = float(os.getenv("MARGIN_PERCENT", config.mudrex.margin_percent))
    except ValueError:
        pass
        
    try:
        config.mudrex.leverage = int(os.getenv("LEVERAGE", config.mudrex.leverage))
    except ValueError:
        pass

    symbol = config.strategy.symbol
    qty_step = config.mudrex.quantity_step
    initial_equity = config.mudrex.initial_equity
    min_order_value = config.mudrex.min_order_value
    margin_percent = config.mudrex.margin_percent
    leverage = config.mudrex.leverage

    MUDREX_SPACING = 0.6
    POLL_INTERVAL = 120

    logger.info(
        "INSTITUTIONAL BOT | Symbol=%s | Mode=%s | ML Threshold=%.2f",
        symbol, "PAPER" if paper else "LIVE", strategy.cfg.get('ml_threshold', 0.35)
    )
    logger.info("Config: %.1f%% Margin, %dx Leverage. Min Notional: $%.2f", 
                margin_percent, leverage, min_order_value)

    if not paper:
        try:
            client._resolve_asset(symbol)
            time.sleep(1)
            logger.info("Bot ready. Monitoring market for signals.")
        except Exception as e:
            logger.warning("Setup warning: %s", e)

    while True:
        try:
            if paper:
                equity = initial_equity
            else:
                try:
                    equity = client.get_futures_balance()
                    if equity <= 0: equity = initial_equity
                except Exception as e:
                    logger.warning("Balance fetch failed: %s; using initial_equity", e)
                    equity = initial_equity
                time.sleep(MUDREX_SPACING)

            # Fetch Bybit data (2s delay to avoid rate limit bursts)
            time.sleep(2)
            df = fetch_klines_dataframe(symbol, interval="5", limit=600)
            if len(df) < 300:
                logger.warning("Insufficient data: %d bars", len(df))
                time.sleep(POLL_INTERVAL)
                continue

            if not paper:
                time.sleep(MUDREX_SPACING)
            positions = client.get_open_positions(symbol) if not paper else []
            position = get_current_position(positions, symbol)

            # Only place trades if flat
            if position:
                logger.info("Position already open: %s. Monitoring...", position)
            else:
                signal, diag = strategy.evaluate(df)
                if signal:
                    qty, lev = compute_position_params(
                        signal, equity, qty_step, min_order_value, margin_percent, leverage
                    )
                    
                    if qty <= 0:
                        logger.debug("Position size 0 (likely minimum order not met). Skipping trade.")
                        time.sleep(POLL_INTERVAL)
                        continue
                    
                    order_type = "LONG" if signal.signal == Signal.LONG else "SHORT"
                    if paper:
                        logger.info(
                            "[PAPER] SIGNAL: %s qty=%.4f lev=%dx prob=%.3f score=%d risk=%.2f%%",
                            order_type, qty, lev, signal.probability, signal.score, signal.risk_pct
                        )
                    else:
                        try:
                            # Re-verify leverage for this order
                            client.set_leverage(symbol, lev, config.mudrex.margin_type)
                            time.sleep(MUDREX_SPACING)
                            
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
                                    "TRADE EXECUTED: %s qty=%.4f lev=%dx entry=%.2f sl=%.2f tp=%.2f",
                                    order_type, qty, lev, signal.entry_price, signal.stop_loss, signal.take_profit
                                )
                            else:
                                logger.error("Trade failed: %s", result)
                        except MudrexAPIError as e:
                            logger.error("API Error: %s", e)
                else:
                    _log_signal_heartbeat(diag)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Loop error: %s", e)
            time.sleep(30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    config = Config(strategy=StrategyConfig(), mudrex=MudrexConfig())
    run(config, paper=args.paper)
