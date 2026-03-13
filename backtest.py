"""
Backtest XAUT EMA Pullback using Bybit klines.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.bybit_klines import fetch_historical_bybit
from strategy.ema_pullback import EMAPullbackStrategy, Signal


def run_backtest(
    symbol: str = "XAUTUSDT",
    days: int = 365,
    initial_equity: float = 10000.0,
) -> None:
    print(f"Fetching {days} days of {symbol} 5m data from Bybit...")
    df = fetch_historical_bybit(symbol, interval="5", days=days)
    print(f"Loaded {len(df)} candles.")

    strategy = EMAPullbackStrategy(
        ema_period=21,
        tap_threshold_pct=0.2,
        stop_loss_buffer_pct=0.5,
        take_profit_rr=2.0,
        risk_per_trade_pct=1.0,
        use_rsi_filter=True,
        use_macd_filter=False,
        first_tap_only=True,
    )

    equity = initial_equity
    position = None
    entry_equity = initial_equity
    sl = tp = 0.0
    trades = []

    for i in range(50, len(df)):
        window = df.iloc[: i + 1]
        signal = strategy.evaluate(window, equity=equity, current_position=position)

        row = df.iloc[i]
        high, low = row["high"], row["low"]
        risk_amount = entry_equity * 0.01 if position else 0

        if position == "long":
            if low <= sl:
                equity += -risk_amount
                trades.append({"pnl": -risk_amount, "exit": "SL"})
                position = None
            elif high >= tp:
                equity += risk_amount * 2.0
                trades.append({"pnl": risk_amount * 2.0, "exit": "TP"})
                position = None

        elif position == "short":
            if high >= sl:
                equity += -risk_amount
                trades.append({"pnl": -risk_amount, "exit": "SL"})
                position = None
            elif low <= tp:
                equity += risk_amount * 2.0
                trades.append({"pnl": risk_amount * 2.0, "exit": "TP"})
                position = None

        if signal and position is None:
            position = signal.signal.value
            sl = signal.stop_loss
            tp = signal.take_profit
            entry_equity = equity

    total_return = (equity - initial_equity) / initial_equity * 100
    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"\nBacktest: {symbol} 5m | Data: Bybit")
    print("-" * 50)
    print(f"Period:         {len(df)} candles (~{len(df) * 5 / 60 / 24:.0f} days)")
    print(f"Initial equity: ${initial_equity:,.0f}")
    print(f"Final equity:   ${equity:,.0f}")
    print(f"Total return:   {total_return:+.1f}%")
    print(f"Trades:         {len(trades)}")
    print(f"Win rate:       {win_rate:.1f}%")
    print(f"Profit factor:  {profit_factor:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--symbol", default="XAUTUSDT")
    args = parser.parse_args()
    run_backtest(symbol=args.symbol, days=args.days)
