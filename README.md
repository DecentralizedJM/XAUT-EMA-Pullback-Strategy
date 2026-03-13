# XAUT EMA Pullback Strategy

Production-ready EMA Pullback strategy for **XAUTUSDT** futures. Deployable to Railway.

- **Execution**: [Mudrex Futures API](https://docs.trade.mudrex.com/docs/overview) (where trades are placed)
- **Prices**: Bybit klines (Mudrex uses Bybit as broker, so we follow Bybit prices)

---

## System Architecture

```mermaid
flowchart TB
    subgraph External [External Services]
        Bybit[Bybit API]
        Mudrex[Mudrex API]
    end

    subgraph Bot [XAUT EMA Pullback Bot]
        DataLayer[Data Layer]
        StrategyLayer[Strategy Layer]
        ExchangeLayer[Exchange Layer]
    end

    subgraph DataLayer
        BybitKlines[Bybit Klines]
    end

    subgraph StrategyLayer
        EMAPullback[EMA Pullback]
    end

    subgraph ExchangeLayer
        MudrexClient[Mudrex Client]
    end

    Bybit -->|"OHLCV 5m"| BybitKlines
    BybitKlines -->|"DataFrame"| EMAPullback
    EMAPullback -->|"Trade Signal"| MudrexClient
    MudrexClient -->|"Orders"| Mudrex
```

---

## Data Flow

```mermaid
flowchart LR
    subgraph Step1 [1. Fetch]
        A[Bybit REST] -->|"GET /v5/market/kline"| B[OHLCV]
    end

    subgraph Step2 [2. Process]
        B --> C[Compute EMA]
        C --> D[Compute RSI]
        D --> E[Check Tap Zone]
    end

    subgraph Step3 [3. Decide]
        E --> F{Signal?}
        F -->|"Yes"| G[Entry + SL + TP]
        F -->|"No"| H[Wait]
    end

    subgraph Step4 [4. Execute]
        G --> I[Mudrex API]
        I --> I
    end
```

---

## Strategy Logic Flow

```mermaid
flowchart TD
    Start[New Bar] --> FetchOHLCV[Fetch Bybit Klines]
    FetchOHLCV --> ComputeIndicators[Compute EMA21, RSI14, MACD]
    ComputeIndicators --> CheckTrend[Trend Check]

    CheckTrend --> LongCondition{Price above EMA?}
    LongCondition -->|Yes| LongTap{In tap zone?}
    LongCondition -->|No| ShortCondition{Price below EMA?}

    LongTap -->|Yes| LongRSI{RSI > 50?}
    LongTap -->|No| NoSignal[No Signal]

    LongRSI -->|Yes| LongFirstTap{First tap only?}
    LongRSI -->|No| NoSignal

    LongFirstTap -->|OK| LongSignal[LONG Signal]
    LongFirstTap -->|Skip| NoSignal

    ShortCondition -->|Yes| ShortTap{In tap zone?}
    ShortCondition -->|No| NoSignal

    ShortTap -->|Yes| ShortRSI{RSI < 50?}
    ShortTap -->|No| NoSignal

    ShortRSI -->|Yes| ShortFirstTap{First tap only?}
    ShortRSI -->|No| NoSignal

    ShortFirstTap -->|OK| ShortSignal[SHORT Signal]
    ShortFirstTap -->|Skip| NoSignal

    LongSignal --> PlaceOrder[Place Order via Mudrex]
    ShortSignal --> PlaceOrder
```

---

## Bot Main Loop

```mermaid
flowchart TD
    Init[Start Bot] --> SetLeverage[Set Leverage on Mudrex]
    SetLeverage --> LoopStart[Loop Start]

    LoopStart --> GetBalance[Get Futures Balance]
    GetBalance --> FetchKlines[Fetch Bybit Klines]
    FetchKlines --> GetPositions[Get Open Positions]

    GetPositions --> Evaluate[Evaluate Strategy]
    Evaluate --> HasSignal{Signal?}

    HasSignal -->|No| Sleep[Sleep 60s]
    Sleep --> LoopStart

    HasSignal -->|Yes| RoundQty[Round Quantity]
    RoundQty --> PlaceOrder[Place Market Order]
    PlaceOrder -->|"SL + TP attached"| Sleep
```

---

## Component Architecture

```mermaid
flowchart TB
    subgraph Bot [bot.py]
        MainLoop[Main Loop]
        RoundQty[round_quantity]
        GetPosition[get_current_position]
    end

    subgraph Strategy [strategy/ema_pullback.py]
        EMAPullback[EMAPullbackStrategy]
        Evaluate[evaluate]
        Signal[TradeSignal]
    end

    subgraph Data [data/bybit_klines.py]
        FetchKlines[fetch_klines]
        FetchDataframe[fetch_klines_dataframe]
        FetchHistorical[fetch_historical_bybit]
    end

    subgraph Exchange [exchange/mudrex_client.py]
        MudrexClient[MudrexClient]
        RateLimiter[RateLimiter]
        PlaceOrder[place_market_order]
    end

    MainLoop --> FetchDataframe
    MainLoop --> GetPosition
    MainLoop --> PlaceOrder
    MainLoop --> Evaluate

    Evaluate --> EMAPullback
    EMAPullback --> Signal

    FetchDataframe --> FetchKlines
    FetchHistorical --> FetchKlines

    PlaceOrder --> MudrexClient
    MudrexClient --> RateLimiter
```

---

## File Structure

```
XAUT-EMA-Pullback-Strategy/
├── bot.py                 # Main trading loop (production)
├── backtest.py            # Backtest on Bybit historical data
├── config.py              # Strategy + Mudrex configuration
├── requirements.txt
├── .env.example           # MUDREX_API_SECRET
│
├── strategy/
│   └── ema_pullback.py    # EMA pullback logic, RSI filter
│
├── data/
│   └── bybit_klines.py    # Bybit kline fetcher (REST)
│
├── exchange/
│   └── mudrex_client.py   # Mudrex API client (rate-limited)
│
├── Dockerfile             # Railway deployment
├── Procfile
├── railway.toml
└── .dockerignore
```

---

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ema_period` | 21 | EMA length for trend |
| `tap_threshold_pct` | 0.2 | Tap zone size (% around EMA) |
| `stop_loss_buffer_pct` | 0.5 | SL distance beyond EMA |
| `take_profit_rr` | 2.0 | Risk-reward ratio |
| `risk_per_trade_pct` | 1.0 | Risk per trade (%) |
| `use_rsi_filter` | true | RSI > 50 long, RSI < 50 short |
| `first_tap_only` | true | Only first pullback after trend flip |

---

## Setup

```bash
git clone https://github.com/DecentralizedJM/XAUT-EMA-Pullback-Strategy.git
cd XAUT-EMA-Pullback-Strategy
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add MUDREX_API_SECRET
```

---

## Usage

```bash
# Backtest (Bybit data, no API keys)
python backtest.py --days 365

# Paper trading (no real orders)
python bot.py --paper

# Live trading (Mudrex)
python bot.py
```

---

## Deploy to Railway

1. Push to GitHub: [DecentralizedJM/XAUT-EMA-Pullback-Strategy](https://github.com/DecentralizedJM/XAUT-EMA-Pullback-Strategy)
2. Create a Railway project → Deploy from GitHub
3. Add `MUDREX_API_SECRET` in Railway dashboard
4. Deploy – Railway runs `python bot.py` 24/7

---

## API References

- [Mudrex Futures API](https://docs.trade.mudrex.com/docs/overview)
- [Bybit Kline API](https://bybit-exchange.github.io/docs/v5/market/kline)

---

## Disclaimer

Trading futures involves risk. Backtest and paper trade before using real capital.
