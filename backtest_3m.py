"""
3-Month Backtest — XAUT Institutional 11-Filter Strategy (Full ML Gate)
========================================================================
Fetches 92 days of 5m + 1H + 1D klines from Bybit (no API key needed).
Applies all 10 technical filters + ML Random Forest gate (filter #11).

Run: python3 backtest_3m.py
"""

import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import ephem
import joblib
import os

warnings.filterwarnings("ignore")

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL           = "XAUTUSDT"
DAYS             = 92
RR               = 2.5
ATR_SL           = 1.5
SCORE_MIN        = 5
ACCOUNT_USD      = 10_000
BASE_RISK_PCT    = {5: 0.8, 6: 1.2, 7: 1.5}   # % of account per score tier
MAX_BARS_FWD     = 250
FULL_MOON_THRESH = 85.0
ML_THRESHOLD     = 0.35
BYBIT_URL        = "https://api.bybit.com/v5/market/kline"
MODEL_DIR        = "saved_model"

FEATURE_COLS = [
    's1L','s2L','s3L','s4L','s5L','s6','s7L','score_long',
    'ret_1b','ret_30m','ret_1h','ret_4h',
    'ema21_slope','ema50_slope','rsi','rsi_slope','rsi_ma','bb_pos',
    'atr_ratio','atr_pctile','vol_ratio','body_ratio',
    'h1_adx','h1_rsi','h1_adx_slope','h1_rsi_slope',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','lunar_sin',
]

print(f"\n{'='*64}")
print(f"  XAUT Institutional 11-Filter Backtest — Last {DAYS} Days (Full ML)")
print(f"{'='*64}\n")

# ── Load ML model ─────────────────────────────────────────────────────────────
print("Loading ML model …", end=" ", flush=True)
model  = joblib.load(os.path.join(MODEL_DIR, "xauusd_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
cfg    = joblib.load(os.path.join(MODEL_DIR, "model_config.joblib"))
ML_THRESHOLD = cfg.get("ml_threshold", ML_THRESHOLD)
print(f"OK  ({model.n_estimators} trees, threshold={ML_THRESHOLD})")

# ── Lunar phase precompute ────────────────────────────────────────────────────
print("Computing lunar phases …", end=" ", flush=True)
start_date = datetime.utcnow() - timedelta(days=DAYS + 5)
date_range = pd.date_range(start_date.strftime("%Y-%m-%d"),
                           (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d"), freq="D")
lunar_map = {}
for d in date_range:
    m = ephem.Moon()
    m.compute(d.strftime("%Y/%m/%d"))
    lunar_map[d.date()] = float(m.phase)
print("OK")

def get_lunar(dt_date):
    return lunar_map.get(dt_date, 50.0)

# ── Fetch Bybit klines ────────────────────────────────────────────────────────
def fetch_bybit(interval: str, days: int) -> pd.DataFrame:
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []
    cur_end  = end_ms
    calls    = 0
    while cur_end > start_ms:
        params = dict(category="linear", symbol=SYMBOL,
                      interval=interval, limit=1000, end=cur_end)
        r = requests.get(BYBIT_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
        rows = data["result"]["list"]
        if not rows:
            break
        all_rows.extend(rows)
        earliest = int(rows[-1][0])
        if earliest <= start_ms:
            break
        cur_end = earliest - 1
        calls += 1
        if calls % 5 == 0:
            time.sleep(0.3)
    if not all_rows:
        raise ValueError("No data from Bybit.")
    df = pd.DataFrame(all_rows,
                      columns=["ts","open","high","low","close","volume","turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - timedelta(days=days)
    return df[df["ts"] >= cutoff]

# ── Indicator helpers ─────────────────────────────────────────────────────────
def ema(s, p):   return s.ewm(span=p, adjust=False).mean()
def rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); lo = (-d).clip(lower=0)
    return 100 - 100 / (1 + g.ewm(alpha=1/p,adjust=False).mean()
                              / lo.ewm(alpha=1/p,adjust=False).mean().replace(0,1e-9))
def atr(h, l, c, p=14):
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(alpha=1/p,adjust=False).mean()
def adx(h, l, c, p=14):
    up=h.diff(); dn=-l.diff()
    pdm=up.where((up>dn)&(up>0),0.); ndm=dn.where((dn>up)&(dn>0),0.)
    at=atr(h,l,c,p)
    pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(), pdi, ndi
def bolb(s, p=20, k=2):
    mid = s.rolling(p).mean(); std = s.rolling(p).std()
    return mid+k*std, mid-k*std

# ── Fetch data ────────────────────────────────────────────────────────────────
print("Fetching 5m klines …", end=" ", flush=True)
m5 = fetch_bybit("5", DAYS + 10)
print(f"{len(m5):,} bars  ({m5['ts'].iloc[0].date()} → {m5['ts'].iloc[-1].date()})")

print("Fetching 1H klines …", end=" ", flush=True)
h1_raw = fetch_bybit("60", DAYS + 40)
print(f"{len(h1_raw):,} bars")

print("Fetching 1D klines …", end=" ", flush=True)
d1_raw = fetch_bybit("D", 450)
print(f"{len(d1_raw):,} bars\n")

# ── Build 5m indicators ───────────────────────────────────────────────────────
df = m5.copy()
df.set_index("ts", inplace=True)

df["ema8"]  = ema(df["close"], 8)
df["ema21"] = ema(df["close"], 21)
df["ema50"] = ema(df["close"], 50)
df["rsi14"] = rsi(df["close"], 14)
df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
df["atr_avg50"] = df["atr14"].rolling(50).mean()
df["atr_ratio"] = (df["atr14"] / df["atr_avg50"].replace(0,np.nan)).fillna(1.0)
df["atr_pctile"] = df["atr14"].rolling(252).rank(pct=True)
macd_line = ema(df["close"],12) - ema(df["close"],26)
df["macd_hist"] = macd_line - ema(macd_line, 9)
df["vol_ma"]  = df["volume"].rolling(20).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0,np.nan)
df["body_ratio"] = (df["close"]-df["open"]).abs() / (df["high"]-df["low"]).replace(0,np.nan)
df["tap_zone"] = df["ema21"] * 0.0020

bb_up, bb_dn = bolb(df["close"],20,2)
df["bb_pos"] = (df["close"] - bb_dn) / (bb_up - bb_dn).replace(0,np.nan)

# Returns
for n, label in [(1,'1b'),(6,'30m'),(12,'1h'),(48,'4h')]:
    df[f'ret_{label}'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n) * 100
df['ema21_slope'] = (df['ema21'] - df['ema21'].shift(6)) / df['ema21'].shift(6) * 100
df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(12)) / df['ema50'].shift(12) * 100
df['rsi_slope']  = df['rsi14'] - df['rsi14'].shift(5)
df['rsi_ma']     = df['rsi14'].rolling(14).mean()

# Time features
df["hour"]  = df.index.hour
df["dow"]   = df.index.dayofweek
df["month"] = df.index.month
df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
df['dow_sin']   = np.sin(2*np.pi*df['dow']/7)
df['dow_cos']   = np.cos(2*np.pi*df['dow']/7)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)

# Lunar
df['date'] = df.index.date
df['lunar_pct']       = df['date'].apply(get_lunar)
df['lunar_sin']       = np.sin(2*np.pi*df['lunar_pct']/100)
df['full_moon_avoid'] = df['lunar_pct'] > FULL_MOON_THRESH

# ── 1H indicators ─────────────────────────────────────────────────────────────
h1 = h1_raw.copy()
h1.set_index("ts", inplace=True)
h1["ema21"]  = ema(h1["close"], 21)
h1["ema200"] = ema(h1["close"], 200)
h1["rsi14"]  = rsi(h1["close"], 14)
h1_adx, h1_pdi, h1_ndi = adx(h1["high"], h1["low"], h1["close"], 14)
h1["adx"] = h1_adx; h1["pdi"] = h1_pdi; h1["ndi"] = h1_ndi
h1["adx_slope"] = h1["adx"] - h1["adx"].shift(3)
h1["rsi_slope"] = h1["rsi14"] - h1["rsi14"].shift(3)

h1_lag = h1[["ema21","ema200","rsi14","adx","pdi","ndi","adx_slope","rsi_slope"]].shift(1)
for col in h1_lag.columns:
    df[f"h1_{col}"] = h1_lag[col].reindex(df.index, method="ffill")

# ── Daily EMA200 ──────────────────────────────────────────────────────────────
d1 = d1_raw.copy()
d1.set_index("ts", inplace=True)
d1["d_ema200"] = ema(d1["close"], 200)
df["d_ema200"] = d1["d_ema200"].shift(1).reindex(df.index, method="ffill")

# ── Filter to backtest window ─────────────────────────────────────────────────
bt_start = pd.Timestamp.utcnow().replace(tzinfo=None) - timedelta(days=DAYS)
df = df[df.index >= bt_start].copy()
print(f"Backtest window : {df.index[0].date()} → {df.index[-1].date()}  ({len(df):,} bars)\n")

# ── Technical filters ─────────────────────────────────────────────────────────
d = df
df["s1L"] = ((d["ema8"]>d["ema21"])&(d["ema21"]>d["ema50"])).astype(int)
df["s2L"] = ((d["rsi14"]>50)&(d["rsi14"]<70)).astype(int)
df["s3L"] = (d["macd_hist"]>0).astype(int)
df["s4L"] = (d["h1_ema21"]>d["h1_ema200"]).astype(int)
df["s5L"] = ((d["h1_adx"]>20)&(d["h1_pdi"]>d["h1_ndi"])).astype(int)
df["s6"]  = (d["volume"]>d["vol_ma"]).astype(int)
df["s7L"] = (d["h1_rsi14"]>50).astype(int)
df["scoreL"] = df[["s1L","s2L","s3L","s4L","s5L","s6","s7L"]].sum(axis=1)

df["s1S"] = ((d["ema8"]<d["ema21"])&(d["ema21"]<d["ema50"])).astype(int)
df["s2S"] = ((d["rsi14"]>30)&(d["rsi14"]<50)).astype(int)
df["s3S"] = (d["macd_hist"]<0).astype(int)
df["s4S"] = (d["h1_ema21"]<d["h1_ema200"]).astype(int)
df["s5S"] = ((d["h1_adx"]>20)&(d["h1_ndi"]>d["h1_pdi"])).astype(int)
df["s7S"] = (d["h1_rsi14"]<50).astype(int)
df["scoreS"] = df[["s1S","s2S","s3S","s4S","s5S","s6","s7S"]].sum(axis=1)

df["tap_long"]  = (d["close"]>d["ema21"]) & (d["close"]<=d["ema21"]+d["tap_zone"])
df["tap_short"] = (d["close"]<d["ema21"]) & (d["close"]>=d["ema21"]-d["tap_zone"])
df["session"]   = (d["dow"].isin([1,2,3]))&(d["hour"]>=8)&(d["hour"]<19)&(d["month"]!=6)
df["macro_L"]   = d["close"] > d["d_ema200"]
df["macro_S"]   = d["close"] < d["d_ema200"]
df["lunar_ok"]  = ~d["full_moon_avoid"]

df["cand_L"] = df["tap_long"]  & (df["scoreL"]>=SCORE_MIN) & df["session"] & df["macro_L"] & df["lunar_ok"]
df["cand_S"] = df["tap_short"] & (df["scoreS"]>=SCORE_MIN) & df["session"] & df["macro_S"] & df["lunar_ok"]

# ── Simulation arrays ─────────────────────────────────────────────────────────
close_arr = df["close"].values
high_arr  = df["high"].values
low_arr   = df["low"].values
atr_arr   = df["atr14"].values
ts_arr    = df.index.to_numpy()
cL_arr    = df["cand_L"].values
cS_arr    = df["cand_S"].values
sL_arr    = df["scoreL"].values
sS_arr    = df["scoreS"].values

# Pre-extract feature arrays for speed
feat_arr = {}
for f in FEATURE_COLS:
    col = f
    # rsi column is named rsi14, h1_rsi is h1_rsi14
    if f == 'rsi':         col = 'rsi14'
    elif f == 'h1_rsi':    col = 'h1_rsi14'
    elif f == 'h1_adx_slope': col = 'h1_adx_slope'
    elif f == 'h1_rsi_slope': col = 'h1_rsi_slope'
    try:
        feat_arr[f] = df[col].values
    except KeyError:
        feat_arr[f] = np.zeros(len(df))

s1S_arr=df["s1S"].values; s2S_arr=df["s2S"].values
s3S_arr=df["s3S"].values; s4S_arr=df["s4S"].values
s5S_arr=df["s5S"].values; s7S_arr=df["s7S"].values

# ── Forward simulation ────────────────────────────────────────────────────────
trades     = []
in_trade   = False
trade_end_i = -1
ml_filtered = 0

print("Running simulation with ML gate …")
for i in range(200, len(df) - MAX_BARS_FWD):
    if in_trade and i < trade_end_i:
        continue
    in_trade = False

    is_long  = bool(cL_arr[i])
    is_short = bool(cS_arr[i])
    if not (is_long or is_short):
        continue

    direction = "LONG" if is_long else "SHORT"
    score     = int(sL_arr[i]) if is_long else int(sS_arr[i])
    c  = close_arr[i]
    at = atr_arr[i]
    if np.isnan(at) or at <= 0:
        at = c * 0.003

    # Build feature vector
    row = []
    for f in FEATURE_COLS:
        v = float(feat_arr[f][i]) if not np.isnan(feat_arr[f][i]) else 0.0
        if direction == "SHORT" and f in ['ret_1b','ret_30m','ret_1h','ret_4h',
                                           'ema21_slope','ema50_slope','rsi_slope','bb_pos']:
            v = -v
        if f == 'score_long':
            v = float(score)
        if f in ['s1L','s2L','s3L','s4L','s5L','s7L'] and direction == "SHORT":
            map_ = {'s1L':'s1S','s2L':'s2S','s3L':'s3S','s4L':'s4S','s5L':'s5S','s7L':'s7S'}
            v = float(feat_arr.get(map_[f], feat_arr[f])[i])
            if np.isnan(v): v = 0.0
        row.append(v)

    X = np.array([row])
    X_scaled = scaler.transform(X)
    ml_prob  = float(model.predict_proba(X_scaled)[0, 1])

    if ml_prob < ML_THRESHOLD:
        ml_filtered += 1
        continue

    # SL / TP
    if direction == "LONG":
        sl = c - ATR_SL * at
        tp = c + (c - sl) * RR
    else:
        sl = c + ATR_SL * at
        tp = c - (sl - c)  * RR

    # Position size (inverse vol)
    base_r   = BASE_RISK_PCT.get(score, 0.8) / 100
    atr_ma50 = np.nanmean(atr_arr[max(0,i-50):i])
    vol_adj  = float(np.clip(atr_ma50/at if at>0 else 1.0, 0.4, 2.0))
    risk_pct = base_r * vol_adj
    risk_usd = ACCOUNT_USD * risk_pct
    sl_dist  = abs(c - sl)
    qty      = risk_usd / sl_dist if sl_dist > 0 else 0

    outcome  = None
    exit_bar = None
    exit_px  = None
    for j in range(i+1, min(i+MAX_BARS_FWD, len(df))):
        h = high_arr[j]; lo = low_arr[j]
        if direction == "LONG":
            if lo <= sl:  outcome="LOSS"; exit_px=sl;  exit_bar=j; break
            if h  >= tp:  outcome="WIN";  exit_px=tp;  exit_bar=j; break
        else:
            if h  >= sl:  outcome="LOSS"; exit_px=sl;  exit_bar=j; break
            if lo <= tp:  outcome="WIN";  exit_px=tp;  exit_bar=j; break

    if outcome is None:
        outcome  = "TIMEOUT"
        exit_bar = i + MAX_BARS_FWD
        exit_px  = close_arr[min(exit_bar, len(df)-1)]

    pnl_pts = (exit_px-c) if direction=="LONG" else (c-exit_px)
    pnl_usd = pnl_pts * qty

    trades.append({
        "entry_time": ts_arr[i],
        "exit_time":  ts_arr[min(exit_bar, len(ts_arr)-1)],
        "dir":        direction,
        "score":      score,
        "ml_prob":    round(ml_prob, 3),
        "entry":      round(c,2),
        "sl":         round(sl,2),
        "tp":         round(tp,2),
        "exit_px":    round(exit_px,2),
        "outcome":    outcome,
        "pnl_usd":    round(pnl_usd,2),
    })

    in_trade    = True
    trade_end_i = exit_bar

# ── Results ──────────────────────────────────────────────────────────────────
if not trades:
    print("No trades passed all 11 filters in this period.")
    sys.exit(0)

tdf = pd.DataFrame(trades)
tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
tdf["exit_time"]  = pd.to_datetime(tdf["exit_time"])

wins     = tdf[tdf["outcome"]=="WIN"]
losses   = tdf[tdf["outcome"]=="LOSS"]
timeouts = tdf[tdf["outcome"]=="TIMEOUT"]
total    = len(tdf)
n_wins   = len(wins)
n_loss   = len(losses)
n_time   = len(timeouts)
win_rate = n_wins / total * 100

gross_pnl  = tdf["pnl_usd"].sum()
cumulative = tdf["pnl_usd"].cumsum()
peak       = cumulative.cummax()
max_dd     = (cumulative - peak).min()
rtn_pct    = gross_pnl / ACCOUNT_USD * 100

tdf["date"] = tdf["exit_time"].dt.date
daily_pnl   = tdf.groupby("date")["pnl_usd"].sum()
sharpe      = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std()>0 else 0.0
weeks       = DAYS / 7
trades_pw   = total / weeks

avg_win  = wins["pnl_usd"].mean() if len(wins) else 0
avg_loss = losses["pnl_usd"].mean() if len(losses) else 0
profit_f = abs(avg_win / avg_loss) if avg_loss!=0 else float("inf")

raw_cands = int(cL_arr.sum()) + int(cS_arr.sum())
print(f"\n{'─'*64}")
print(f"  RESULTS  {df.index[0].date()} → {df.index[-1].date()}")
print(f"{'─'*64}")
print(f"  Raw candidates (10 filters) : {raw_cands}")
print(f"  Filtered by ML gate         : {ml_filtered}  ({ml_filtered/(raw_cands or 1)*100:.1f}%)")
print(f"  Trades taken                : {total}")
print(f"  Wins / Losses               : {n_wins} / {n_loss}  ({n_time} timed-out)")
print(f"  Win Rate                    : {win_rate:.1f}%")
print(f"  Trades / Week               : {trades_pw:.1f}")
print(f"{'─'*64}")
print(f"  Net P&L (USD)               : ${gross_pnl:+,.2f}")
print(f"  Return on Account           : {rtn_pct:+.2f}%")
print(f"  Max Drawdown                : ${max_dd:,.2f}  ({max_dd/ACCOUNT_USD*100:.2f}%)")
print(f"  Sharpe Ratio                : {sharpe:.2f}")
print(f"  Profit Factor               : {profit_f:.2f}")
print(f"  Avg Win / Avg Loss          : ${avg_win:+,.2f} / ${avg_loss:+,.2f}")
print(f"{'─'*64}\n")

print(f"  Trade Log ({total} trades passing all 11 filters)")
print(f"  {'Date':<12} {'Dir':<6} {'Sc':>2} {'ML%':>5} {'Entry':>8} {'Exit':>8} {'P&L':>9}  {'Result'}")
print(f"  {'─'*66}")
for _, r in tdf.iterrows():
    marker = "+" if r["outcome"]=="WIN" else ("-" if r["outcome"]=="LOSS" else "~")
    print(f"  {str(r['entry_time'].date()):<12} {r['dir']:<6} {r['score']:>2}"
          f" {r['ml_prob']*100:>4.0f}%"
          f" {r['entry']:>8.2f} {r['exit_px']:>8.2f} ${r['pnl_usd']:>+8.2f}  {marker} {r['outcome']}")

print(f"\n{'='*64}\n")
