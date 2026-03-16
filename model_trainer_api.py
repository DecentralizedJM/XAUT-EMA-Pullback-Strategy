"""
RETRAIN — XAUT Institutional ML Model (API-based)
==================================================
Fetches ~2 years of 5m OHLCV from Bybit public API (no key required),
builds the full feature set, labels trades via forward simulation,
and saves a retrained Random Forest model with updated hyperparameters:
  - n_estimators: 800  (was 600)
  - max_depth:     8   (was 6)
  - min_samples_leaf: 25 (was 15)
  - ml_threshold:  0.50 (was 0.35)  ← tighter gate

Run: PYTHONPATH=/tmp/pyextra python3 model_trainer_api.py
Output: saved_model/xauusd_model.joblib + scaler.joblib + model_config.joblib
"""

import time, warnings, os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import ephem
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
os.makedirs("saved_model", exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL            = "XAUTUSDT"
FETCH_DAYS        = 730          # ~2 years from Bybit
RR                = 2.5
ATR_SL            = 1.5
MAX_BARS_FORWARD  = 250
FULL_MOON_THRESH  = 85.0
ML_THRESHOLD      = 0.50         # ← raised from 0.35
BYBIT_URL         = "https://api.bybit.com/v5/market/kline"

FEATURE_COLS = [
    's1L','s2L','s3L','s4L','s5L','s6','s7L','score_long',
    'ret_1b','ret_30m','ret_1h','ret_4h',
    'ema21_slope','ema50_slope','rsi','rsi_slope','rsi_ma','bb_pos',
    'atr_ratio','atr_pctile','vol_ratio','body_ratio',
    'h1_adx','h1_rsi','h1_adx_slope','h1_rsi_slope',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','lunar_sin',
]

# ── Indicator helpers ─────────────────────────────────────────────────────────
def ema_c(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi_c(s,p=14):
    d=s.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    return 100-100/(1+g.ewm(alpha=1/p,adjust=False).mean()
                       /l.ewm(alpha=1/p,adjust=False).mean().replace(0,1e-9))
def atr_c(h,l,c,p=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(alpha=1/p,adjust=False).mean()
def adx_c(h,l,c,p=14):
    up=h.diff(); dn=-l.diff()
    pdm=up.where((up>dn)&(up>0),0.); ndm=dn.where((dn>up)&(dn>0),0.)
    at=atr_c(h,l,c,p)
    pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(),pdi,ndi
def bolb(s,p=20,k=2):
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    return mid+k*std, mid-k*std

# ── Fetch Bybit (paginated) ───────────────────────────────────────────────────
def fetch_bybit(interval: str, days: int) -> pd.DataFrame:
    end_ms   = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rows = []; cur_end = end_ms; calls = 0
    while cur_end > start_ms:
        params = dict(category="linear", symbol=SYMBOL,
                      interval=interval, limit=1000, end=cur_end)
        r = requests.get(BYBIT_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit: {data.get('retMsg')}")
        rows = data["result"]["list"]
        if not rows: break
        all_rows.extend(rows)
        earliest = int(rows[-1][0])
        if earliest <= start_ms: break
        cur_end = earliest - 1
        calls += 1
        if calls % 5 == 0: time.sleep(0.3)
    df = pd.DataFrame(all_rows,
                      columns=["ts","open","high","low","close","volume","turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - timedelta(days=days)
    return df[df["ts"] >= cutoff]

# ── Fetch data ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  XAUT ML Model Retrainer — {FETCH_DAYS}-Day Bybit Dataset")
print(f"{'='*60}\n")

print("Fetching 5m klines from Bybit …", end=" ", flush=True)
df = fetch_bybit("5", FETCH_DAYS)
print(f"{len(df):,} bars  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})")

print("Fetching 1H klines …", end=" ", flush=True)
h1_raw = fetch_bybit("60", FETCH_DAYS + 30)
print(f"{len(h1_raw):,} bars")

print("Fetching 1D klines …", end=" ", flush=True)
d1_raw = fetch_bybit("D", FETCH_DAYS + 250)
print(f"{len(d1_raw):,} bars\n")

# ── Lunar phase map ───────────────────────────────────────────────────────────
print("Computing lunar phases …", end=" ", flush=True)
start_d = df["ts"].iloc[0].date()
end_d   = df["ts"].iloc[-1].date() + timedelta(days=2)
dates   = pd.date_range(str(start_d), str(end_d), freq="D")
lunar_map = {}
for d in dates:
    m = ephem.Moon(); m.compute(d.strftime("%Y/%m/%d")); lunar_map[d.date()] = float(m.phase)
print("OK\n")

# ── Build 5m features ─────────────────────────────────────────────────────────
print("Computing indicators …")
df.set_index("ts", inplace=True)

df["ema8"]     = ema_c(df["close"],8)
df["ema21"]    = ema_c(df["close"],21)
df["ema50"]    = ema_c(df["close"],50)
df["rsi14"]    = rsi_c(df["close"],14)
df["atr14"]    = atr_c(df["high"],df["low"],df["close"],14)
df["atr_avg50"]= df["atr14"].rolling(50).mean()
df["atr_ratio"]= (df["atr14"]/df["atr_avg50"].replace(0,np.nan)).fillna(1.0)
df["atr_pctile"]= df["atr14"].rolling(252).rank(pct=True)

macd_l = ema_c(df["close"],12) - ema_c(df["close"],26)
df["macd_hist"] = macd_l - ema_c(macd_l,9)
df["vol_ma"]   = df["volume"].rolling(20).mean()
df["vol_ratio"]= df["volume"]/df["vol_ma"].replace(0,np.nan)
df["body_ratio"]=(df["close"]-df["open"]).abs()/(df["high"]-df["low"]).replace(0,np.nan)
df["tap_zone"] = df["ema21"]*0.0020

bb_up,bb_dn    = bolb(df["close"])
df["bb_pos"]   = (df["close"]-bb_dn)/(bb_up-bb_dn).replace(0,np.nan)

for n,lbl in [(1,'1b'),(6,'30m'),(12,'1h'),(48,'4h')]:
    df[f"ret_{lbl}"]=(df["close"]-df["close"].shift(n))/df["close"].shift(n)*100
df["ema21_slope"]=(df["ema21"]-df["ema21"].shift(6))/df["ema21"].shift(6)*100
df["ema50_slope"]=(df["ema50"]-df["ema50"].shift(12))/df["ema50"].shift(12)*100
df["rsi_slope"] = df["rsi14"]-df["rsi14"].shift(5)
df["rsi_ma"]    = df["rsi14"].rolling(14).mean()

df["hour"]  = df.index.hour
df["dow"]   = df.index.dayofweek
df["month"] = df.index.month
df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7)
df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12)

df["date"]             = df.index.date
df["lunar_pct"]        = df["date"].apply(lambda d: lunar_map.get(d,50.0))
df["lunar_sin"]        = np.sin(2*np.pi*df["lunar_pct"]/100)
df["full_moon_avoid"]  = df["lunar_pct"] > FULL_MOON_THRESH

# ── 1H indicators ─────────────────────────────────────────────────────────────
h1 = h1_raw.copy().set_index("ts")
h1["ema21"]  = ema_c(h1["close"],21)
h1["ema200"] = ema_c(h1["close"],200)
h1["rsi14"]  = rsi_c(h1["close"],14)
adxv,pdiv,ndiv = adx_c(h1["high"],h1["low"],h1["close"],14)
h1["adx"]=adxv; h1["pdi"]=pdiv; h1["ndi"]=ndiv
h1["adx_slope"]=h1["adx"]-h1["adx"].shift(3)
h1["rsi_slope"]=h1["rsi14"]-h1["rsi14"].shift(3)
h1_lag = h1[["ema21","ema200","rsi14","adx","pdi","ndi","adx_slope","rsi_slope"]].shift(1)
for col in h1_lag.columns:
    df[f"h1_{col}"] = h1_lag[col].reindex(df.index, method="ffill")

# ── Daily EMA200 ──────────────────────────────────────────────────────────────
d1 = d1_raw.copy().set_index("ts")
d1["d_ema200"] = ema_c(d1["close"],200)
df["d_ema200"] = d1["d_ema200"].shift(1).reindex(df.index, method="ffill")

# ── Filter flags ──────────────────────────────────────────────────────────────
d=df
df["s1L"]=((d["ema8"]>d["ema21"])&(d["ema21"]>d["ema50"])).astype(int)
df["s2L"]=((d["rsi14"]>50)&(d["rsi14"]<70)).astype(int)
df["s3L"]=(d["macd_hist"]>0).astype(int)
df["s4L"]=(d["h1_ema21"]>d["h1_ema200"]).astype(int)
df["s5L"]=((d["h1_adx"]>20)&(d["h1_pdi"]>d["h1_ndi"])).astype(int)
df["s6"] =(d["volume"]>d["vol_ma"]).astype(int)
df["s7L"]=(d["h1_rsi14"]>50).astype(int)
df["sL"] = df[["s1L","s2L","s3L","s4L","s5L","s6","s7L"]].sum(axis=1)

df["s1S"]=((d["ema8"]<d["ema21"])&(d["ema21"]<d["ema50"])).astype(int)
df["s2S"]=((d["rsi14"]>30)&(d["rsi14"]<50)).astype(int)
df["s3S"]=(d["macd_hist"]<0).astype(int)
df["s4S"]=(d["h1_ema21"]<d["h1_ema200"]).astype(int)
df["s5S"]=((d["h1_adx"]>20)&(d["h1_ndi"]>d["h1_pdi"])).astype(int)
df["s7S"]=(d["h1_rsi14"]<50).astype(int)
df["sS"] = df[["s1S","s2S","s3S","s4S","s5S","s6","s7S"]].sum(axis=1)

df["tap_L"]  =(d["close"]>d["ema21"])&(d["close"]<=d["ema21"]+d["tap_zone"])
df["tap_S"]  =(d["close"]<d["ema21"])&(d["close"]>=d["ema21"]-d["tap_zone"])
df["session"]=(d["dow"].isin([1,2,3]))&(d["hour"]>=8)&(d["hour"]<19)&(d["month"]!=6)
df["mac_L"]  = d["close"]>d["d_ema200"]
df["mac_S"]  = d["close"]<d["d_ema200"]
df["l_ok"]   = ~d["full_moon_avoid"]

df["cL"]=(df["tap_L"])&(df["sL"]>=5)&df["session"]&df["mac_L"]&df["l_ok"]
df["cS"]=(df["tap_S"])&(df["sS"]>=5)&df["session"]&df["mac_S"]&df["l_ok"]

df.reset_index(inplace=True)

# ── Forward label simulation ──────────────────────────────────────────────────
print("Labeling candidates via forward simulation …")
c_arr  = df["close"].values; h_arr  = df["high"].values
lo_arr = df["low"].values;   at_arr = df["atr14"].values
cL_arr = df["cL"].values;    cS_arr = df["cS"].values
sL_arr = df["sL"].values;    sS_arr = df["sS"].values

# feature col → df column mapping
COL_MAP = {"rsi":"rsi14","h1_rsi":"h1_rsi14","h1_adx":"h1_adx",
           "h1_adx_slope":"h1_adx_slope","h1_rsi_slope":"h1_rsi_slope"}

labeled = []
for i in range(300, len(df)-MAX_BARS_FORWARD):
    is_L=bool(cL_arr[i]); is_S=bool(cS_arr[i])
    if not (is_L or is_S): continue
    c=c_arr[i]; at=at_arr[i]
    if np.isnan(at) or at<=0: at=c*0.003
    dire = "long" if is_L else "short"
    score= int(sL_arr[i]) if is_L else int(sS_arr[i])
    sl   = c-ATR_SL*at if is_L else c+ATR_SL*at
    tp   = c+(c-sl)*RR if is_L else c-(sl-c)*RR
    label=None
    for j in range(i+1, min(i+MAX_BARS_FORWARD,len(df))):
        hj=h_arr[j]; lj=lo_arr[j]
        if dire=="long":
            if lj<=sl: label=0; break
            if hj>=tp: label=1; break
        else:
            if hj>=sl: label=0; break
            if lj<=tp: label=1; break
    if label is None: continue

    row={"label":label,"direction":dire,"score":score}
    for f in FEATURE_COLS:
        dcol = COL_MAP.get(f, f)
        try:   v=float(df[dcol].iloc[i])
        except KeyError: v=0.0
        if np.isnan(v): v=0.0
        if dire=="short":
            if f in ["ret_1b","ret_30m","ret_1h","ret_4h",
                     "ema21_slope","ema50_slope","rsi_slope","bb_pos"]:
                v=-v
            maps={"s1L":"s1S","s2L":"s2S","s3L":"s3S",
                  "s4L":"s4S","s5L":"s5S","s7L":"s7S"}
            if f in maps:
                try: v=float(df[maps[f]].iloc[i])
                except: v=0.0
            if f=="score_long":
                try: v=float(df["sS"].iloc[i])
                except: v=float(score)
        row[f]=v
    labeled.append(row)

ML_df = pd.DataFrame(labeled)
X = ML_df[FEATURE_COLS].astype(float).values
y = ML_df["label"].values
print(f"  Samples: {len(ML_df):,}  |  wins={y.sum():,}  losses={(y==0).sum():,}  "
      f"win_rate={y.mean()*100:.1f}%\n")

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training Random Forest (800 trees, depth=8, leaf=25) …")
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=800, max_depth=8, min_samples_leaf=25,
    max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1)
model.fit(X_sc, y)

# 5-fold CV for confidence
print("Running 5-fold cross-validation (ROC-AUC) …", end=" ", flush=True)
cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,  "saved_model/xauusd_model.joblib")
joblib.dump(scaler, "saved_model/scaler.joblib")
joblib.dump({
    "feature_cols":         FEATURE_COLS,
    "rr":                   RR,
    "atr_sl":               ATR_SL,
    "full_moon_threshold":  FULL_MOON_THRESH,
    "ml_threshold":         ML_THRESHOLD,
    "trained_on":           f"{df['ts'].iloc[0].date()} to {df['ts'].iloc[-1].date()} (Bybit API)",
    "n_samples":            len(ML_df),
    "cv_auc_mean":          round(float(cv_scores.mean()), 4),
}, "saved_model/model_config.joblib")

print(f"\nModel saved to saved_model/")
print(f"  xauusd_model.joblib  ({model.n_estimators} trees)")
print(f"  scaler.joblib        ({len(FEATURE_COLS)} features)")
print(f"  ml_threshold         : {ML_THRESHOLD}")
print(f"  cv_auc               : {cv_scores.mean():.3f}")
print(f"\nNext: run PYTHONPATH=/tmp/pyextra python3 backtest_3m.py")
