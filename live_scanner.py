"""
LIVE SCANNER — XAU/USD INSTITUTIONAL STRATEGY
==============================================
Fetches live 5m gold data from Yahoo Finance every 5 minutes.
Applies all filters: F#1 + F#3 + F#8 + F#11 (ML).
Outputs clear signals with entry, SL, TP levels.

SETUP:
  1. Run model_trainer.py ONCE first
  2. pip3 install yfinance
  3. python3 live_scanner.py

MODES:
  A) Continuous auto-scan (every 5 min): python3 live_scanner.py
  B) One-shot check right now:            python3 live_scanner.py --once
  C) Paper trade log:                     python3 live_scanner.py --paper
"""

import pandas as pd, numpy as np, time, sys, os, json, ephem, warnings
import argparse
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    os.system('pip3 install yfinance -q')
    import yfinance as yf

try:
    import joblib
except ImportError:
    os.system('pip3 install joblib -q')
    import joblib

# ─── Load saved model ─────────────────────────────────────────────────────────
MODEL_DIR = 'saved_model'
if not os.path.exists(f'{MODEL_DIR}/xauusd_model.joblib'):
    print("❌ Model not found. Run 'python3 model_trainer.py' first.")
    sys.exit(1)

model  = joblib.load(f'{MODEL_DIR}/xauusd_model.joblib')
scaler = joblib.load(f'{MODEL_DIR}/scaler.joblib')
cfg    = joblib.load(f'{MODEL_DIR}/model_config.joblib')

FEATURE_COLS      = cfg['feature_cols']
RR                = cfg['rr']           # 2.5
ATR_SL            = cfg['atr_sl']       # 1.5
ML_THRESH         = cfg['ml_threshold'] # 0.35
FULL_MOON_THRESH  = cfg['full_moon_threshold']  # 85.0

# ─── Indicator helpers (identical to backtest) ─────────────────────────────────
def ema_c(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi_c(s,p=14):
    d=s.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/p,adjust=False).mean(); al=l.ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,1e-9))
def atr_c(h,l,c,p=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(alpha=1/p,adjust=False).mean()
def adx_c(h,l,c,p=14):
    up=h.diff(); dn=-l.diff()
    pdm=up.where((up>dn)&(up>0),0.0); ndm=dn.where((dn>up)&(dn>0),0.0)
    at=atr_c(h,l,c,p)
    pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(),pdi,ndi
def bolb(s,p=20,k=2):
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    return mid+k*std,mid-k*std,mid

def get_lunar(dt_utc):
    m=ephem.Moon()
    m.compute(dt_utc.strftime('%Y/%m/%d'))
    return m.phase

def get_live_data(ticker='GC=F', bars=600):
    """Download live 5m OHLCV data from Yahoo Finance. GC=F = Gold Futures."""
    try:
        raw=yf.download(ticker, period='5d', interval='5m', progress=False, auto_adjust=True)
        if raw.empty or len(raw)<200:
            # Fallback to spot gold
            raw=yf.download('XAUUSD=X', period='5d', interval='5m', progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns=raw.columns.get_level_values(0)
        raw=raw.rename(columns=str.lower)
        raw=raw[['open','high','low','close','volume']].dropna()
        raw.index=pd.to_datetime(raw.index)
        if raw.index.tz is None: raw.index=raw.index.tz_localize('UTC')
        else: raw.index=raw.index.tz_convert('UTC')
        return raw.tail(bars)
    except Exception as e:
        print(f"  ⚠️  Data fetch error: {e}"); return None

def compute_features(df5m):
    """Compute all 32 indicators from raw 5m OHLCV data."""
    df=df5m.copy()
    df['hour']=df.index.hour; df['dow']=df.index.dayofweek
    df['month']=df.index.month; df['date']=df.index.normalize()

    # Core
    df['ema8']=ema_c(df['close'],8); df['ema21']=ema_c(df['close'],21)
    df['ema50']=ema_c(df['close'],50)
    df['rsi']=rsi_c(df['close'],14); df['atr']=atr_c(df['high'],df['low'],df['close'],14)
    ml_v=ema_c(df['close'],12)-ema_c(df['close'],26); df['macd_hist']=ml_v-ema_c(ml_v,9)
    df['vol_ma']=df['volume'].rolling(20).mean()
    df['atr_avg50']=df['atr'].rolling(50).mean()
    df['atr_ratio']=(df['atr']/df['atr_avg50'].replace(0,np.nan)).fillna(1.0)
    df['tap_zone']=df['ema21']*0.20/100

    # Momentum
    for n,label in [(1,'1b'),(6,'30m'),(12,'1h'),(48,'4h')]:
        df[f'ret_{label}']=(df['close']-df['close'].shift(n))/df['close'].shift(n)*100
    df['ema21_slope']=(df['ema21']-df['ema21'].shift(6))/df['ema21'].shift(6)*100
    df['ema50_slope']=(df['ema50']-df['ema50'].shift(12))/df['ema50'].shift(12)*100
    df['rsi_slope']=df['rsi']-df['rsi'].shift(5);  df['rsi_ma']=df['rsi'].rolling(14).mean()
    bb_up,bb_dn,_=bolb(df['close'],20,2)
    df['bb_pos']=(df['close']-bb_dn)/(bb_up-bb_dn).replace(0,np.nan)
    df['vol_ratio']=df['volume']/df['vol_ma'].replace(0,np.nan)
    df['atr_pctile']=df['atr'].rolling(252).rank(pct=True)
    df['body_ratio']=(df['close']-df['open']).abs()/(df['high']-df['low']).replace(0,np.nan)

    # Cyclical time
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']=np.sin(2*np.pi*df['dow']/7);    df['dow_cos']=np.cos(2*np.pi*df['dow']/7)
    df['month_sin']=np.sin(2*np.pi*df['month']/12)

    # Lunar
    today=df.index[-1].date()
    lunar_pct=get_lunar(df.index[-1])
    df['lunar_prev']=lunar_pct; df['lunar_sin']=np.sin(2*np.pi*lunar_pct/100)

    # 1H indicators (resample → shift back → ffill)
    h1=df[['open','high','low','close','volume']].resample('1H').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    h1['ema21']=ema_c(h1['close'],21); h1['ema200']=ema_c(h1['close'],200)
    h1['rsi']=rsi_c(h1['close'],14)
    adx_v,pdi,ndi=adx_c(h1['high'],h1['low'],h1['close'],14)
    h1['adx']=adx_v; h1['pdi']=pdi; h1['ndi']=ndi
    h1['adx_slope']=h1['adx']-h1['adx'].shift(3); h1['rsi_slope']=h1['rsi']-h1['rsi'].shift(3)
    h1_s=h1[['ema21','ema200','rsi','adx','pdi','ndi','adx_slope','rsi_slope']].shift(1)
    for col in h1_s.columns: df[f'h1_{col}']=h1_s[col].reindex(df.index,method='ffill')

    # Daily EMA200
    d1=df[['open','high','low','close','volume']].resample('1D').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    d1['ema200']=ema_c(d1['close'],200)
    df['d1_ema200']=d1[['ema200']].shift(1)['ema200'].reindex(df.index,method='ffill')

    return df

def evaluate_bar(df):
    """Evaluate the LAST bar in df and return signal dict."""
    d=df.iloc[-1]
    c=d['close']; at=d['atr'] if not np.isnan(d['atr']) and d['atr']>0 else c*0.003

    # ── Signals
    s1L=int((d['ema8']>d['ema21'])&(d['ema21']>d['ema50']))
    s2L=int((d['rsi']>50)&(d['rsi']<70))
    s3L=int(d['macd_hist']>0)
    s4L=int(d['h1_ema21']>d['h1_ema200'])
    s5L=int((d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi']))
    s6 =int(d['volume']>d['vol_ma'])
    s7L=int(d['h1_rsi']>50)
    score_long=s1L+s2L+s3L+s4L+s5L+s6+s7L

    s1S=int((d['ema8']<d['ema21'])&(d['ema21']<d['ema50']))
    s2S=int((d['rsi']>30)&(d['rsi']<50))
    s3S=int(d['macd_hist']<0)
    s4S=int(d['h1_ema21']<d['h1_ema200'])
    s5S=int((d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi']))
    s7S=int(d['h1_rsi']<50)
    score_short=s1S+s2S+s3S+s4S+s5S+s6+s7S

    # ── Gate checks
    tap_zone=d['ema21']*0.20/100
    tap_long=(d['close']>d['ema21'])&(d['close']<=d['ema21']+tap_zone)
    tap_short=(d['close']<d['ema21'])&(d['close']>=d['ema21']-tap_zone)
    hour=int(d['hour']); dow=int(d['dow']); month=int(d['month'])
    valid_session=(dow in [1,2,3])&(8<=hour<19)&(month!=6)
    macro_long=d['close']>d['d1_ema200']
    macro_short=d['close']<d['d1_ema200']
    lunar_pct=d['lunar_prev']
    full_moon_avoid=lunar_pct>FULL_MOON_THRESH
    lunar_ok=not full_moon_avoid

    # ── Determine candidate
    can_long =tap_long  & (score_long>=5)  & valid_session & macro_long  & lunar_ok
    can_short=tap_short & (score_short>=5) & valid_session & macro_short & lunar_ok

    result={
        'timestamp': df.index[-1],
        'price': round(c, 2),
        'signal': 'NONE',
        'score': 0,
        'ml_prob': 0.0,
        'sl': None, 'tp': None,
        'risk_pct': None,
        'atr': round(at, 3),
        'ema21': round(d['ema21'], 2),
        'rsi': round(d['rsi'], 1),
        'h1_adx': round(d['h1_adx'], 1),
        'lunar_pct': round(lunar_pct, 1),
        'session_ok': valid_session,
        'macro_bias': 'BULL' if macro_long else 'BEAR',
        'full_moon_avoid': full_moon_avoid,
        'filters_pass': False,
    }

    direction=None
    if can_long:   direction='LONG';  score=score_long
    elif can_short: direction='SHORT'; score=score_short
    else: return result

    # Build feature vector
    feat_map={'s1L':s1L if direction=='LONG' else s1S,
              's2L':s2L if direction=='LONG' else s2S,
              's3L':s3L if direction=='LONG' else s3S,
              's4L':s4L if direction=='LONG' else s4S,
              's5L':s5L if direction=='LONG' else s5S,
              's6':s6, 's7L':s7L if direction=='LONG' else s7S,
              'score_long':score,
              'ret_1b':   float(d.get('ret_1b',0)),
              'ret_30m':  float(d.get('ret_30m',0)),
              'ret_1h':   float(d.get('ret_1h',0)),
              'ret_4h':   float(d.get('ret_4h',0)),
              'ema21_slope':float(d.get('ema21_slope',0)),
              'ema50_slope':float(d.get('ema50_slope',0)),
              'rsi':float(d['rsi']), 'rsi_slope':float(d.get('rsi_slope',0)),
              'rsi_ma':float(d.get('rsi_ma',50)), 'bb_pos':float(d.get('bb_pos',0.5)),
              'atr_ratio':float(d.get('atr_ratio',1)), 'atr_pctile':float(d.get('atr_pctile',0.5)),
              'vol_ratio':float(d.get('vol_ratio',1)), 'body_ratio':float(d.get('body_ratio',0.5)),
              'h1_adx':float(d['h1_adx']),'h1_rsi':float(d['h1_rsi']),
              'h1_adx_slope':float(d.get('h1_adx_slope',0)),'h1_rsi_slope':float(d.get('h1_rsi_slope',0)),
              'hour_sin':float(d.get('hour_sin',0)),'hour_cos':float(d.get('hour_cos',1)),
              'dow_sin':float(d.get('dow_sin',0)),'dow_cos':float(d.get('dow_cos',1)),
              'month_sin':float(d.get('month_sin',0)),'lunar_sin':float(d.get('lunar_sin',0))}

    if direction=='SHORT':
        for f in ['ret_1b','ret_30m','ret_1h','ret_4h','ema21_slope','ema50_slope','rsi_slope','bb_pos']:
            feat_map[f]=-feat_map[f]

    X=np.array([[feat_map[f] for f in FEATURE_COLS]])
    X_scaled=scaler.transform(X)
    ml_prob=model.predict_proba(X_scaled)[0,1]

    # Sizing (F#3)
    BASE_RISK={5:0.008,6:0.012,7:0.015}
    base_r=BASE_RISK.get(score,0.008)
    atr_avg=float(d.get('atr_avg50',at)) if not np.isnan(d.get('atr_avg50',at)) else at
    vol_ratio=np.clip(atr_avg/at if at>0 else 1.0, 0.4, 2.0)
    risk_pct=base_r*vol_ratio

    if direction=='LONG': sl=c-ATR_SL*at; tp=c+(c-sl)*RR
    else:                 sl=c+ATR_SL*at; tp=c-(sl-c)*RR

    result.update({
        'signal': direction if ml_prob>=ML_THRESH else 'WEAK',
        'score': score,
        'ml_prob': round(float(ml_prob),4),
        'sl': round(sl,2), 'tp': round(tp,2),
        'risk_pct': round(risk_pct*100,2),
        'filters_pass': ml_prob>=ML_THRESH,
    })
    return result

def print_signal(sig, paper_mode=False):
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    direction=sig['signal']
    is_trade=(direction in ['LONG','SHORT']) and sig['filters_pass']

    print("\n" + "═"*60)
    print(f"  XAU/USD SCANNER  |  {now}")
    print("═"*60)
    print(f"  Price   : ${sig['price']:.2f}")
    print(f"  EMA21   : ${sig['ema21']:.2f}")
    print(f"  ATR     : {sig['atr']:.2f}")
    print(f"  RSI 5m  : {sig['rsi']:.1f}")
    print(f"  1H ADX  : {sig['h1_adx']:.1f}")
    print(f"  Lunar   : {sig['lunar_pct']:.1f}% illumination {'🌕 AVOID' if sig['full_moon_avoid'] else '🌙 OK'}")
    print(f"  Session : {'✅ Active' if sig['session_ok'] else '❌ Off hours (Tue-Thu 08-19 UTC)'}")
    print(f"  Macro   : {sig['macro_bias']} {'📈' if sig['macro_bias']=='BULL' else '📉'}")
    print()

    if is_trade:
        icon='🟢' if direction=='LONG' else '🔴'
        print(f"  {icon} ═══════════════════════════════════════")
        print(f"  {icon}  SIGNAL: {direction}  (Score {sig['score']}/7 | ML={sig['ml_prob']:.3f})")
        print(f"  {icon} ═══════════════════════════════════════")
        print(f"  {'│'} Entry  : ${sig['price']:.2f}")
        print(f"  {'│'} SL     : ${sig['sl']:.2f}  ({abs(sig['price']-sig['sl']):.2f} pts away)")
        print(f"  {'│'} TP     : ${sig['tp']:.2f}  ({abs(sig['price']-sig['tp']):.2f} pts away)")
        print(f"  {'│'} R:R    : 1:{RR}")
        print(f"  {'│'} Risk   : {sig['risk_pct']:.2f}% of account")
        print(f"  {'│'} ATR    : {sig['atr']:.3f}  (SL at {ATR_SL}×ATR)")
        if paper_mode:
            log_paper_trade(sig)
    elif direction=='WEAK':
        print(f"  ⚠️  CANDIDATE: {direction} (Score {sig['score']}/7) but ML={sig['ml_prob']:.3f} < {ML_THRESH}")
        print(f"     (needs ML >= {ML_THRESH}, currently below threshold)")
    else:
        print(f"  ⚫  NO SIGNAL  — conditions not met")
        print(f"     (need: EMA tap + score≥5 + session + macro + lunar)")

    print("═"*60)
    return is_trade

def log_paper_trade(sig, log_file='paper_trades.json'):
    trades=[]
    if os.path.exists(log_file):
        with open(log_file) as f:
            try: trades=json.load(f)
            except: trades=[]
    trades.append({
        'time':     sig['timestamp'].strftime('%Y-%m-%d %H:%M UTC'),
        'signal':   sig['signal'],
        'entry':    sig['price'],
        'sl':       sig['sl'],
        'tp':       sig['tp'],
        'score':    sig['score'],
        'ml_prob':  sig['ml_prob'],
        'risk_pct': sig['risk_pct'],
        'status':   'OPEN',
    })
    with open(log_file,'w') as f: json.dump(trades,f,indent=2)
    print(f"  📝 Logged to {log_file}")

def show_paper_log(log_file='paper_trades.json'):
    if not os.path.exists(log_file): print("No paper trades yet."); return
    with open(log_file) as f: trades=json.load(f)
    print(f"\n{'─'*70}")
    print(f"  PAPER TRADE LOG — {len(trades)} trades")
    print(f"{'─'*70}")
    for i,t in enumerate(trades[-10:],1):
        status_icon={'OPEN':'🟡','WIN':'✅','LOSS':'❌'}.get(t['status'],'⚪')
        print(f"  {i:2}. {t['time']}  {t['signal']:<6} @ ${t['entry']:.2f}  "
              f"SL=${t['sl']:.2f}  TP=${t['tp']:.2f}  ML={t['ml_prob']:.3f}  {status_icon}{t['status']}")
    print(f"{'─'*70}\n")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='XAU/USD Live Scanner')
    parser.add_argument('--once',  action='store_true', help='Run once and exit')
    parser.add_argument('--paper', action='store_true', help='Log signals to paper_trades.json')
    parser.add_argument('--log',   action='store_true', help='Show paper trade log')
    parser.add_argument('--ticker',default='GC=F',      help='Yahoo Finance ticker (default: GC=F)')
    args=parser.parse_args()

    if args.log:
        show_paper_log(); sys.exit(0)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
  XAU/USD INSTITUTIONAL SCANNER — LIVE DEMO MODE
  Model trained on: {cfg['trained_on']}
  Samples: {cfg['n_samples']:,} | Filters: F#1 + F#3 + F#8 + F#11 ML
  ML Threshold: {ML_THRESH} | Data: {args.ticker} (Yahoo Finance)
  Expected: ~7-8 signals/week (Tue-Thu 08-19 UTC)
╚══════════════════════════════════════════════════════════════╝
""")

    scan_count=0
    while True:
        scan_count+=1
        print(f"\n[Scan #{scan_count}] Fetching {args.ticker} 5m data...")
        data=get_live_data(args.ticker)

        if data is None or len(data)<200:
            print("  ⚠️  Insufficient data, retrying in 60s...")
            if args.once: sys.exit(1)
            time.sleep(60); continue

        print(f"  Downloaded {len(data)} bars. Latest: {data.index[-1]} @ ${data['close'].iloc[-1]:.2f}")

        try:
            df=compute_features(data)
            sig=evaluate_bar(df)
            is_trade=print_signal(sig, paper_mode=args.paper)
        except Exception as e:
            print(f"  ❌ Error computing signal: {e}")
            import traceback; traceback.print_exc()

        if args.once:
            sys.exit(0)

        # Wait until next 5m bar
        now=datetime.now(timezone.utc)
        secs_to_next=300-(now.second + now.minute%5*60)
        if secs_to_next<=0: secs_to_next+=300
        print(f"\n  ⏱️  Next scan in {secs_to_next:.0f}s  ({now.strftime('%H:%M:%S')} UTC)")
        time.sleep(secs_to_next)
