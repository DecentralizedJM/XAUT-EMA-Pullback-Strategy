"""
STEP 1: TRAIN & SAVE THE INSTITUTIONAL ML MODEL
================================================
Run this ONCE to train on all available historical data and save the model.
Then the live_scanner.py loads this saved model for real-time signals.

Run: python3 model_trainer.py
Output: saved_model/xauusd_model.joblib + saved_model/scaler.joblib
"""

import pandas as pd, numpy as np, os
import ephem, warnings, joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs('saved_model', exist_ok=True)

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
    return mid+k*std, mid-k*std, mid

RR=2.5; ATR_SL=1.5; MAX_BARS_FORWARD=250
FULL_MOON_THRESHOLD=85.0

FEATURE_COLS = [
    's1L','s2L','s3L','s4L','s5L','s6','s7L','score_long',
    'ret_1b','ret_30m','ret_1h','ret_4h',
    'ema21_slope','ema50_slope','rsi','rsi_slope','rsi_ma','bb_pos',
    'atr_ratio','atr_pctile','vol_ratio','body_ratio',
    'h1_adx','h1_rsi','h1_adx_slope','h1_rsi_slope',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','lunar_sin',
]

print("Computing lunar phases...")
def get_lunar(d_str):
    m=ephem.Moon(); m.compute(d_str); return m.phase
dates=pd.date_range('2021-01-01','2026-04-01',freq='D')
lunar_df=pd.DataFrame({'date':dates})
lunar_df['lunar_pct']=lunar_df['date'].apply(lambda d: get_lunar(d.strftime('%Y/%m/%d')))
lunar_df['full_moon_avoid']=(lunar_df['lunar_pct']>FULL_MOON_THRESHOLD).shift(1).fillna(False)
lunar_df['lunar_prev']=lunar_df['lunar_pct'].shift(1).fillna(50)
lunar_df.set_index('date',inplace=True)

print("Loading historical data...")
df=pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt']=pd.to_datetime(df['timestamp'],unit='ms')
df['hour']=df['dt'].dt.hour; df['dow']=df['dt'].dt.dayofweek
df['month']=df['dt'].dt.month; df['date']=df['dt'].dt.normalize()
df['year']=df['dt'].dt.year

df['ema8']=ema_c(df['close'],8); df['ema21']=ema_c(df['close'],21)
df['ema50']=ema_c(df['close'],50)
df['rsi']=rsi_c(df['close'],14); df['atr']=atr_c(df['high'],df['low'],df['close'],14)
ml_v=ema_c(df['close'],12)-ema_c(df['close'],26)
sl_m=ema_c(ml_v,9); df['macd_hist']=ml_v-sl_m
df['vol_ma']=df['volume'].rolling(20).mean()
df['atr_avg50']=df['atr'].rolling(50).mean()
df['atr_ratio']=(df['atr']/df['atr_avg50'].replace(0,np.nan)).fillna(1.0)
df['tap_zone']=df['ema21']*0.20/100

for n,label in [(1,'1b'),(6,'30m'),(12,'1h'),(48,'4h')]:
    df[f'ret_{label}']=(df['close']-df['close'].shift(n))/df['close'].shift(n)*100
df['ema21_slope']=(df['ema21']-df['ema21'].shift(6))/df['ema21'].shift(6)*100
df['ema50_slope']=(df['ema50']-df['ema50'].shift(12))/df['ema50'].shift(12)*100
df['rsi_slope']=df['rsi']-df['rsi'].shift(5)
df['rsi_ma']=df['rsi'].rolling(14).mean()
bb_up,bb_dn,_=bolb(df['close'],20,2)
df['bb_pos']=(df['close']-bb_dn)/(bb_up-bb_dn).replace(0,np.nan)
df['vol_ratio']=df['volume']/df['vol_ma'].replace(0,np.nan)
df['atr_pctile']=df['atr'].rolling(252).rank(pct=True)
df['body_ratio']=(df['close']-df['open']).abs()/(df['high']-df['low']).replace(0,np.nan)
df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
df['dow_sin']=np.sin(2*np.pi*df['dow']/7);    df['dow_cos']=np.cos(2*np.pi*df['dow']/7)
df['month_sin']=np.sin(2*np.pi*df['month']/12)

df=df.merge(lunar_df[['full_moon_avoid','lunar_prev']],left_on='date',right_index=True,how='left')
df['full_moon_avoid']=df['full_moon_avoid'].fillna(False)
df['lunar_sin']=np.sin(2*np.pi*df['lunar_prev']/100)

df.set_index('dt',inplace=True)
h1=df[['open','high','low','close','volume']].resample('1H').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
h1['ema21']=ema_c(h1['close'],21); h1['ema200']=ema_c(h1['close'],200)
h1['rsi']=rsi_c(h1['close'],14)
adx_v,pdi,ndi=adx_c(h1['high'],h1['low'],h1['close'],14)
h1['adx']=adx_v; h1['pdi']=pdi; h1['ndi']=ndi
h1['adx_slope']=h1['adx']-h1['adx'].shift(3); h1['rsi_slope']=h1['rsi']-h1['rsi'].shift(3)
h1_s=h1[['ema21','ema200','rsi','adx','pdi','ndi','adx_slope','rsi_slope']].shift(1)
for col in h1_s.columns: df[f'h1_{col}']=h1_s[col].reindex(df.index,method='ffill')

d1=df[['open','high','low','close','volume']].resample('1D').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
d1['ema200']=ema_c(d1['close'],200)
df['d1_ema200']=d1[['ema200']].shift(1)['ema200'].reindex(df.index,method='ffill')
df.reset_index(inplace=True)

d=df
df['s1L']=((d['ema8']>d['ema21'])&(d['ema21']>d['ema50'])).astype(int)
df['s2L']=((d['rsi']>50)&(d['rsi']<70)).astype(int)
df['s3L']=(d['macd_hist']>0).astype(int)
df['s4L']=(d['h1_ema21']>d['h1_ema200']).astype(int)
df['s5L']=((d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi'])).astype(int)
df['s6']=(d['volume']>d['vol_ma']).astype(int)
df['s7L']=(d['h1_rsi']>50).astype(int)
df['score_long']=df[['s1L','s2L','s3L','s4L','s5L','s6','s7L']].sum(axis=1)
df['s1S']=((d['ema8']<d['ema21'])&(d['ema21']<d['ema50'])).astype(int)
df['s2S']=((d['rsi']>30)&(d['rsi']<50)).astype(int)
df['s3S']=(d['macd_hist']<0).astype(int)
df['s4S']=(d['h1_ema21']<d['h1_ema200']).astype(int)
df['s5S']=((d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi'])).astype(int)
df['s7S']=(d['h1_rsi']<50).astype(int)
df['score_short']=df[['s1S','s2S','s3S','s4S','s5S','s6','s7S']].sum(axis=1)
df['tap_long']=(d['close']>d['ema21'])&(d['close']<=d['ema21']+df['tap_zone'])
df['tap_short']=(d['close']<d['ema21'])&(d['close']>=d['ema21']-df['tap_zone'])
df['valid_session']=(d['dow'].isin([1,2,3]))&(d['hour']>=8)&(d['hour']<19)&(d['month']!=6)
df['macro_long']=d['close']>d['d1_ema200']
df['macro_short']=d['close']<d['d1_ema200']
df['lunar_ok']=~d['full_moon_avoid']
df['cand_long']=(df['tap_long']&(df['score_long']>=5)&df['valid_session']&df['macro_long']&df['lunar_ok'])
df['cand_short']=(df['tap_short']&(df['score_short']>=5)&df['valid_session']&df['macro_short']&df['lunar_ok'])

print("Labeling ALL 5 years of candidates via forward simulation...")
c_arr=df['close'].values; h_arr=df['high'].values; lo_arr=df['low'].values; at_arr=df['atr'].values
cL=df['cand_long'].values; cS=df['cand_short'].values
sL=df['score_long'].values; sS=df['score_short'].values

labeled_rows=[]
for i in range(300, len(df)-MAX_BARS_FORWARD):
    is_long=cL[i]; is_short=cS[i]
    if not (is_long or is_short): continue
    c=c_arr[i]; at=at_arr[i]
    if np.isnan(at) or at<=0: at=c*0.003
    direction='long' if is_long else 'short'
    score=int(sL[i]) if is_long else int(sS[i])
    if direction=='long': sl_p=c-ATR_SL*at; tp_p=c+(c-sl_p)*RR
    else:                 sl_p=c+ATR_SL*at; tp_p=c-(sl_p-c)*RR
    label=None
    for j in range(i+1, min(i+MAX_BARS_FORWARD, len(df))):
        hj=h_arr[j]; lj=lo_arr[j]
        if direction=='long':
            if lj<=sl_p: label=0; break
            if hj>=tp_p: label=1; break
        else:
            if hj>=sl_p: label=0; break
            if lj<=tp_p: label=1; break
    if label is None: continue
    row={'label':label,'direction':direction,'score':score,'entry_idx':i,'ts':df['timestamp'].iloc[i]}
    for f in FEATURE_COLS:
        v=df[f].iloc[i]
        if pd.isna(v): v=0.0
        if direction=='short' and f in ['s1L','s2L','s3L','s4L','s5L','s7L','score_long',
                                          'ret_1b','ret_30m','ret_1h','ret_4h',
                                          'ema21_slope','ema50_slope','rsi_slope','bb_pos']:
            map_={'s1L':'s1S','s2L':'s2S','s3L':'s3S','s4L':'s4S','s5L':'s5S','s7L':'s7S'}
            v=df[map_[f]].iloc[i] if f in map_ else (df['score_short'].iloc[i] if f=='score_long' else -v)
        row[f]=v
    labeled_rows.append(row)

ML_df=pd.DataFrame(labeled_rows)
X=ML_df[FEATURE_COLS].astype(float).values
y=ML_df['label'].values
print(f"  Total training samples: {len(ML_df):,}  (wins={y.sum()}, losses={(y==0).sum()})")

# Train on ALL data (full 5 years)
print("Training final model on ALL 5 years...")
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

model=RandomForestClassifier(
    n_estimators=600, max_depth=6, min_samples_leaf=15,
    max_features='sqrt', class_weight='balanced',
    random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# Save
joblib.dump(model,  'saved_model/xauusd_model.joblib')
joblib.dump(scaler, 'saved_model/scaler.joblib')
joblib.dump({'feature_cols':FEATURE_COLS,
             'rr':RR,'atr_sl':ATR_SL,
             'full_moon_threshold':FULL_MOON_THRESHOLD,
             'ml_threshold':0.35,
             'trained_on':'2021-03-01 to 2026-03-15',
             'n_samples':len(ML_df)},
            'saved_model/model_config.joblib')

print(f"\n✅ Model saved to saved_model/")
print(f"   xauusd_model.joblib  ({model.n_estimators} trees)")
print(f"   scaler.joblib        (StandardScaler, {len(FEATURE_COLS)} features)")
print(f"   model_config.joblib  (params & metadata)")
print(f"\nNext: run python3 live_scanner.py")
