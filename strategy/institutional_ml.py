"""
Institutional ML Strategy for Mudrex.
Combines 7 technical filters + Lunar Cycle + Rolling ML Prediction.
"""

import pandas as pd
import numpy as np
import ephem
import joblib
import os
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

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
    probability: float
    score: int
    risk_pct: float

def ema_c(s, p): return s.ewm(span=p, adjust=False).mean()
def rsi_c(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, adjust=False).mean()
    al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, 1e-9))
def atr_c(h, l, c, p=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(1)
    return tr.ewm(alpha=1/p, adjust=False).mean()
def adx_c(h, l, c, p=14):
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0.0); ndm = dn.where((dn>up)&(dn>0), 0.0)
    at = atr_c(h,l,c,p)
    pdi = 100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi = 100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(), pdi, ndi
def bolb(s, p=20, k=2):
    mid = s.rolling(p).mean(); std = s.rolling(p).std()
    return mid + k*std, mid - k*std, mid

class InstitutionalMLStrategy:
    def __init__(self, model_dir: str = "saved_model"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.cfg = None
        self._load_model()
        
    def _load_model(self):
        try:
            self.model = joblib.load(os.path.join(self.model_dir, "xauusd_model.joblib"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
            self.cfg = joblib.load(os.path.join(self.model_dir, "model_config.joblib"))
            logger.info("Institutional ML model loaded successfully")
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)
            raise

    def get_lunar_phase(self, dt):
        m = ephem.Moon()
        m.compute(dt.strftime("%Y/%m/%d"))
        return m.phase

    def evaluate(self, df5m: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df5m) < 200:
            return None
        
        df = df5m.copy()
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        df['hour'] = df.index.hour
        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # ── Indicators
        df['ema8'] = ema_c(df['close'], 8)
        df['ema21'] = ema_c(df['close'], 21)
        df['ema50'] = ema_c(df['close'], 50)
        df['rsi'] = rsi_c(df['close'], 14)
        df['atr'] = atr_c(df['high'], df['low'], df['close'], 14)
        ml_v = ema_c(df['close'], 12) - ema_c(df['close'], 26)
        df['macd_hist'] = ml_v - ema_c(ml_v, 9)
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['atr_avg50'] = df['atr'].rolling(50).mean()
        
        # ── Institutional Features
        for n, label in [(1,'1b'),(6,'30m'),(12,'1h'),(48,'4h')]:
            df[f'ret_{label}'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n) * 100
        df['ema21_slope'] = (df['ema21'] - df['ema21'].shift(6)) / df['ema21'].shift(6) * 100
        df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(12)) / df['ema50'].shift(12) * 100
        df['rsi_slope'] = df['rsi'] - df['rsi'].shift(5)
        df['rsi_ma'] = df['rsi'].rolling(14).mean()
        bb_up, bb_dn, _ = bolb(df['close'], 20, 2)
        df['bb_pos'] = (df['close'] - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)
        df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
        df['atr_pctile'] = df['atr'].rolling(252).rank(pct=True)
        df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, np.nan)
        df['atr_ratio'] = (df['atr'] / df['atr_avg50'].replace(0, np.nan)).fillna(1.0)
        
        # ── Time Features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        
        # ── Lunar Feature
        last_dt = df.index[-1]
        lunar_pct = self.get_lunar_phase(last_dt)
        df['lunar_sin'] = np.sin(2 * np.pi * lunar_pct / 100)
        
        # ── 1H Indicators
        h1 = df[['open','high','low','close','volume']].resample('1h').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        h1['ema21'] = ema_c(h1['close'], 21)
        h1['ema200'] = ema_c(h1['close'], 200)
        h1['rsi'] = rsi_c(h1['close'], 14)
        adx_v, pdi, ndi = adx_c(h1['high'], h1['low'], h1['close'], 14)
        h1['adx'] = adx_v; h1['pdi'] = pdi; h1['ndi'] = ndi
        h1['adx_slope'] = h1['adx'] - h1['adx'].shift(3)
        h1['rsi_slope'] = h1['rsi'] - h1['rsi'].shift(3)
        h1_s = h1[['ema21','ema200','rsi','adx','pdi','ndi','adx_slope','rsi_slope']].shift(1)
        for col in h1_s.columns:
            df[f'h1_{col}'] = h1_s[col].reindex(df.index, method='ffill')
            
        # ── Daily EMA200
        d1 = df[['open','high','low','close','volume']].resample('1d').agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        d1['ema200'] = ema_c(d1['close'], 200)
        df['d1_ema200'] = d1[['ema200']].shift(1)['ema200'].reindex(df.index, method='ffill')
        
        d = df.iloc[-1]
        
        # ── Technical Filters
        s1L = int((d['ema8'] > d['ema21']) & (d['ema21'] > d['ema50']))
        s2L = int((d['rsi'] > 50) & (d['rsi'] < 70))
        s3L = int(d['macd_hist'] > 0)
        s4L = int(d['h1_ema21'] > d['h1_ema200'])
        s5L = int((d['h1_adx'] > 20) & (d['h1_pdi'] > d['h1_ndi']))
        s6 = int(d['volume'] > d['vol_ma'])
        s7L = int(d['h1_rsi'] > 50)
        score_long = s1L + s2L + s3L + s4L + s5L + s6 + s7L
        
        s1S = int((d['ema8'] < d['ema21']) & (d['ema21'] < d['ema50']))
        s2S = int((d['rsi'] > 30) & (d['rsi'] < 50))
        s3S = int(d['macd_hist'] < 0)
        s4S = int(d['h1_ema21'] < d['h1_ema200'])
        s5S = int((d['h1_adx'] > 20) & (d['h1_ndi'] > d['h1_pdi']))
        s7S = int(d['h1_rsi'] < 50)
        score_short = s1S + s2S + s3S + s4S + s5S + s6 + s7S
        
        tap_zone = d['ema21'] * 0.20 / 100
        tap_long = (d['close'] > d['ema21']) & (d['close'] <= d['ema21'] + tap_zone)
        tap_short = (d['close'] < d['ema21']) & (d['close'] >= d['ema21'] - tap_zone)
        
        valid_session = (int(d['dow']) in [1,2,3]) & (8 <= int(d['hour']) < 19) & (int(d['month']) != 6)
        macro_long = d['close'] > d['d1_ema200']
        macro_short = d['close'] < d['d1_ema200']
        full_moon_avoid = lunar_pct > self.cfg['full_moon_threshold']
        
        can_long = tap_long and score_long >= 5 and valid_session and macro_long and not full_moon_avoid
        can_short = tap_short and score_short >= 5 and valid_session and macro_short and not full_moon_avoid
        
        direction = None
        if can_long:
            direction = Signal.LONG
            score = score_long
        elif can_short:
            direction = Signal.SHORT
            score = score_short
        else:
            return None
            
        # ── ML Filter
        feat_map = {
            's1L': s1L if direction == Signal.LONG else s1S,
            's2L': s2L if direction == Signal.LONG else s2S,
            's3L': s3L if direction == Signal.LONG else s3S,
            's4L': s4L if direction == Signal.LONG else s4S,
            's5L': s5L if direction == Signal.LONG else s5S,
            's6': s6,
            's7L': s7L if direction == Signal.LONG else s7S,
            'score_long': score,
            'ret_1b': float(d.get('ret_1b', 0)),
            'ret_30m': float(d.get('ret_30m', 0)),
            'ret_1h': float(d.get('ret_1h', 0)),
            'ret_4h': float(d.get('ret_4h', 0)),
            'ema21_slope': float(d.get('ema21_slope', 0)),
            'ema50_slope': float(d.get('ema50_slope', 0)),
            'rsi': float(d['rsi']),
            'rsi_slope': float(d.get('rsi_slope', 0)),
            'rsi_ma': float(d.get('rsi_ma', 50)),
            'bb_pos': float(d.get('bb_pos', 0.5)),
            'atr_ratio': float(d.get('atr_ratio', 1)),
            'atr_pctile': float(d.get('atr_pctile', 0.5)),
            'vol_ratio': float(d.get('vol_ratio', 1)),
            'body_ratio': float(d.get('body_ratio', 0.5)),
            'h1_adx': float(d['h1_adx']),
            'h1_rsi': float(d['h1_rsi']),
            'h1_adx_slope': float(d.get('h1_adx_slope', 0)),
            'h1_rsi_slope': float(d.get('h1_rsi_slope', 0)),
            'hour_sin': float(d.get('hour_sin', 0)),
            'hour_cos': float(d.get('hour_cos', 1)),
            'dow_sin': float(d.get('dow_sin', 0)),
            'dow_cos': float(d.get('dow_cos', 1)),
            'month_sin': float(d.get('month_sin', 0)),
            'lunar_sin': float(d.get('lunar_sin', 0))
        }
        
        if direction == Signal.SHORT:
            for f in ['ret_1b','ret_30m','ret_1h','ret_4h','ema21_slope','ema50_slope','rsi_slope','bb_pos']:
                feat_map[f] = -feat_map[f]
                
        X = np.array([[feat_map[f] for f in self.cfg['feature_cols']]])
        X_scaled = self.scaler.transform(X)
        ml_prob = self.model.predict_proba(X_scaled)[0, 1]
        
        if ml_prob < self.cfg['ml_threshold']:
            logger.info("Candidate %s rejected by ML: prob=%.3f < %.3f", direction, ml_prob, self.cfg['ml_threshold'])
            return None
            
        # ── Position Sizing (F#3)
        base_risks = {5: 0.008, 6: 0.012, 7: 0.015}
        base_r = base_risks.get(score, 0.008)
        at = d['atr']
        atr_avg = float(d.get('atr_avg50', at))
        vol_ratio = np.clip(atr_avg/at if at > 0 else 1.0, 0.4, 2.0)
        risk_pct = base_r * vol_ratio * 100  # Convert to percentage
        
        # ── SL/TP
        if direction == Signal.LONG:
            sl = d['close'] - self.cfg['atr_sl'] * at
            tp = d['close'] + (d['close'] - sl) * self.cfg['rr']
        else:
            sl = d['close'] + self.cfg['atr_sl'] * at
            tp = d['close'] - (sl - d['close']) * self.cfg['rr']
            
        return TradeSignal(
            signal=direction,
            entry_price=round(float(d['close']), 2),
            stop_loss=round(float(sl), 2),
            take_profit=round(float(tp), 2),
            probability=float(ml_prob),
            score=score,
            risk_pct=float(risk_pct)
        )
