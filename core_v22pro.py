# -*- coding: utf-8 -*-
# Futures Pro V22 Pro+ (Balanced Trend) — SINGLE FILE APP
# Core engine (PA/OF/AI/Backtest/Summary) + Multi-TF + REST API + Web UI
# ---------------------------------------------------------------

from __future__ import annotations
import os, math, time, sys, logging, warnings, json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# ============ ML & Joblib ============
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load as joblib_load

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import feedparser  # type: ignore
    from textblob import TextBlob  # type: ignore
except Exception:
    feedparser = None
    TextBlob = None

warnings.filterwarnings("ignore")
log = logging.getLogger("V22-Pro+")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ================== تنظیمات ==================
DEFAULT_TF = "15m"
HTF_TF     = "1h"
USE_HTF    = True

AI_ENABLED = True
CALIBRATE_PROBS = True
CV_FOLDS   = 4
CV_GAP     = 10

DELTA_CLIP_ATR_MULT = 2.5
USE_BANDS  = True

W_ML_IND = 0.4
W_PA     = 0.3
W_OF     = 0.3
CONFIDENCE_MIN = 54.0

ACCOUNT_EQUITY = 10_000.0
RISK_PER_TRADE = 0.005
PORTFOLIO_RISK_CAP = 0.25
COST_BPS = 2.0
SLIPPAGE_BPS = 1.0

AI_TRAIN_WINDOW = 300
MODEL_DIR = os.path.join(os.getcwd(), "models_v22p")
os.makedirs(MODEL_DIR, exist_ok=True)

EXCHANGE_NAME = os.environ.get("EXCHANGE", "okx")
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
TESTNET = os.environ.get("TESTNET", "false").lower() == "true"

USE_LIMIT_ENTRY   = True
POST_ONLY         = True
PRICE_PROTECT_BPS = 4.0

SOFT_FILTERS = True
HORIZON      = 4

SYMBOLS = ["BTC-USD", "ETH-USD"]

FUND_TTL_SEC = 60*30
FUND_FEEDS = {
    "BTC": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
        "https://cointelegraph.com/rss/tag/bitcoin"
    ],
    "ETH": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US",
        "https://cointelegraph.com/rss/tag/ethereum"
    ]
}
_FUND_CACHE: Dict[str, Tuple[float, str, Optional[float]]] = {}

# ---- Summary Bull% Weights (no fundamentals inside calc) ----
SUMMARY_W_AI    = 0.35
SUMMARY_W_PA    = 0.20
SUMMARY_W_OF    = 0.20
SUMMARY_W_TREND = 0.15
SUMMARY_W_HTF   = 0.05
SUMMARY_W_RISK  = 0.05

# ================= ابزارها (Data Layor) =================
def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c: c.lower() for c in df.columns}
    d = df.rename(columns=col_map).copy()
    alias = {"o":"open","h":"high","l":"low","c":"close","v":"volume","price":"close"}
    for k,v in alias.items():
        if k in d.columns and v not in d.columns:
            d[v] = d[k]
    for req in ("open","high","low","close"):
        if req not in d.columns:
            raise ValueError(f"missing column: {req}")
    if "volume" not in d.columns:
        d["volume"] = 0.0
    return d

def retry_call(fn, desc="call", n:int=3, delay:float=0.5):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            if i == n-1:
                raise
            log.debug("retry %s %d/%d: %s", desc, i+1, n, e)
            time.sleep(delay)

def map_symbol_to_yf(symbol: str) -> str:
    base = symbol.split("/")[0].replace(":USDT","")
    return f"{base}-USD"

def build_data_exchange() -> Optional["ccxt.Exchange"]:
    if ccxt is None: return None
    try:
        if EXCHANGE_NAME == "binance":
            ex = ccxt.binance({"options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "bybit":
            ex = ccxt.bybit({"options":{"defaultType":"future"}})
        elif EXCHANGE_NAME == "okx":
            ex = ccxt.okx({"options":{"defaultType":"future"}})
        else:
            return None
        ex.apiKey = API_KEY or None
        ex.secret = API_SECRET or None
        ex.set_sandbox_mode(TESTNET)
        retry_call(ex.load_markets, "load_markets")
        return ex
    except Exception as e:
        log.warning("exchange init failed: %s", e)
        return None

def fetch_ohlcv(symbol:str, timeframe:str, limit:int=900) -> pd.DataFrame:
    # 1) Try exchange via ccxt
    def _from_ex():
        ex = build_data_exchange()
        if not ex: return None
        rows = retry_call(lambda: ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit),
                          f"fetch_ohlcv({symbol},{timeframe})")
        if rows and len(rows)>0:
            df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        return None
    try:
        d = _from_ex()
        if d is not None: return d
    except Exception as e:
        log.warning("exchange fetch failed %s %s", symbol, e)

    # 2) Fallback to yfinance
    try:
        if yf is None: return pd.DataFrame()
        yf_sym = map_symbol_to_yf(symbol)
        tf_map = {"1m":"1m","3m":"2m","5m":"5m","15m":"15m","30m":"30m","1h":"60m","4h":"60m","1d":"1d"}
        itv = tf_map.get(timeframe, "15m")
        period = "7d" if itv.endswith("m") else "1mo"
        dd = yf.download(yf_sym, interval=itv, period=period, progress=False)
        if dd is None or dd.empty: return pd.DataFrame()
        dd = dd.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        dd.index.name = "timestamp"
        return dd[["open","high","low","close","volume"]].dropna()
    except Exception as e:
        log.warning("yf fallback failed %s %s", symbol, e)
        return pd.DataFrame()

def fetch_htf_confirm(symbol:str, htf:str, limit:int=360) -> Optional[pd.Series]:
    d = fetch_ohlcv(symbol, htf, limit)
    if d is None or d.empty: return None
    d["ema20"]  = ema(d["close"],20)
    d["ema50"]  = ema(d["close"],50)
    d["ema200"] = ema(d["close"],200)
    m, s = macd(d["close"])
    d["macd"], d["macds"] = m, s
    d.dropna(inplace=True)
    return d.iloc[-1] if len(d) else None

# ================= اندیکاتورهای پایه =================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period:int=14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    gain_s = pd.Series(gain, index=series.index).rolling(period).mean()
    loss_s = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain_s/(loss_s + 1e-9)
    return 100 - (100/(1+rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(series, fast)
    es = ema(series, slow)
    m  = ef - es
    s  = m.ewm(span=signal, adjust=False).mean()
    return m,s

def bollinger(series: pd.Series, window=20, n=2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + n*sd
    lower = ma - n*sd
    bbpos = (series - ma) / (2*sd + 1e-9)
    return upper, lower, bbpos

def atr(df: pd.DataFrame, period:int=14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    hl = h-l
    hc = (h-c.shift()).abs()
    lc = (l-c.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def volatility_regime(close: pd.Series, window:int=50) -> pd.Series:
    r = close.pct_change()
    v = r.rolling(window).std()
    return (v / (v.rolling(window).mean() + 1e-9)).clip(0,3)

# ============== افزوده‌های Balanced =================
def supertrend_dir(df: pd.DataFrame, period:int=10, multiplier:float=3.0) -> pd.Series:
    d = df.copy()
    h,l,c = d["high"], d["low"], d["close"]
    tr1 = (h - l).abs()
    tr2 = (h - c.shift()).abs()
    tr3 = (l - c.shift()).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    atr_st = tr.ewm(span=period, adjust=False).mean()

    hl2 = (h + l) / 2.0
    basic_ub = hl2 + multiplier * atr_st
    basic_lb = hl2 - multiplier * atr_st

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(d)):
        final_ub.iloc[i] = (basic_ub.iloc[i] if (basic_ub.iloc[i] < final_ub.iloc[i-1]) or (c.iloc[i-1] > final_ub.iloc[i-1])
                            else final_ub.iloc[i-1])
        final_lb.iloc[i] = (basic_lb.iloc[i] if (basic_lb.iloc[i] > final_lb.iloc[i-1]) or (c.iloc[i-1] < final_lb.iloc[i-1])
                            else final_lb.iloc[i-1])

    st = pd.Series(index=d.index, dtype=float)
    st.iloc[0] = 0.0
    for i in range(1, len(d)):
        prev = st.iloc[i-1]
        if c.iloc[i] > final_ub.iloc[i]:
            st.iloc[i] = 1.0
        elif c.iloc[i] < final_lb.iloc[i]:
            st.iloc[i] = -1.0
        else:
            st.iloc[i] = prev
    return st.fillna(0.0)

def ichimoku(df: pd.DataFrame):
    d = df.copy()
    high, low, close = d["high"], d["low"], d["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    spanA  = ((tenkan + kijun) / 2.0).shift(26)
    spanB  = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    cloud_top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    cloud_bot = pd.concat([spanA, spanB], axis=1).min(axis=1)
    above_cloud = (close > cloud_top).astype(int)
    below_cloud = (close < cloud_bot).astype(int)
    inside_cloud = ((close <= cloud_top) & (close >= cloud_bot)).astype(int)
    tk_cross = (tenkan > kijun).astype(int) - (tenkan < kijun).astype(int)
    return tenkan, kijun, spanA, spanB, above_cloud, below_cloud, inside_cloud, tk_cross

def cmf(df: pd.DataFrame, period:int=20) -> pd.Series:
    d = df.copy()
    mfm = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / ((d["high"] - d["low"]).replace(0,np.nan))
    mfv = (mfm.fillna(0.0)) * d["volume"]
    cmf_v = (mfv.rolling(period).sum()) / (d["volume"].rolling(period).sum().replace(0,np.nan))
    return cmf_v.fillna(0.0)

def risk_flag_series(df: pd.DataFrame, look:int=60) -> pd.Series:
    r = df["close"].pct_change()
    var95 = r.rolling(look).apply(lambda x: np.nanpercentile(x.dropna(), 5) if len(x.dropna())>5 else np.nan)
    thresh = -0.03
    flag = np.where(var95 > thresh, 1, -1)
    return pd.Series(flag, index=df.index).fillna(1)

def add_fibo_levels(df: pd.DataFrame, swing_window:int=120) -> pd.DataFrame:
    c = df["close"]
    hh = c.rolling(swing_window).max()
    ll = c.rolling(swing_window).min()
    rng = (hh-ll).replace(0, np.nan)
    df["fib_000"], df["fib_100"] = ll, hh
    df["fib_236"] = ll + 0.236*rng
    df["fib_382"] = ll + 0.382*rng
    df["fib_500"] = ll + 0.500*rng
    df["fib_618"] = ll + 0.618*rng
    df["fib_786"] = ll + 0.786*rng
    for lvl in ("fib_236","fib_382","fib_500","fib_618","fib_786"):
        df[f"dist_{lvl}"] = (df["close"]/(df[lvl]+1e-9)) - 1.0
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = normalize_ohlcv_columns(df)
    d["ema20"]  = ema(d["close"], 20)
    d["ema50"]  = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi"]    = rsi(d["close"], 14)
    m, s        = macd(d["close"], 12, 26, 9)
    d["macd"], d["macds"] = m, s
    d["atr"]    = atr(d, 14)
    bbU, bbL, bbpos = bollinger(d["close"], 20, 2.0)
    d["bbU"], d["bbL"], d["bbM"], d["bbpos"] = bbU, bbL, (bbU+bbL)/2.0, bbpos
    d["vol_regime"] = volatility_regime(d["close"], 50)
    d = add_fibo_levels(d, swing_window=120)
    for k in (20,50,200):
        d[f"ema{k}_dist"] = (d["close"] - d[f"ema{k}"]) / (d["close"] + 1e-9)
    for w in (20,50):
        d[f"vol{w}"] = d["close"].pct_change().rolling(w).std()
    d["atr_norm"] = d["atr"] / (d["close"] + 1e-9)

    # Balanced extras
    d["super_dir"] = supertrend_dir(d, period=10, multiplier=3.0)
    tnk, kj, spA, spB, abv, blw, ins, tkx = ichimoku(d)
    d["ichi_tenkan"], d["ichi_kijun"] = tnk, kj
    d["ichi_spanA"], d["ichi_spanB"] = spA, spB
    d["ichi_above"], d["ichi_below"], d["ichi_inside"] = abv, blw, ins
    d["ichi_tk_cross"] = tkx
    d["cmf20"] = cmf(d, period=20)
    d["risk_flag"] = risk_flag_series(d, look=60)

    d.dropna(inplace=True)
    return d

# ================== Price Action ==================
def _pivots(h: pd.Series, l: pd.Series, left:int=3, right:int=3):
    hh_left = h.shift(1).rolling(left).max()
    ll_left = l.shift(1).rolling(left).min()
    hh_right = h.shift(-1).rolling(right).max()
    ll_right = l.shift(-1).rolling(right).min()
    ph = ((h > hh_left) & (h.shift(-1) >= hh_right)).fillna(False).astype(int)
    pl = ((l < ll_left) & (l.shift(-1) <= ll_right)).fillna(False).astype(int)
    return ph, pl

def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    o,h,l,c = d["open"], d["high"], d["low"], d["close"]
    ph, pl = _pivots(h,l, left=3, right=3)
    d["pivot_high"], d["pivot_low"] = ph, pl

    d["last_swh"] = np.where(ph==1, h, np.nan)
    d["last_swl"] = np.where(pl==1, l, np.nan)
    d["last_swh"] = pd.Series(d["last_swh"]).ffill()
    d["last_swl"] = pd.Series(d["last_swl"]).ffill()

    d["ms_up"]   = ((h > d["last_swh"].shift(1)) & (l > d["last_swl"].shift(1))).astype(int)
    d["ms_down"] = ((h < d["last_swh"].shift(1)) & (l < d["last_swl"].shift(1))).astype(int)

    d["bos_up"] = (h > d["last_swh"].shift(1)).astype(int)
    d["bos_dn"] = (l < d["last_swl"].shift(1)).astype(int)

    d["fvg_up"] = (((l.shift(-1) > h.shift(1)) | (l > h.shift(2))).fillna(False)).astype(int)
    d["fvg_dn"] = (((h.shift(-1) < l.shift(1)) | (h < l.shift(2))).fillna(False)).astype(int)

    prev_high, prev_low = h.shift(1), l.shift(1)
    d["liq_sweep_up"] = ((h > prev_high) & (c < prev_high)).fillna(False).astype(int)
    d["liq_sweep_dn"] = ((l < prev_low)  & (c > prev_low)).fillna(False).astype(int)

    d["engulf_bull"] = ((c>o) & (d["close"].shift(1)<d["open"].shift(1)) &
                        (c>d["open"].shift(1)) & (o<d["close"].shift(1))).fillna(False).astype(int)
    d["engulf_bear"] = ((c<o) & (d["close"].shift(1)>d["open"].shift(1)) &
                        (c<d["open"].shift(1)) & (o>d["close"].shift(1))).fillna(False).astype(int)

    prev_bear = (d["close"].shift(1) < d["open"].shift(1)).astype(int)
    prev_bull = (d["close"].shift(1) > d["open"].shift(1)).astype(int)
    d["bull_ob"] = ((prev_bear==1) & (d["bos_up"]==1)).astype(int)
    d["bear_ob"] = ((prev_bull==1) & (d["bos_dn"]==1)).astype(int)

    pa_raw = (
        + 0.9*d["ms_up"]   - 0.9*d["ms_down"]
        + 0.8*d["bos_up"]  - 0.8*d["bos_dn"]
        + 0.4*d["engulf_bull"] - 0.4*d["engulf_bear"]
        + 0.5*d["bull_ob"] - 0.5*d["bear_ob"]
        - 0.5*d["liq_sweep_up"] + 0.5*d["liq_sweep_dn"]
        + 0.2*d["fvg_up"]  - 0.2*d["fvg_dn"]
    )
    m = pa_raw.rolling(150, min_periods=30).apply(lambda x: np.nanmax(np.abs(x)) if len(x)>0 else 1.0)
    m = m.replace(0,1.0).fillna(method="ffill").fillna(1.0)
    d["pa_score"] = (pa_raw / m).clip(-1,1).fillna(0.0)
    d["pa_score_pos"] = ((d["pa_score"]+1)/2.0).clip(0,1)
    return d

# ================ Order Flow (OHLCV-based proxy) ================
def calc_vwap(df: pd.DataFrame, session: str = "rolling", window: int = 96) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan).fillna(0.0)
    if session == "cumulative":
        pv = (tp * vol).cumsum()
        vv = vol.cumsum().replace(0, np.nan)
        return (pv / vv).fillna(method="bfill").fillna(method="ffill")
    pv = (tp * vol).rolling(window, min_periods=max(10, window//5)).sum()
    vv = vol.rolling(window, min_periods=max(10, window//5)).sum().replace(0, np.nan)
    return (pv / vv).fillna(method="bfill").fillna(method="ffill")

def cumulative_delta(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    body = (df["close"] - df.get("open", df["close"].shift(1).fillna(df["close"])))
    span = (df["high"] - df["low"]).replace(0, np.nan).fillna(1e-9)
    power = (body / span).clip(-1, 1)
    delta = power * df["volume"]
    cvd = delta.ewm(alpha=alpha, adjust=False).mean().cumsum()
    return cvd

def volume_profile(df: pd.DataFrame, lookback: int = 384, bins: int = 24):
    d = df.iloc[-lookback:].copy()
    if len(d) < 10: return np.nan, np.nan, np.nan
    prices = d["close"].values
    vols   = d["volume"].values
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if hi <= lo: return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins+1)
    idx   = np.clip(np.digitize(prices, edges)-1, 0, bins-1)
    vol_hist = np.zeros(bins)
    for i, v in zip(idx, vols): vol_hist[i] += v
    if vol_hist.sum() <= 0: return np.nan, np.nan, np.nan
    vpoc_bin = int(np.argmax(vol_hist))
    center = (edges[vpoc_bin] + edges[vpoc_bin+1]) * 0.5
    nz = np.where(vol_hist>0, vol_hist, np.nan)
    hvn_bin = int(np.nanargmax(nz)) if np.isfinite(nz).any() else vpoc_bin
    lvn_bin = int(np.nanargmin(nz)) if np.isfinite(nz).any() else vpoc_bin
    hvn = (edges[hvn_bin] + edges[hvn_bin+1]) * 0.5
    lvn = (edges[lvn_bin] + edges[lvn_bin+1]) * 0.5
    return center, hvn, lvn

def liquidity_pools(df: pd.DataFrame, lookback: int = 40, tol: float = 0.0008):
    close = df["close"].values
    hi = df["high"].values
    lo = df["low"].values
    pools_up = np.zeros(len(df), dtype=int)
    pools_dn = np.zeros(len(df), dtype=int)
    for i in range(2, len(df)):
        lo_i, hi_i = lo[i], hi[i]
        win_hi = hi[max(0, i-lookback):i]
        win_lo = lo[max(0, i-lookback):i]
        if np.any(np.abs(win_hi - hi_i)/max(close[i],1e-9) < tol):
            pools_up[i] = 1
        if np.any(np.abs(win_lo - lo_i)/max(close[i],1e-9) < tol):
            pools_dn[i] = 1
    return pd.Series(pools_up, index=df.index), pd.Series(pools_dn, index=df.index)

def imbalance_metric(df: pd.DataFrame):
    opn = df.get("open", df["close"].shift(1).fillna(df["close"]))
    rng = (df["high"] - df["low"]).replace(0, np.nan).fillna(1e-9)
    upv = ((df["close"] - opn).clip(lower=0) / rng) * df["volume"]
    dnv = ((opn - df["close"]).clip(lower=0) / rng) * df["volume"]
    imb = (upv - dnv) / (upv + dnv + 1e-9)
    return imb.fillna(0.0)

def add_order_flow(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["vwap"] = calc_vwap(d, session="rolling", window=96)
    d["vwap_dist"] = (d["close"] / (d["vwap"]+1e-9)) - 1.0

    d["cvd"] = cumulative_delta(d, alpha=0.4)
    d["cvd_ma_fast"] = d["cvd"].ewm(span=20, adjust=False).mean()
    d["cvd_ma_slow"] = d["cvd"].ewm(span=60, adjust=False).mean()
    d["cvd_trend_up"] = (d["cvd_ma_fast"] > d["cvd_ma_slow"]).astype(int)
    d["cvd_trend_dn"] = (d["cvd_ma_fast"] < d["cvd_ma_slow"]).astype(int)

    vpoc, hvn, lvn = volume_profile(d, lookback=384, bins=24)
    d["vpoc_price"] = vpoc
    d["hvn_price"]  = hvn
    d["lvn_price"]  = lvn
    d["dist_vpoc"] = (d["close"]/(d["vpoc_price"]+1e-9)) - 1.0 if np.isfinite(vpoc) else 0.0

    lp_up, lp_dn = liquidity_pools(d, lookback=40, tol=0.0008)
    d["liq_pool_up"] = lp_up
    d["liq_pool_dn"] = lp_dn

    d["imbalance"] = imbalance_metric(d)

    of_score = (
        0.9 * d["cvd_trend_up"] - 0.9 * d["cvd_trend_dn"] +
        0.7 * (d["imbalance"].clip(-1,1)) +
        0.5 * (-(d["vwap_dist"]).clip(-1,1)) +
        0.4 * (-(d["dist_vpoc"])).clip(-1,1) +
        -0.6 * d["liq_pool_up"] + 0.6 * d["liq_pool_dn"]
    )
    m = of_score.rolling(150, min_periods=30).apply(lambda x: np.nanmax(np.abs(x)) if len(x)>0 else 1.0).replace(0,1.0)
    d["of_score"] = (of_score / m).clip(-1,1).fillna(0.0)
    d["of_score_pos"] = ((d["of_score"]+1)/2.0)
    return d

# ===================== AI Features / Classifier & Regression =====================
AI_FEATURES = [
    "ema_trend","macd_cross","rsi_overbought","rsi_oversold",
    "candle_bull","bb_break_upper","bb_break_lower","bbpos_clip",
    "atr_norm","vol_regime",
    # PA
    "ms_up","ms_down","bos_up","bos_dn","fvg_up","fvg_dn",
    "bull_ob","bear_ob","liq_sweep_up","liq_sweep_dn",
    "engulf_bull","engulf_bear","pa_score",
    # OF
    "cvd_trend_up","cvd_trend_dn","imbalance","vwap_dist","dist_vpoc","of_score",
    # Fibo distances
    "dist_fib_236","dist_fib_382","dist_fib_500","dist_fib_618","dist_fib_786",
    # Balanced extras
    "supertrend_sig","ichimoku_sig","cmf_pos","risk_good"
]

def build_ai_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["ema_trend"]      = (x["close"]>x["ema20"]).astype(int) + (x["close"]>x["ema50"]).astype(int)
    x["macd_cross"]     = (x["macd"]>x["macds"]).astype(int)
    x["rsi_overbought"] = (x["rsi"]>70).astype(int)
    x["rsi_oversold"]   = (x["rsi"]<30).astype(int)
    x["candle_bull"]    = (x["close"]>x["open"]).astype(int) if "open" in x else (x["ret"]>0).astype(int)
    x["bb_break_upper"] = (x["close"]>x["bbU"]).astype(int)
    x["bb_break_lower"] = (x["close"]<x["bbL"]).astype(int)
    x["bbpos_clip"]     = np.clip(x["bbpos"], -1.0, 1.0)
    x["atr_norm"]       = (x["atr"]/(x["close"]+1e-9)).fillna(0.0)
    x["vol_regime"]     = x["vol_regime"].clip(0.0, 3.0)
    # Balanced extras
    x["supertrend_sig"] = (x.get("super_dir", 0) > 0).astype(int)
    ichi_sig = ((x.get("ichi_tenkan", 0) > x.get("ichi_kijun", 0)) &
                (x["close"] > pd.concat([x.get("ichi_spanA", x["close"]), x.get("ichi_spanB", x["close"])], axis=1).max(axis=1))).astype(int)
    x["ichimoku_sig"] = ichi_sig
    x["cmf_pos"] = (x.get("cmf20", 0) > 0).astype(int)
    x["risk_good"] = (x.get("risk_flag", 1) > 0).astype(int)

    for col in ["ms_up","ms_down","bos_up","bos_dn","fvg_up","fvg_dn",
                "bull_ob","bear_ob","liq_sweep_up","liq_sweep_dn",
                "engulf_bull","engulf_bear","pa_score",
                "cvd_trend_up","cvd_trend_dn","imbalance","vwap_dist","dist_vpoc","of_score",
                "dist_fib_236","dist_fib_382","dist_fib_500","dist_fib_618","dist_fib_786",
                "supertrend_sig","ichimoku_sig","cmf_pos","risk_good"]:
        if col not in x.columns:
            if col.startswith("dist_fib_"):
                key = col.replace("dist_","")
                x[col] = x.get(key, pd.Series(0.0, index=x.index))
            else:
                x[col] = 0

    x["target"] = (x["close"].shift(-1) > x["close"]).astype(int)
    return x.dropna()

def model_path(symbol:str, tf:str)->str:
    safe = symbol.replace("/","_")
    return os.path.join(MODEL_DIR, f"xgb_{safe}_{tf}.joblib")

def model_path_reg(symbol:str, tf:str, tag:str="huber")->str:
    safe = symbol.replace("/","_")
    return os.path.join(MODEL_DIR, f"gbr_delta_{tag}_{safe}_{tf}.joblib")

def _purged_time_series_split(n_samples:int, n_splits:int, gap:int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tscv.split(np.arange(n_samples)):
        te_start = te[0]
        tr = tr[tr < (te_start - gap)]
        if len(tr)==0:
            continue
        yield tr, te

def _make_classifier():
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1
        )
    return GradientBoostingClassifier(random_state=42)

def ai_train_and_prob_up(d: pd.DataFrame, symbol:str, timeframe:str) -> Optional[float]:
    if len(d) < max(100, AI_TRAIN_WINDOW//2): return None
    x = build_ai_features(d)
    if len(x) < max(100, AI_TRAIN_WINDOW//2): return None
    xw = x.iloc[-AI_TRAIN_WINDOW:] if len(x) > AI_TRAIN_WINDOW else x.copy()
    last_feats = xw.iloc[[-1]][AI_FEATURES]
    X = xw[AI_FEATURES]
    y = xw["target"]
    if len(X) < 120: return None
    base_model = _make_classifier()
    if len(X) >= 160 and hasattr(base_model, "fit"):
        for tr_idx, te_idx in _purged_time_series_split(len(X), max(2, min(CV_FOLDS, len(X)//40)), CV_GAP):
            base_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    base_model.fit(X.iloc[:-1], y.iloc[:-1])
    if hasattr(base_model, "predict_proba"):
        prob_up = float(base_model.predict_proba(last_feats)[0,1])*100.0
    else:
        p = float(base_model.predict(last_feats)[0])
        prob_up = 60.0 if p==1 else 40.0

    try:
        if CALIBRATE_PROBS and len(X) >= 200 and hasattr(base_model, "predict_proba"):
            cut = int(len(X)*0.85)
            mdl_prefit = _make_classifier()
            mdl_prefit.fit(X.iloc[:cut], y.iloc[:cut])
            calib = CalibratedClassifierCV(base_estimator=mdl_prefit, method="isotonic", cv="prefit")
            calib.fit(X.iloc[cut:], y.iloc[cut:])
            prob_up = float(calib.predict_proba(last_feats)[0,1])*100.0
            dump(calib, model_path(symbol, timeframe))
        else:
            dump(base_model, model_path(symbol, timeframe))
    except Exception as e:
        log.debug("Calibration/save failed: %s", e)
        try: dump(base_model, model_path(symbol, timeframe))
        except: pass
    return float(round(prob_up,2))

# ---- Δ$ Regression + Quantiles ----
def _build_reg_features(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    x = d.copy()
    x["ret1"] = np.log(x["close"] / x["close"].shift(1))
    x["ret2"] = np.log(x["close"].shift(1) / x["close"].shift(2))
    x["ret3"] = np.log(x["close"].shift(2) / x["close"].shift(3))
    x["vol20"] = x["ret1"].rolling(20).std()
    x["vol50"] = x["ret1"].rolling(50).std()
    x["atr_norm"] = (x["atr"]/(x["close"]+1e-9))
    x["ema20_dist"]  = (x["close"]/ (x["ema20"]+1e-9)) - 1.0
    x["ema50_dist"]  = (x["close"]/ (x["ema50"]+1e-9)) - 1.0
    x["ema200_dist"] = (x["close"]/ (x["ema200"]+1e-9)) - 1.0
    x["macd_hist"] = x["macd"] - x["macds"]
    x["bb_width"]  = ((x["bbU"] - x["bbL"]) / (x["close"]+1e-9))
    x["bb_pos"]    = np.clip(x["bbpos"], -1.5, 1.5)
    x["dist_fib_382"] = x.get("dist_fib_382", (x["close"]/(x.get("fib_382", x["close"])+1e-9))-1.0)
    x["dist_fib_618"] = x.get("dist_fib_618", (x["close"]/(x.get("fib_618", x["close"])+1e-9))-1.0)
    x["y_logret"] = np.log(x["close"].shift(-1) / x["close"])
    cols = ["ret1","ret2","ret3","vol20","vol50","atr_norm",
            "ema20_dist","ema50_dist","ema200_dist","macd_hist",
            "bb_width","bb_pos","rsi","vol_regime","dist_fib_382","dist_fib_618"]
    x = x[cols+["y_logret","close","atr"]].dropna()
    return x, cols

def _fit_gbr(X, y, **kw):
    m = GradientBoostingRegressor(**kw)
    m.fit(X, y)
    return m

def ai_train_and_pred_delta(d: pd.DataFrame, symbol:str, timeframe:str) -> Optional[Tuple[float,float, Optional[float], Optional[float]]]:
    try:
        if len(d) < max(120, AI_TRAIN_WINDOW//2): return None
        feats, cols = _build_reg_features(d)
        if len(feats) < max(120, AI_TRAIN_WINDOW//2): return None
        w = feats.iloc[-AI_TRAIN_WINDOW:] if len(feats) > AI_TRAIN_WINDOW else feats.copy()
        X_tr = w.iloc[:-1][cols]; y_tr = w.iloc[:-1]["y_logret"]; X_last = w.iloc[[-1]][cols]
        last_close = float(w.iloc[-1]["close"]); last_atr = float(w.iloc[-1]["atr"])
        if len(X_tr) < 100: return None

        path_point = model_path_reg(symbol, timeframe, "huber")
        try:
            m_point = joblib_load(path_point)
        except Exception:
            m_point = None

        m_point = _fit_gbr(
            X_tr, y_tr,
            loss="huber", alpha=0.9, n_estimators=900, learning_rate=0.045,
            max_depth=3, subsample=0.9, random_state=42
        )
        try: dump(m_point, path_point)
        except Exception as e: log.debug("Reg save failed: %s", e)

        yhat = float(m_point.predict(X_last)[0])
        delta_raw = last_close*(math.exp(yhat) - 1.0)
        atr_clip = max(1e-9, DELTA_CLIP_ATR_MULT*last_atr)
        delta_clip = float(np.clip(delta_raw, -atr_clip, atr_clip))

        # quantiles
        q_lo, q_hi = 0.2, 0.8
        path_lo = model_path_reg(symbol, timeframe, f"q{int(q_lo*100)}")
        path_hi = model_path_reg(symbol, timeframe, f"q{int(q_hi*100)}")
        try:
            m_lo = joblib_load(path_lo); m_hi = joblib_load(path_hi)
        except Exception:
            m_lo = None; m_hi = None
        m_lo = _fit_gbr(X_tr, y_tr, loss="quantile", alpha=q_lo, n_estimators=650, learning_rate=0.05,
                        max_depth=3, subsample=0.9, random_state=43)
        m_hi = _fit_gbr(X_tr, y_tr, loss="quantile", alpha=q_hi, n_estimators=650, learning_rate=0.05,
                        max_depth=3, subsample=0.9, random_state=44)
        try: dump(m_lo, path_lo); dump(m_hi, path_hi)
        except Exception: pass

        y_lo = float(m_lo.predict(X_last)[0]); y_hi = float(m_hi.predict(X_last)[0])
        d_low  = last_close*(math.exp(y_lo) - 1.0)
        d_high = last_close*(math.exp(y_hi) - 1.0)

        return round(delta_raw, 2), round(delta_clip, 2), round(d_low,2), round(d_high,2)
    except Exception as e:
        log.debug("Δ$ regression error: %s", e)
        return None

# ===================== Fundamentals (RSS-only sentiment) =====================
def _funda_from_text(text:str)->float:
    if TextBlob is None: return 0.0
    try: return float(TextBlob(text).sentiment.polarity)
    except Exception: return 0.0

def sym_base(symbol:str) -> str:
    return symbol.split("/")[0].split("-")[0].upper()

def fundamental_sentiment(symbol:str)->Tuple[str, Optional[float]]:
    base = sym_base(symbol)
    now = time.time()
    if base in _FUND_CACHE and now - _FUND_CACHE[base][0] <= FUND_TTL_SEC:
        return _FUND_CACHE[base][1], _FUND_CACHE[base][2]
    urls = FUND_FEEDS.get(base)
    if not urls or feedparser is None:
        _FUND_CACHE[base] = (now, "N/A", None); return "N/A", None
    scores=[]
    try:
        for url in urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                t = getattr(entry, "title", "") or ""
                if not t: continue
                scores.append(_funda_from_text(t))
    except Exception as e:
        log.debug("fundamental fetch failed %s: %s", base, e)
    if not scores:
        _FUND_CACHE[base] = (now, "N/A", None); return "N/A", None
    avg = sum(scores)/len(scores)
    perc = round(((avg+1)/2)*100.0, 2)
    label = "Bullish 📈" if avg>0.05 else ("Bearish 📉" if avg<-0.05 else "Neutral ⚖️")
    _FUND_CACHE[base] = (now, label, perc)
    return label, perc

# ===================== Signal/Scoring =====================
def score_signal(last: pd.Series) -> Tuple[float,str,str,float]:
    ema_up   = 1 if (last["close"]>last["ema20"]>last["ema50"]) else (-1 if (last["close"]<last["ema20"]<last["ema50"]) else 0)
    mac_up   = 1 if last["macd"]>last["macds"] else (-1 if last["macd"]<last["macds"] else 0)
    pa_sc    = float(last.get("pa_score",0.0))
    of_sc    = float(last.get("of_score",0.0))
    st_dir   = int(np.sign(last.get("super_dir",0)))
    ichi_above = int(last.get("ichi_above",0))
    ichi_tk    = int(np.sign(last.get("ichi_tk_cross",0)))
    ichi_sig   = 1 if (ichi_above==1 and ichi_tk>0) else (-1 if (int(last.get("ichi_below",0))==1 and ichi_tk<0) else 0)
    cmf_pos = 1 if float(last.get("cmf20",0.0))>0 else (-1 if float(last.get("cmf20",0.0))<0 else 0)
    risk_ok = 1 if int(last.get("risk_flag",1))>0 else -1
    w = {"ema":0.20,"mac":0.10,"pa":0.20,"of":0.20,"st":0.10,"ichi":0.10,"cmf":0.05,"risk":0.05}
    score = (
        w["ema"]*ema_up +
        w["mac"]*mac_up +
        w["pa"] *pa_sc +
        w["of"] *of_sc +
        w["st"] *st_dir +
        w["ichi"]*ichi_sig +
        w["cmf"]*cmf_pos +
        w["risk"]*risk_ok
    )
    score = float(np.clip(score, -1.0, 1.0))
    thr = 0.15
    if score > thr:
        trend, color = "Bullish", "#16a34a"
    elif score < -thr:
        trend, color = "Bearish", "#dc2626"
    else:
        trend, color = "Neutral", ""
    cof = float(np.clip(40.0 + 45.0*abs(score), 10.0, 90.0))
    return round(score,3), trend, color, round(cof,2)

def prediction_targets(last_price: float, atr_val: float, score: float, trend: str,
                      ai_align: Optional[float] = None) -> Tuple[float, float]:
    a = atr_val if atr_val > 0 else last_price * 0.01
    vol = float(np.clip(a / (last_price + 1e-9), 0, 0.05))
    base_k = 0.9 + 0.6 * (abs(score) / 1.0)
    k = base_k * (1.0 - 0.7 * vol)
    if ai_align is not None:
        k *= (0.92 + 0.16 * ai_align)
    if trend == "Bullish":
        tp = last_price + k * 1.35 * a
        sl = last_price - k * 0.95 * a
    elif trend == "Bearish":
        tp = last_price - k * 1.35 * a
        sl = last_price + k * 0.95 * a
    else:
        tp = last_price + k * 0.65 * a
        sl = last_price - k * 0.65 * a
    return round(float(tp), 2), round(float(sl), 2)

@dataclass
class SignalResult:
    trend: str
    score: float
    confidence: float
    color: str
    last: float
    atr: float
    tp: float
    sl: float
    ai_prob_up: Optional[float] = None
    htf_bias: str = "flat"
    vol_regime: float = 1.0
    delta_usd_raw: Optional[float] = None
    delta_usd_clip: Optional[float] = None
    delta_low: Optional[float] = None
    delta_high: Optional[float] = None
    funda_label: str = "N/A"
    funda_pct: Optional[float] = None

def analyze_symbol(symbol: str, timeframe: str) -> Optional[SignalResult]:
    df = fetch_ohlcv(symbol, timeframe, limit=900)
    if df is None or df.empty: return None
    d = compute_indicators(df)
    if d is None or d.empty: return None
    d = add_price_action(d)
    d = add_order_flow(d)
    # HTF confirm
    h_bias = "flat"
    if USE_HTF:
        h = fetch_htf_confirm(symbol, HTF_TF, limit=360)
        if h is not None:
            if (h["close"] > h["ema20"] > h["ema50"] > h["ema200"]) and (h["macd"] > h["macds"]):
                h_bias = "bull"
            elif (h["close"] < h["ema20"] < h["ema50"] < h["ema200"]) and (h["macd"] < h["macds"]):
                h_bias = "bear"

    last = d.iloc[-1]
    score, trend, color, conf = score_signal(last)

    # AI Prob(up)
    ai_prob = None
    ai_align = None
    if AI_ENABLED:
        ai_prob = ai_train_and_prob_up(d, symbol, timeframe)
        if ai_prob is not None:
            p = ai_prob / 100.0
            ai_align = p if trend == "Bullish" else (1.0 - p) if trend == "Bearish" else 0.5

    # Δ$ prediction
    delta_raw = delta_clip = d_lo = d_hi = None
    if AI_ENABLED:
        res = ai_train_and_pred_delta(d, symbol, timeframe)
        if res is not None:
            delta_raw, delta_clip, d_lo, d_hi = res

    # Soft filters
    if SOFT_FILTERS:
        if ai_align is not None:
            conf *= (0.9 + 0.2 * float(np.clip(ai_align, 0.0, 1.0)))
        if USE_HTF and h_bias != "flat" and trend in ("Bullish", "Bearish"):
            if (trend == "Bullish" and h_bias == "bull") or (trend == "Bearish" and h_bias == "bear"):
                conf *= 1.07
            elif (trend == "Bullish" and h_bias == "bear") or (trend == "Bearish" and h_bias == "bull"):
                conf *= 0.90
        reg = float(np.clip(last.get("vol_regime", 1.0), 0.0, 3.0))
        if reg > 1.7: conf *= 0.92
        if "fib_500" in last.index and (last["fib_500"] or last["fib_500"] == 0):
            near_mid = abs((last["close"] / (last["fib_500"] + 1e-9)) - 1.0)
            if near_mid < 0.004: conf *= 0.88

    # Mix ML/PA/OF confidence
    pa = float(last.get("pa_score_pos", 0.5))
    of = float(last.get("of_score_pos", 0.5))
    ml_ind_conf = conf / 100.0
    den = max(1e-9, W_ML_IND + W_PA + W_OF)
    conf_mix = (W_ML_IND * ml_ind_conf + W_PA * pa + W_OF * of) / den
    conf = float(np.clip(conf_mix * 100.0, 0.0, 100.0))

    trend_dir = 1 if trend == "Bullish" else (-1 if trend == "Bearish" else 0)
    of_dir = 1 if last.get("of_score", 0) > 0 else (-1 if last.get("of_score", 0) < 0 else 0)
    if trend_dir * of_dir < 0:
        conf *= 0.92
    conf = float(np.clip(conf, 0.0, 100.0))

    tp, sl = prediction_targets(float(last["close"]), float(last["atr"]), score, trend, ai_align)
    display_trend, display_color = trend, color
    if conf < CONFIDENCE_MIN:
        display_trend, display_color = "Neutral", ""

    f_label, f_pct = fundamental_sentiment(symbol)
    return SignalResult(
        trend=display_trend, score=score, confidence=round(conf, 2), color=display_color,
        last=float(last["close"]), atr=float(last["atr"]), tp=tp, sl=sl,
        ai_prob_up=ai_prob, htf_bias=h_bias, vol_regime=float(last.get("vol_regime", 1.0)),
        delta_usd_raw=delta_raw, delta_usd_clip=delta_clip, delta_low=d_lo, delta_high=d_hi,
        funda_label=f_label, funda_pct=f_pct
    )

# ============== Summary Bull% (no fundamentals inside calc) ==============
def _comp_htf_component(htf_bias: str) -> float:
    if htf_bias == "bull": return 0.60
    if htf_bias == "bear": return 0.40
    return 0.50

def _comp_risk_component(last: pd.Series) -> float:
    base = 0.60 if int(last.get("risk_flag", 1)) > 0 else 0.40
    reg = float(np.clip(last.get("vol_regime", 1.0), 0.0, 3.0))
    penalty = max(0.0, reg - 1.0) * 0.05
    return float(np.clip(base - penalty, 0.0, 1.0))

def build_summary_bull_percent(sig: SignalResult, last: pd.Series) -> Dict[str, float]:
    ai = (sig.ai_prob_up / 100.0) if isinstance(sig.ai_prob_up, (int, float)) else 0.50
    pa = float(last.get("pa_score_pos", 0.5))
    of = float(last.get("of_score_pos", 0.5))
    trend_comp = (sig.score + 1.0) / 2.0
    htf_comp = _comp_htf_component(sig.htf_bias)
    risk_comp = _comp_risk_component(last)
    bull = (
        SUMMARY_W_AI    * ai +
        SUMMARY_W_PA    * pa +
        SUMMARY_W_OF    * of +
        SUMMARY_W_TREND * trend_comp +
        SUMMARY_W_HTF   * htf_comp +
        SUMMARY_W_RISK  * risk_comp
    )
    bull_pct = float(np.clip(bull * 100.0, 0.0, 100.0))
    return {
        "bull_pct": round(bull_pct, 2),
        "ai_pct": round(ai*100.0, 2),
        "pa_pct": round(pa*100.0, 2),
        "of_pct": round(of*100.0, 2),
        "trend_pct": round(trend_comp*100.0, 2),
        "htf_pct": round(htf_comp*100.0, 2),
        "risk_pct": round(risk_comp*100.0, 2),
    }

# ============== Forecast rows ==============
def build_multi_candle_forecast(sig: SignalResult, horizon: int, use_bands: bool = USE_BANDS) -> List[Dict[str, object]]:
    decay = [1.00, 0.80, 0.65, 0.55] + [0.50] * max(0, horizon - 4)
    ai = sig.ai_prob_up if sig.ai_prob_up is not None else None
    ai_align = None
    if ai is not None:
        p = ai / 100.0
        ai_align = p if sig.trend == "Bullish" else (1.0 - p) if sig.trend == "Bearish" else 0.5
    price = sig.last
    base_dir = 1 if sig.trend == "Bullish" else (-1 if sig.trend == "Bearish" else (1 if (ai or 50) >= 50 else -1))
    raw_base = sig.delta_usd_raw if isinstance(sig.delta_usd_raw, (int, float)) else base_dir * 0.35 * sig.atr
    clip_base = sig.delta_usd_clip if isinstance(sig.delta_usd_clip, (int, float)) else float(
        np.clip(raw_base, -DELTA_CLIP_ATR_MULT * sig.atr, DELTA_CLIP_ATR_MULT * sig.atr)
    )
    lo_base = sig.delta_low if isinstance(sig.delta_low, (int, float)) else clip_base * 0.6
    hi_base = sig.delta_high if isinstance(sig.delta_high, (int, float)) else clip_base * 1.4

    rows = []
    for i in range(1, horizon + 1):
        dr = float(raw_base * decay[i - 1])
        dc = float(clip_base * decay[i - 1])
        dcl = float(lo_base * decay[i - 1])
        dch = float(hi_base * decay[i - 1])
        next_price = price + dc
        cof = max(0.2 * sig.confidence, sig.confidence * (1 - 0.08 * (i - 1)))
        tp_i, sl_i = prediction_targets(next_price, sig.atr, sig.score, sig.trend, ai_align)
        score_combo = round((0.55 * cof) + (0.45 * ((ai or 0.0))), 2) if ai is not None else round(cof, 2)
        row = {
            "candle": f"Candle +{i}",
            "tf": DEFAULT_TF,
            "trend": sig.trend,
            "price": next_price,
            "tp": tp_i,
            "sl": sl_i,
            "raw": dr,
            "clip": dc,
            "cof": cof,
            "ai": ai,
            "score": score_combo,
            "reg": sig.vol_regime,
            "htf": sig.htf_bias if sig.htf_bias != "flat" else "flat",
            "fanda": f"{sig.funda_label} {f'{sig.funda_pct:.2f}%' if isinstance(sig.funda_pct, (int, float)) else ''}".strip()
        }
        if use_bands:
            row["p_low"] = price + dcl
            row["p_high"] = price + dch
        rows.append(row)
        price = next_price
    return rows

# ============== Risk & Backtest (compact) ==============
def position_size(last_price: float, atr_val: float, confidence: float, winrate: float=0.54, rr: float=1.5):
    risk_cash = ACCOUNT_EQUITY * RISK_PER_TRADE
    risk_per_unit = atr_val if atr_val > 0 else last_price * 0.01
    units = risk_cash / max(risk_per_unit, 1e-9)
    kelly = winrate - (1 - winrate) / rr
    kelly = float(np.clip(kelly, 0.0, 0.25))
    units *= (0.6 + 0.4 * (kelly / 0.25))
    scale = 0.9 + 0.2 * float(np.clip(confidence / 100.0, 0.0, 1.0))
    units *= scale
    notional = units * last_price
    pos_norm = float(np.clip(notional / (ACCOUNT_EQUITY * PORTFOLIO_RISK_CAP), 0, 1))
    return round(pos_norm, 2), max(1e-8, units)

# ============================================================
#                    Multi-TF + API + WebUI
# ============================================================
from flask import Flask, request, jsonify, Response

VALID_TFS = ["15m","1h","4h","1d"]

app = Flask(__name__)

def analyze(symbol: str, tf: str):
    tf = tf if tf in VALID_TFS else DEFAULT_TF
    res = analyze_symbol(symbol, tf)
    if res is None:
        return None, None, None
    # also need last row for summary components:
    df = fetch_ohlcv(symbol, tf, limit=900)
    d = add_order_flow(add_price_action(compute_indicators(df)))
    last = d.iloc[-1]
    comps = build_summary_bull_percent(res, last)
    forecast = build_multi_candle_forecast(res, horizon=HORIZON, use_bands=USE_BANDS)
    pos_norm, units = position_size(res.last, res.atr, res.confidence, winrate=0.55, rr=1.6)

    payload = {
        "symbol": symbol,
        "tf": tf,
        "signal": asdict(res),
        "summary": {
            **comps,
            "label": "Bullish 📈" if comps["bull_pct"] >= 55 else ("Bearish 📉" if comps["bull_pct"] <= 45 else "Neutral ⚖️")
        },
        "forecast_rows": forecast,
        "position": {"pos_norm": pos_norm, "units": units},
        "meta": {
            "use_htf": USE_HTF, "htf_tf": HTF_TF, "bands": USE_BANDS,
            "weights_conf": {"ml_ind": W_ML_IND, "pa": W_PA, "of": W_OF},
            "weights_summary": {
                "ai": SUMMARY_W_AI, "pa": SUMMARY_W_PA, "of": SUMMARY_W_OF,
                "trend": SUMMARY_W_TREND, "htf": SUMMARY_W_HTF, "risk": SUMMARY_W_RISK
            }
        }
    }
    # also return a compact OHLC for chart
    chart = d[["open","high","low","close","volume"]].tail(300).reset_index()
    chart["timestamp"] = chart["timestamp"].astype(np.int64)//10**6  # ms
    return payload, chart.to_dict(orient="records"), d

@app.get("/api/analyze")
def api_analyze():
    symbol = request.args.get("symbol","BTC-USD")
    tf = request.args.get("tf", DEFAULT_TF)
    payload, chart, _ = analyze(symbol, tf)
    if payload is None:
        return jsonify({"ok": False, "error": "no data"}), 400
    return jsonify({"ok": True, "data": payload, "chart": chart})

@app.get("/api/backtest")
def api_backtest():
    symbol = request.args.get("symbol","BTC-USD")
    tf = request.args.get("tf", DEFAULT_TF)
    # light backtest: reuse analyze to ensure pipeline works
    df = fetch_ohlcv(symbol, tf, limit=1800)
    if df is None or df.empty or len(df) < 400:
        return jsonify({"ok": False, "error": "insufficient data"}), 400
    d = add_order_flow(add_price_action(compute_indicators(df)))
    equity = 1.0; peak = 1.0; wins = 0; losses = 0; rr_list=[]
    last_side=None; entry=None; sl=None; tp=None
    for i in range(60, len(d) - 1):
        row = d.iloc[i]
        s, trend, _, conf = score_signal(row)
        if AI_ENABLED and i > 200:
            sub = d.iloc[: i + 1].copy()
            ai = ai_train_and_prob_up(sub, symbol, tf)
        else:
            ai = None
        ai_align = None
        if ai is not None:
            p = ai / 100.0
            ai_align = p if trend == "Bullish" else (1.0 - p) if trend == "Bearish" else 0.5
        my_tp, my_sl = prediction_targets(row["close"], row["atr"], s, trend, ai_align)
        if last_side is None and conf >= CONFIDENCE_MIN and trend in ("Bullish", "Bearish"):
            last_side = "buy" if trend == "Bullish" else "sell"
            entry = row["close"]; tp = my_tp; sl = my_sl
            continue
        if last_side is not None:
            nxt = d.iloc[i + 1]
            hit_tp = (nxt["high"] >= tp) if last_side=="buy" else (nxt["low"] <= tp)
            hit_sl = (nxt["low"]  <= sl) if last_side=="buy" else (nxt["high"]>= sl)
            if hit_tp or hit_sl or i == len(d) - 2:
                R = abs(entry - sl)
                pnl_hit = ((tp - entry) if last_side == "buy" else (entry - tp)) if hit_tp else ((sl - entry) if last_side=="buy" else (entry - sl))
                rr = (pnl_hit / (R + 1e-9)) - ((2*(COST_BPS+SLIPPAGE_BPS))/10000.0)
                rr_list.append(rr)
                if rr > 0: wins += 1
                else: losses += 1
                equity *= (1 + 0.005 * rr)
                last_side = None; entry = None; sl = None; tp = None
    trades = wins + losses
    winrate = (wins / trades * 100.0) if trades else 0.0
    avg_rr = float(np.mean(rr_list)) if rr_list else 0.0
    pnl_pct = (equity - 1.0) * 100.0
    out = {"symbol":symbol,"tf":tf,"trades":trades,"winrate":round(winrate,2),"avg_rr":round(avg_rr,2),"pnl_pct":round(pnl_pct,2)}
    return jsonify({"ok": True, "backtest": out})

# ------------------------- Web UI -------------------------
HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Futures Pro V22 Pro+ — Balanced Trend</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
</head>
<body class="bg-slate-900 text-slate-100">
  <div class="max-w-6xl mx-auto p-4">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold">Futures Pro V22 Pro+ <span class="text-emerald-400">Balanced</span></h1>
      <a class="text-xs text-slate-400" href="/api/analyze?symbol=BTC-USD&tf=15m">API</a>
    </div>

    <div class="mt-4 grid grid-cols-1 md:grid-cols-4 gap-3">
      <div class="col-span-1 bg-slate-800/60 rounded-2xl p-3 shadow">
        <label class="text-sm font-semibold">نماد</label>
        <select id="symbol" class="w-full mt-1 bg-slate-900 rounded-xl p-2">
          <option>BTC-USD</option>
          <option>ETH-USD</option>
        </select>
        <label class="text-sm font-semibold mt-3 block">تایم‌فریم</label>
        <select id="tf" class="w-full mt-1 bg-slate-900 rounded-xl p-2">
          <option>15m</option>
          <option>1h</option>
          <option>4h</option>
          <option>1d</option>
        </select>
        <button id="run" class="mt-4 w-full bg-emerald-500 hover:bg-emerald-600 text-slate-900 font-bold rounded-xl py-2">تحلیل</button>

        <div id="summary" class="mt-4 text-sm space-y-2"></div>
      </div>

      <div class="col-span-1 md:col-span-3 bg-slate-800/60 rounded-2xl p-3 shadow">
        <div id="chart" class="w-full" style="height: 420px;"></div>
        <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div class="bg-slate-900/60 rounded-xl p-3" id="box-trend"></div>
          <div class="bg-slate-900/60 rounded-xl p-3" id="box-ai"></div>
          <div class="bg-slate-900/60 rounded-xl p-3" id="box-risk"></div>
          <div class="bg-slate-900/60 rounded-xl p-3" id="box-pos"></div>
        </div>
        <div class="mt-4">
          <table class="w-full text-xs">
            <thead class="text-slate-400">
              <tr>
                <th class="text-right p-2">Candle</th>
                <th class="text-right p-2">Price</th>
                <th class="text-right p-2">[P20..P80]</th>
                <th class="text-right p-2">TP</th>
                <th class="text-right p-2">SL</th>
                <th class="text-right p-2">Cof%</th>
                <th class="text-right p-2">AI%</th>
                <th class="text-right p-2">Score</th>
              </tr>
            </thead>
            <tbody id="forecast"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

<script>
let chart;

function fmt(x, d=2){ return (x===null||x===undefined)?"-":Number(x).toFixed(d); }
function colorByTrend(t){ return t==="Bullish"?"#22c55e":(t==="Bearish"?"#ef4444":"#94a3b8"); }

async function run(){
  const symbol = document.getElementById('symbol').value;
  const tf = document.getElementById('tf').value;
  const res = await fetch(`/api/analyze?symbol=${symbol}&tf=${tf}`);
  const js = await res.json();
  if(!js.ok){ alert(js.error||"خطا"); return; }
  const data = js.data, candles = js.chart;

  // summary cards
  const s = data.summary;
  document.getElementById('summary').innerHTML = `
    <div class="flex items-center justify-between">
      <div>برچسب: <span class="font-bold">${s.label}</span></div>
      <div>Bull%: <span class="font-bold text-emerald-400">${fmt(s.bull_pct)}</span></div>
    </div>
    <div class="grid grid-cols-2 gap-2 text-xs">
      <div>AI: ${fmt(s.ai_pct)}%</div>
      <div>PA: ${fmt(s.pa_pct)}%</div>
      <div>OF: ${fmt(s.of_pct)}%</div>
      <div>Trend: ${fmt(s.trend_pct)}%</div>
      <div>HTF: ${fmt(s.htf_pct)}%</div>
      <div>Risk: ${fmt(s.risk_pct)}%</div>
    </div>
  `;

  const sig = data.signal;
  document.getElementById('box-trend').innerHTML =
    `<div class="text-slate-400">Trend</div>
     <div class="text-lg font-bold" style="color:${colorByTrend(sig.trend)}">${sig.trend}</div>
     <div>score=${fmt(sig.score,3)} | cof=${fmt(sig.confidence)}%</div>`;

  document.getElementById('box-ai').innerHTML =
    `<div class="text-slate-400">AI Prob(Up)</div>
     <div class="text-lg font-bold">${fmt(sig.ai_prob_up)}%</div>
     <div>HTF=${sig.htf_bias}</div>`;

  document.getElementById('box-risk').innerHTML =
    `<div class="text-slate-400">Risk</div>
     <div>ATR=${fmt(sig.atr)}</div>
     <div>Regime=${fmt(sig.vol_regime,2)}</div>`;

  const pos = data.position;
  document.getElementById('box-pos').innerHTML =
    `<div class="text-slate-400">Position</div>
     <div>Units≈ <span class="font-bold">${fmt(pos.units,4)}</span></div>
     <div>Norm= ${fmt(pos.pos_norm*100,1)}%</div>`;

  // Forecast table
  const rows = data.forecast_rows;
  const tbody = document.getElementById('forecast');
  tbody.innerHTML = rows.map(r => `
    <tr class="border-t border-slate-700">
      <td class="p-2">${r.candle}</td>
      <td class="p-2">${fmt(r.price)}</td>
      <td class="p-2">${fmt(r.p_low)} .. ${fmt(r.p_high)}</td>
      <td class="p-2">${fmt(r.tp)}</td>
      <td class="p-2">${fmt(r.sl)}</td>
      <td class="p-2">${fmt(r.cof)}</td>
      <td class="p-2">${r.ai===null?"N/A":fmt(r.ai)}%</td>
      <td class="p-2">${fmt(r.score)}</td>
    </tr>
  `).join("");

  // Chart
  const series = candles.map(k => ({
    x: new Date(k.timestamp),
    y: [k.open, k.high, k.low, k.close].map(Number)
  }));
  const last = candles[candles.length-1];
  const tp = sig.tp, sl = sig.sl;

  const options = {
    series: [{ data: series }],
    chart: { type: 'candlestick', height: 420, background: 'transparent', foreColor:'#cbd5e1' },
    plotOptions:{ candlestick:{ colors:{ upward:'#22c55e', downward:'#ef4444'} } },
    xaxis: { type: 'datetime' },
    yaxis: { tooltip: { enabled: true } },
    annotations: {
      yaxis: [
        { y: tp, borderColor:'#22c55e', label:{ text:`TP ${fmt(tp)}` } },
        { y: sl, borderColor:'#ef4444', label:{ text:`SL ${fmt(sl)}` } }
      ]
    }
  };
  if(chart){ chart.destroy(); }
  chart = new ApexCharts(document.querySelector("#chart"), options);
  chart.render();
}

document.getElementById('run').addEventListener('click', run);
window.addEventListener('load', run);
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return Response(HTML, mimetype="text/html")

# ================= CLI entry (optional) =================
def main_once():
    print(f"Futures Pro V22 Pro+ (Balanced) starting...")
    print(f"Exchange: {EXCHANGE_NAME} | Testnet: {TESTNET} | Default TF: {DEFAULT_TF} | HTF:{HTF_TF} (use={USE_HTF}) | "
          f"AI={AI_ENABLED} | SoftFilters={SOFT_FILTERS} | Symbols: {SYMBOLS}")
    print(f"Δ$ clip multiplier (ATR): {DELTA_CLIP_ATR_MULT}  (raw + clip)")
    print(f"Confidence Weights → ML/Ind:{W_ML_IND}  PA:{W_PA}  OF:{W_OF}")
    print(f"Calibration={CALIBRATE_PROBS} | CV_Folds={CV_FOLDS} gap={CV_GAP} | Costs(bps)={COST_BPS}+Slip={SLIPPAGE_BPS} | Bands={USE_BANDS}")

    payloads = {}
    for sym in SYMBOLS:
        p, _, _ = analyze(sym, DEFAULT_TF)
        if p: payloads[sym] = p
    print(json.dumps(payloads, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    mode = "web" if ("--web" in set(a.lower() for a in sys.argv[1:])) or True else "cli"
    if mode == "cli":
        main_once()
    else:
        # Run web server
        port = int(os.environ.get("PORT", "7860"))
        app.run(host="0.0.0.0", port=port, debug=False)
