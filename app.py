"""
AI Stock Screener (Educational Trial) - Indian Markets (NSE/BSE)
------------------------------------------------------------------
Enhanced version with:
- Weighted scoring (not all factors equal)
- Supertrend indicator
- RSI divergence detection
- ATR volatility filter
- 52-week high breakout detection
- Out-of-sample backtesting
- Autocomplete search suggestions
EDUCATIONAL USE ONLY -- NOT FINANCIAL ADVICE.
"""
import re
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Brand palette
GREEN = "#22C55E"
AMBER = "#FBBF24"
RED   = "#EF4444"
BLUE  = "#3B82F6"

st.set_page_config(page_title="AI Stock Screener", page_icon="📊", layout="centered")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

NSE_STOCKS = {
    "RELIANCE":   "Reliance Industries",
    "TCS":        "Tata Consultancy Services",
    "INFY":       "Infosys",
    "WIPRO":      "Wipro",
    "HCLTECH":    "HCL Technologies",
    "TECHM":      "Tech Mahindra",
    "LTIM":       "LTIMindtree",
    "MPHASIS":    "Mphasis",
    "PERSISTENT": "Persistent Systems",
    "HDFCBANK":   "HDFC Bank",
    "ICICIBANK":  "ICICI Bank",
    "SBIN":       "State Bank of India",
    "KOTAKBANK":  "Kotak Mahindra Bank",
    "AXISBANK":   "Axis Bank",
    "INDUSINDBK": "IndusInd Bank",
    "PNB":        "Punjab National Bank",
    "BANKBARODA": "Bank of Baroda",
    "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",
    "HDFCLIFE":   "HDFC Life Insurance",
    "SBILIFE":    "SBI Life Insurance",
    "ICICIPRULI": "ICICI Prudential Life",
    "ICICIGI":    "ICICI Lombard General Insurance",
    "SBICARD":    "SBI Cards",
    "ONGC":       "Oil and Natural Gas Corp",
    "IOC":        "Indian Oil Corporation",
    "BPCL":       "Bharat Petroleum",
    "NTPC":       "NTPC Limited",
    "POWERGRID":  "Power Grid Corporation",
    "COALINDIA":  "Coal India",
    "ADANIGREEN": "Adani Green Energy",
    "ADANIPOWER": "Adani Power",
    "TATAPOWER":  "Tata Power",
    "ADANIENT":   "Adani Enterprises",
    "ADANIPORTS": "Adani Ports & SEZ",
    "MARUTI":     "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "M&M":        "Mahindra & Mahindra",
    "BAJAJ-AUTO": "Bajaj Auto",
    "HEROMOTOCO": "Hero MotoCorp",
    "EICHERMOT":  "Eicher Motors",
    "TVSMOTOR":   "TVS Motor Company",
    "HINDUNILVR": "Hindustan Unilever",
    "ITC":        "ITC Limited",
    "NESTLEIND":  "Nestle India",
    "BRITANNIA":  "Britannia Industries",
    "DABUR":      "Dabur India",
    "GODREJCP":   "Godrej Consumer Products",
    "TATACONSUM": "Tata Consumer Products",
    "MARICO":     "Marico Limited",
    "SUNPHARMA":  "Sun Pharmaceutical",
    "DRREDDY":    "Dr. Reddy's Laboratories",
    "CIPLA":      "Cipla",
    "DIVISLAB":   "Divi's Laboratories",
    "APOLLOHOSP": "Apollo Hospitals",
    "LUPIN":      "Lupin Limited",
    "BIOCON":     "Biocon",
    "TATASTEEL":  "Tata Steel",
    "JSWSTEEL":   "JSW Steel",
    "HINDALCO":   "Hindalco Industries",
    "VEDL":       "Vedanta Limited",
    "JINDALSTEL": "Jindal Steel & Power",
    "ULTRACEMCO": "UltraTech Cement",
    "SHREECEM":   "Shree Cement",
    "AMBUJACEM":  "Ambuja Cements",
    "ACC":        "ACC Limited",
    "BHARTIARTL": "Bharti Airtel",
    "IDEA":       "Vodafone Idea",
    "LT":         "Larsen & Toubro",
    "GRASIM":     "Grasim Industries",
    "DLF":        "DLF Limited",
    "TITAN":      "Titan Company",
    "ASIANPAINT": "Asian Paints",
    "HAVELLS":    "Havells India",
    "BEL":        "Bharat Electronics",
    "SIEMENS":    "Siemens India",
    "PIDILITIND": "Pidilite Industries",
    "ZOMATO":     "Eternal (Zomato)",
    "NYKAA":      "FSN E-Commerce (Nykaa)",
    "PAYTM":      "One97 Communications (Paytm)",
    "IRCTC":      "Indian Railway Catering & Tourism",
    "TRENT":      "Trent Limited",
}

QUICK_PICKS = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","ITC","TATAMOTORS","BAJFINANCE","BHARTIARTL"]

# ── Autocomplete helper ──────────────────────────────────────────────────────
def get_suggestions(query: str, max_results: int = 6) -> list:
    """Return matching (symbol, name) tuples for a partial query."""
    if not query or len(query) < 1:
        return []
    q = query.strip().upper()
    results = []
    for sym, name in NSE_STOCKS.items():
        if sym.startswith(q) or q in name.upper():
            results.append((sym, name))
    # Sort: exact symbol matches first, then name matches
    results.sort(key=lambda x: (0 if x[0].startswith(q) else 1, x[0]))
    return results[:max_results]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Setup")

def _secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

api_key = os.environ.get("GEMINI_API_KEY") or _secret_key()
if api_key:
    st.sidebar.success("✅ AI summaries enabled — nothing to set up!")
else:
    api_key = st.sidebar.text_input(
        "Gemini API key (free — get one at aistudio.google.com)",
        type="password",
    )
    st.sidebar.caption("Your key stays in this session only.")

st.sidebar.divider()
st.sidebar.caption("Educational, rule-based tool. Not financial advice.")

with st.sidebar.expander("📊 Indicator cheat sheet"):
    st.markdown("""
**200 SMA (structural gate — weight: 2)**
- Price above = safe zone · below = avoid all buys

**Supertrend (weight: 2)**
- Green / bullish = uptrend confirmed
- Red / bearish = downtrend, avoid buys

**EMA trend 20 vs 50 (weight: 1)**
- 20 EMA above 50 EMA → uptrend

**Fast EMA 9 vs 20 (weight: 1)**
- 9 EMA above 20 EMA → entry trigger

**RSI (14) + divergence (weight: 1 each)**
- >55 = momentum buy · <45 = weak
- Bearish divergence = price up but RSI down (warning)

**MACD crossover (weight: 1)**
- MACD above Signal → bullish

**Volume confirmation (weight: 1)**
- ≥1.5x avg = high conviction

**ATR volatility filter**
- Extreme ATR spike → signal muted (unreliable moves)

**52-week breakout (weight: 1)**
- Price breaking above 52w high with volume = powerful signal
""")

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period="1y")
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_fundamentals(ticker: str) -> dict:
    keys = [
        "longName","currentPrice","trailingPE","forwardPE",
        "priceToBook","returnOnEquity","debtToEquity",
        "dividendYield","marketCap","fiftyTwoWeekHigh","fiftyTwoWeekLow",
        "profitMargins","earningsGrowth","revenueGrowth",
        "currentRatio","quickRatio","freeCashflow",
    ]
    for attempt in range(2):
        try:
            info = yf.Ticker(ticker).info or {}
            return {k: info.get(k) for k in keys}
        except Exception:
            pass
        time.sleep(2)
    return {}

@st.cache_data(ttl=3600)
def fetch_extended_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

def resolve_ticker(raw: str):
    raw = raw.strip().upper().replace(".NS","").replace(".BO","")
    for suffix in [".NS", ".BO"]:
        symbol = raw + suffix
        df = fetch_price_history(symbol)
        if not df.empty:
            return symbol, df
    return None, pd.DataFrame()

def format_fundamentals(f: dict) -> pd.DataFrame:
    labels = {
        "trailingPE":"P/E Ratio (Trailing)","forwardPE":"P/E Ratio (Forward)",
        "priceToBook":"Price-to-Book","returnOnEquity":"Return on Equity",
        "debtToEquity":"Debt-to-Equity","dividendYield":"Dividend Yield",
        "fiftyTwoWeekHigh":"52-Week High (₹)","fiftyTwoWeekLow":"52-Week Low (₹)",
    }
    rows = []
    if f.get("marketCap"):
        rows.append(("Market Cap (₹ Cr)", round(f["marketCap"]/1e7,1)))
    for key, label in labels.items():
        val = f.get(key)
        if val is not None:
            rows.append((label, round(val,2) if isinstance(val,(int,float)) else val))
    return pd.DataFrame(rows, columns=["Metric","Value"])

# ── Quality gate ──────────────────────────────────────────────────────────────
def run_quality_gate(f: dict) -> list:
    checks = []
    roe = f.get("returnOnEquity")
    if roe is not None:
        checks.append(("ROE", f"{round(roe*100,1)}%", roe>=0.15, "≥15% good · <15% weak",
                        "Measures how efficiently the company uses shareholders' money."))
    dte = f.get("debtToEquity")
    if dte is not None:
        dte_ratio = dte/100 if dte>10 else dte
        checks.append(("Debt-to-Equity", f"{round(dte_ratio,2)}x", dte_ratio<=1.5,
                        "≤1.5x safe · >1.5x high debt","High debt = more interest burden."))
    pe = f.get("trailingPE")
    if pe is not None and pe>0:
        checks.append(("P/E Ratio", f"{round(pe,1)}x", pe<=40,
                        "≤40x reasonable · >40x expensive","High P/E = high expectations already priced in."))
    pm = f.get("profitMargins")
    if pm is not None:
        checks.append(("Profit Margin", f"{round(pm*100,1)}%", pm>=0.08,
                        "≥8% healthy · <8% thin","Thin margins = vulnerable to cost increases."))
    cr = f.get("currentRatio")
    if cr is not None:
        checks.append(("Current Ratio", f"{round(cr,2)}x", cr>=1.0,
                        "≥1x can cover debt · <1x liquidity risk","Can company pay short-term bills?"))
    eg = f.get("earningsGrowth")
    rg = f.get("revenueGrowth")
    growth = eg if eg is not None else rg
    label  = "Earnings Growth" if eg is not None else "Revenue Growth"
    if growth is not None:
        checks.append((label, f"{round(growth*100,1)}%", growth>0,
                        ">0% growing · ≤0% shrinking","Growing business compounds wealth."))
    return checks

# ── Indicators ────────────────────────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def compute_macd(close: pd.Series):
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Returns a boolean Series: True = bullish (price above supertrend line)."""
    atr    = compute_atr(df, period)
    hl2    = (df["High"] + df["Low"]) / 2
    upper  = hl2 + multiplier * atr
    lower  = hl2 - multiplier * atr
    close  = df["Close"]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)  # 1 = bullish, -1 = bearish

    for i in range(1, len(df)):
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]
        prev_close = close.iloc[i-1]
        prev_dir   = direction.iloc[i-1] if i > 1 else 1

        curr_upper = upper.iloc[i] if upper.iloc[i] < prev_upper or prev_close > prev_upper else prev_upper
        curr_lower = lower.iloc[i] if lower.iloc[i] > prev_lower or prev_close < prev_lower else prev_lower

        if prev_dir == -1 and close.iloc[i] > curr_upper:
            curr_dir = 1
        elif prev_dir == 1 and close.iloc[i] < curr_lower:
            curr_dir = -1
        else:
            curr_dir = prev_dir

        direction.iloc[i]  = curr_dir
        supertrend.iloc[i] = curr_lower if curr_dir == 1 else curr_upper

    direction.iloc[0] = 1
    return direction == 1   # True = bullish

def detect_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 14) -> str:
    """
    Detects bearish or bullish RSI divergence over the last `lookback` bars.
    Returns: 'bearish', 'bullish', or 'none'
    """
    if len(close) < lookback + 5:
        return "none"
    recent_close = close.iloc[-lookback:]
    recent_rsi   = rsi.iloc[-lookback:]

    price_high1 = recent_close.iloc[:lookback//2].max()
    price_high2 = recent_close.iloc[lookback//2:].max()
    rsi_high1   = recent_rsi.iloc[:lookback//2].max()
    rsi_high2   = recent_rsi.iloc[lookback//2:].max()

    price_low1  = recent_close.iloc[:lookback//2].min()
    price_low2  = recent_close.iloc[lookback//2:].min()
    rsi_low1    = recent_rsi.iloc[:lookback//2].min()
    rsi_low2    = recent_rsi.iloc[lookback//2:].min()

    # Bearish: price makes higher high but RSI makes lower high
    if price_high2 > price_high1 * 1.005 and rsi_high2 < rsi_high1 - 2:
        return "bearish"
    # Bullish: price makes lower low but RSI makes higher low
    if price_low2 < price_low1 * 0.995 and rsi_low2 > rsi_low1 + 2:
        return "bullish"
    return "none"

def compute_indicators(df: pd.DataFrame) -> dict:
    close      = df["Close"]
    volume     = df["Volume"]
    rsi        = compute_rsi(close)
    macd_line, sig_line = compute_macd(close)
    ema9       = close.ewm(span=9,  adjust=False).mean()
    ema20      = close.ewm(span=20, adjust=False).mean()
    ema50      = close.ewm(span=50, adjust=False).mean()
    sma200     = close.rolling(200).mean()
    vol_avg10  = volume.rolling(10).mean()
    atr        = compute_atr(df)
    atr_avg    = atr.rolling(20).mean()
    supertrend_bull = compute_supertrend(df)
    rsi_div    = detect_rsi_divergence(close, rsi)

    week52_high = df["High"].rolling(252).max()
    week52_low  = df["Low"].rolling(252).min()

    def _f(s): return round(float(s.iloc[-1]),2) if pd.notna(s.iloc[-1]) else None

    # ATR spike: current ATR > 2x its 20-period average → volatile, mute signals
    atr_spike = (
        atr.iloc[-1] > 2.0 * atr_avg.iloc[-1]
        if pd.notna(atr.iloc[-1]) and pd.notna(atr_avg.iloc[-1])
        else False
    )

    # 52w breakout: price within 1% of 52w high AND volume ≥ 1.5x avg
    vol_ratio_now = (volume.iloc[-1] / vol_avg10.iloc[-1]) if (
        pd.notna(vol_avg10.iloc[-1]) and vol_avg10.iloc[-1] > 0
    ) else 0
    near_52w_high = (
        week52_high.iloc[-1] is not None
        and pd.notna(week52_high.iloc[-1])
        and close.iloc[-1] >= week52_high.iloc[-1] * 0.99
    )
    breakout_52w = near_52w_high and vol_ratio_now >= 1.5

    return {
        "price":         round(float(close.iloc[-1]), 2),
        "rsi":           _f(rsi),
        "macd":          _f(macd_line),
        "macd_signal":   _f(sig_line),
        "ema9":          _f(ema9),
        "ema20":         _f(ema20),
        "ema50":         _f(ema50),
        "sma200":        _f(sma200),
        "volume":        int(volume.iloc[-1]),
        "vol_avg10":     int(vol_avg10.iloc[-1]) if pd.notna(vol_avg10.iloc[-1]) else None,
        "atr_spike":     atr_spike,
        "supertrend_bull": bool(supertrend_bull.iloc[-1]) if pd.notna(supertrend_bull.iloc[-1]) else None,
        "rsi_divergence":  rsi_div,
        "breakout_52w":    breakout_52w,
        "week52_high":     _f(week52_high),
        "week52_low":      _f(week52_low),
    }

# ── WEIGHTED Scoring ──────────────────────────────────────────────────────────
# Max possible score = +12, min = -12
# BUY ≥ 7, SELL ≤ -7, else HOLD
# Weights:
#   200 SMA structural gate : 2
#   Supertrend              : 2
#   EMA trend 20v50         : 1
#   Fast EMA 9v20           : 1
#   RSI momentum            : 1
#   RSI divergence          : 1  (bearish = -1, bullish = +1)
#   MACD crossover          : 1
#   Volume confirmation     : 1
#   52w breakout            : 1
#   ATR spike               : mutes score by 50% if triggered (rounded)
MAX_SCORE = 11

def compute_signal(ind: dict):
    breakdown = []
    score     = 0

    # ── 1. 200 SMA structural gate (weight 2) ───────────────────────────────
    above_200 = ind["sma200"] is not None and ind["price"] > ind["sma200"]
    if ind["sma200"] is not None:
        pts = 2 if above_200 else -2
        score += pts
        breakdown.append((
            "200 SMA (structural gate)", f"₹{ind['price']} vs ₹{ind['sma200']}",
            "🟢 Bullish" if above_200 else "🔴 Bearish",
            "Price above 200 SMA — long-term uptrend intact" if above_200
            else "Price below 200 SMA — structurally compromised",
            f"Weight: 2 · Points: {pts:+d}",
        ))

    # ── 2. Supertrend (weight 2) ─────────────────────────────────────────────
    if ind["supertrend_bull"] is not None:
        bull = ind["supertrend_bull"]
        pts  = 2 if bull else -2
        score += pts
        breakdown.append((
            "Supertrend (10,3)", "Bullish" if bull else "Bearish",
            "🟢 Bullish" if bull else "🔴 Bearish",
            "Supertrend green — trend confirmed bullish" if bull
            else "Supertrend red — trend confirmed bearish",
            f"Weight: 2 · Points: {pts:+d}",
        ))

    # ── 3. EMA trend 20 vs 50 (weight 1) ────────────────────────────────────
    if ind["ema20"] is not None and ind["ema50"] is not None:
        bull = ind["ema20"] > ind["ema50"]
        pts  = 1 if bull else -1
        score += pts
        breakdown.append((
            "EMA trend (20 vs 50)", f"₹{ind['ema20']} vs ₹{ind['ema50']}",
            "🟢 Bullish" if bull else "🔴 Bearish",
            "20 EMA above 50 EMA — short-term momentum upward" if bull
            else "20 EMA below 50 EMA — short-term momentum downward",
            f"Weight: 1 · Points: {pts:+d}",
        ))

    # ── 4. Fast EMA 9 vs 20 (weight 1) ──────────────────────────────────────
    if ind["ema9"] is not None and ind["ema20"] is not None:
        bull = ind["ema9"] > ind["ema20"]
        pts  = 1 if bull else -1
        score += pts
        breakdown.append((
            "Fast EMA (9 vs 20)", f"₹{ind['ema9']} vs ₹{ind['ema20']}",
            "🟢 Bullish" if bull else "🔴 Bearish",
            "9 EMA above 20 EMA — quick momentum burst" if bull
            else "9 EMA below 20 EMA — momentum fading",
            f"Weight: 1 · Points: {pts:+d}",
        ))

    # ── 5. RSI momentum (weight 1) ───────────────────────────────────────────
    if ind["rsi"] is not None:
        if ind["rsi"] > 55:
            score += 1
            breakdown.append(("RSI (14)", ind["rsi"], "🟢 Bullish",
                               "Above 55 — momentum building", "Weight: 1 · Points: +1"))
        elif ind["rsi"] < 45:
            score -= 1
            breakdown.append(("RSI (14)", ind["rsi"], "🔴 Bearish",
                               "Below 45 — momentum weakening", "Weight: 1 · Points: -1"))
        else:
            breakdown.append(("RSI (14)", ind["rsi"], "⚪ Neutral",
                               "45–55 — no strong momentum signal", "Weight: 1 · Points: 0"))

    # ── 6. RSI Divergence (weight 1) ─────────────────────────────────────────
    div = ind.get("rsi_divergence", "none")
    if div == "bearish":
        score -= 1
        breakdown.append(("RSI Divergence", "Bearish divergence", "🔴 Warning",
                           "Price making higher highs but RSI is not — reversal risk",
                           "Weight: 1 · Points: -1"))
    elif div == "bullish":
        score += 1
        breakdown.append(("RSI Divergence", "Bullish divergence", "🟢 Confirmation",
                           "Price making lower lows but RSI is not — reversal to upside possible",
                           "Weight: 1 · Points: +1"))

    # ── 7. MACD crossover (weight 1) ─────────────────────────────────────────
    if ind["macd"] is not None and ind["macd_signal"] is not None:
        bull = ind["macd"] > ind["macd_signal"]
        pts  = 1 if bull else -1
        score += pts
        breakdown.append((
            "MACD crossover", f"{ind['macd']} vs {ind['macd_signal']}",
            "🟢 Bullish" if bull else "🔴 Bearish",
            "MACD above signal line — bullish crossover" if bull
            else "MACD below signal line — bearish crossover",
            f"Weight: 1 · Points: {pts:+d}",
        ))

    # ── 8. Volume confirmation (weight 1) ────────────────────────────────────
    if ind["vol_avg10"] is not None and ind["vol_avg10"] > 0:
        vol_ratio = ind["volume"] / ind["vol_avg10"]
        if vol_ratio >= 1.5:
            score += 1
            breakdown.append(("Volume", f"{vol_ratio:.1f}x avg", "🟢 Bullish",
                               f"Volume {vol_ratio:.1f}x the 10-day avg — strong conviction",
                               "Weight: 1 · Points: +1"))
        elif vol_ratio < 0.8:
            score -= 1
            breakdown.append(("Volume", f"{vol_ratio:.1f}x avg", "🔴 Bearish",
                               "Volume well below average — move lacks conviction",
                               "Weight: 1 · Points: -1"))
        else:
            breakdown.append(("Volume", f"{vol_ratio:.1f}x avg", "⚪ Neutral",
                               "Volume near average — no strong confirmation",
                               "Weight: 1 · Points: 0"))

    # ── 9. 52-week breakout (weight 1) ───────────────────────────────────────
    if ind.get("breakout_52w"):
        score += 1
        breakdown.append(("52-Week Breakout", f"₹{ind['price']} near ₹{ind['week52_high']}",
                           "🟢 Bullish",
                           "Price at 52-week high with strong volume — powerful breakout signal",
                           "Weight: 1 · Points: +1"))

    # ── ATR spike muting ──────────────────────────────────────────────────────
    atr_muted = False
    if ind.get("atr_spike"):
        original_score = score
        score = int(round(score * 0.5))
        atr_muted = True
        breakdown.append(("ATR Volatility Filter", "⚠️ Spike detected", "🟡 Muted",
                           f"Extreme volatility detected — score muted from {original_score} to {score}. "
                           "Moves during high ATR spikes are unreliable.",
                           "Weight: modifier · Score halved"))

    # ── Final signal ──────────────────────────────────────────────────────────
    if score >= 7 and above_200:
        signal = "BUY"
    elif score <= -7:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, breakdown, atr_muted

# ── Vectorised score for backtest ─────────────────────────────────────────────
def compute_score_series(df: pd.DataFrame):
    close      = df["Close"]
    volume     = df["Volume"]
    rsi        = compute_rsi(close)
    macd_line, sig_line = compute_macd(close)
    ema9       = close.ewm(span=9,  adjust=False).mean()
    ema20      = close.ewm(span=20, adjust=False).mean()
    ema50      = close.ewm(span=50, adjust=False).mean()
    sma200     = close.rolling(200).mean()
    vol_avg10  = volume.rolling(10).mean()
    atr        = compute_atr(df)
    atr_avg    = atr.rolling(20).mean()
    st_bull    = compute_supertrend(df)

    above_200 = close > sma200

    # Weighted scores
    sma200_pts = np.where(close > sma200, 2, -2)
    sma200_pts = np.where(sma200.isna(), 0, sma200_pts)

    st_pts = np.where(st_bull, 2, -2)

    ema_trend_pts = np.where(ema20 > ema50, 1, -1)
    ema_trend_pts = np.where(ema20.isna() | ema50.isna(), 0, ema_trend_pts)

    fast_ema_pts = np.where(ema9 > ema20, 1, -1)
    fast_ema_pts = np.where(ema9.isna() | ema20.isna(), 0, fast_ema_pts)

    rsi_pts = np.select([rsi > 55, rsi < 45], [1, -1], default=0)
    rsi_pts = np.where(rsi.isna(), 0, rsi_pts)

    macd_pts = np.where(macd_line > sig_line, 1, -1)
    macd_pts = np.where(macd_line.isna() | sig_line.isna(), 0, macd_pts)

    vol_ratio = volume / vol_avg10.replace(0, np.nan)
    vol_pts   = np.select([vol_ratio >= 1.5, vol_ratio < 0.8], [1, -1], default=0)
    vol_pts   = np.where(vol_avg10.isna(), 0, vol_pts)

    # 52w breakout
    week52_high = df["High"].rolling(252).max()
    near_52w = close >= week52_high * 0.99
    breakout_pts = np.where(near_52w & (vol_ratio >= 1.5), 1, 0)

    score = pd.Series(
        sma200_pts + st_pts + ema_trend_pts + fast_ema_pts +
        rsi_pts + macd_pts + vol_pts + breakout_pts,
        index=df.index,
    )

    # ATR muting
    atr_spike = atr > 2.0 * atr_avg
    score = score.where(~atr_spike, (score * 0.5).round().astype(int))

    signal = pd.Series(
        np.select(
            [(score >= 7) & above_200, score <= -7],
            ["BUY", "SELL"],
            default="HOLD",
        ),
        index=df.index,
    )
    return score, signal

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, holding_days=(5,20,60), oos_split: float = 0.6):
    """
    Out-of-sample backtest:
    - First `oos_split` fraction = in-sample (used to understand the strategy)
    - Last `1-oos_split` fraction = out-of-sample (true test, never seen by strategy design)
    """
    df     = df.copy()
    warmup = 200
    if len(df) <= warmup + max(holding_days) + 50:
        return None, None, None, None, None

    score, signal = compute_score_series(df)
    close         = df["Close"]
    daily_return  = close.pct_change()
    acted_signal  = signal.shift(1)

    total_len  = len(df) - warmup
    split_idx  = warmup + int(total_len * oos_split)

    # Full period equity curves
    strategy_return = daily_return.where(acted_signal == "BUY", 0.0)
    strat_window    = strategy_return.iloc[warmup:].fillna(0)
    hold_window     = daily_return.iloc[warmup:].fillna(0)
    cum_strategy    = (1 + strat_window).cumprod()
    cum_buyhold     = (1 + hold_window).cumprod()

    # Out-of-sample only
    oos_strat   = strategy_return.iloc[split_idx:].fillna(0)
    oos_hold    = daily_return.iloc[split_idx:].fillna(0)
    cum_oos_strat = (1 + oos_strat).cumprod()
    cum_oos_hold  = (1 + oos_hold).cumprod()

    # Summary table
    bt = pd.DataFrame({"Signal": signal, "DailyReturn": daily_return}, index=df.index).iloc[warmup:]
    for h in holding_days:
        bt[f"fwd_{h}d"] = close.shift(-h) / close - 1

    table_rows = []
    for sig in ["BUY","HOLD","SELL"]:
        subset = bt[bt["Signal"] == sig]
        table_rows.append({"Signal": sig, "Occurrences": len(subset)})
    summary_df = pd.DataFrame(table_rows)

    for h, label in zip(holding_days, ["~1 week","~1 month","~3 months"]):
        avg_returns, win_rates = [], []
        for sig in ["BUY","HOLD","SELL"]:
            valid = bt[bt["Signal"] == sig][f"fwd_{h}d"].dropna()
            avg_returns.append(round(valid.mean()*100, 2) if len(valid) else None)
            win_rates.append(round((valid > 0).mean()*100, 1) if len(valid) else None)
        summary_df[f"Avg return ({label})"]  = avg_returns
        summary_df[f"Win rate % ({label})"]  = win_rates

    oos_split_date = df.index[split_idx].strftime("%b %Y") if split_idx < len(df) else "—"

    return summary_df, cum_strategy, cum_buyhold, cum_oos_strat, oos_split_date

def run_hybrid_backtest(df: pd.DataFrame):
    df     = df.copy()
    warmup = 200
    if len(df) <= warmup + 60:
        return None, None, None, None

    score, _ = compute_score_series(df)
    close    = df["Close"]
    daily_return = close.pct_change()

    above_200_series = close > close.rolling(200).mean()

    in_position    = False
    position_flags = []
    for i, idx in enumerate(df.index):
        s         = score.loc[idx]
        above_200 = above_200_series.loc[idx]
        if not in_position:
            if s >= 7 and above_200:
                in_position = True
        else:
            if s <= -7:
                in_position = False
        position_flags.append(in_position)

    position       = pd.Series(position_flags, index=df.index)
    acted_position = position.shift(1).fillna(False)
    hybrid_return  = daily_return.where(acted_position, 0.0)

    strat_window = hybrid_return.iloc[warmup:].fillna(0)
    hold_window  = daily_return.iloc[warmup:].fillna(0)
    cum_hybrid   = (1 + strat_window).cumprod()
    cum_buyhold  = (1 + hold_window).cumprod()

    entries  = position.diff().fillna(0)
    n_trades = int((entries == 1).sum())
    avg_hold = int(position.sum() / max(n_trades, 1))

    return cum_hybrid, cum_buyhold, n_trades, avg_hold

def make_backtest_chart(cum_strategy, cum_buyhold, cum_hybrid=None, cum_oos=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_strategy.index, y=cum_strategy,
                             name="Original strategy", line=dict(color=AMBER, width=2)))
    fig.add_trace(go.Scatter(x=cum_buyhold.index, y=cum_buyhold,
                             name="Buy & hold", line=dict(color="#9CA3AF", width=2, dash="dot")))
    if cum_hybrid is not None:
        fig.add_trace(go.Scatter(x=cum_hybrid.index, y=cum_hybrid,
                                 name="Hybrid strategy", line=dict(color=GREEN, width=2.5)))
    if cum_oos is not None:
        fig.add_trace(go.Scatter(x=cum_oos.index, y=cum_oos,
                                 name="Out-of-sample only", line=dict(color=BLUE, width=2, dash="dash")))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Growth of ₹1"),
        hovermode="x unified",
        font=dict(color="#E5E7EB"),
    )
    return fig

def make_price_chart(df: pd.DataFrame) -> go.Figure:
    close   = df["Close"]
    sma200  = close.rolling(200).mean()
    ema20   = close.ewm(span=20, adjust=False).mean()
    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std

    # Supertrend line
    st_bull = compute_supertrend(df)

    week52_high = df["High"].max()
    week52_low  = df["Low"].min()
    vol_colors  = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72,0.28], vertical_spacing=0.04)

    # Bollinger band shading
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(99,102,241,0.15)",
                             name="Bollinger Band", hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=close, name="Price",
                             line=dict(color=GREEN, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="20 EMA",
                             line=dict(color=AMBER, width=1.3, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, name="200 SMA",
                             line=dict(color=RED, width=1.3, dash="dot")), row=1, col=1)

    # Supertrend coloured dots
    st_line = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        # place supertrend marker below/above price
        if st_bull.iloc[i]:
            st_line.iloc[i] = close.iloc[i] * 0.97
        else:
            st_line.iloc[i] = close.iloc[i] * 1.03

    fig.add_trace(go.Scatter(
        x=df.index[st_bull], y=st_line[st_bull],
        mode="markers", marker=dict(color=GREEN, size=3),
        name="Supertrend ▲", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index[~st_bull], y=st_line[~st_bull],
        mode="markers", marker=dict(color=RED, size=3),
        name="Supertrend ▼", hoverinfo="skip"), row=1, col=1)

    fig.add_hline(y=week52_high, line=dict(color="rgba(229,231,235,0.4)", dash="dash", width=1),
                  annotation_text="52w High", annotation_font_size=10,
                  annotation_position="top left", row=1, col=1)
    fig.add_hline(y=week52_low, line=dict(color="rgba(229,231,235,0.4)", dash="dash", width=1),
                  annotation_text="52w Low", annotation_font_size=10,
                  annotation_position="bottom left", row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        height=440, margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", font=dict(color="#E5E7EB"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
    fig.update_yaxes(showgrid=False, title_text="Volume", title_font_size=10, row=2, col=1)
    return fig

def make_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 34}},
        gauge={
            "axis": {"range": [-MAX_SCORE, MAX_SCORE], "tickwidth": 1},
            "bar": {"color": "#E5E7EB", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-MAX_SCORE, -7], "color": RED},
                {"range": [-7, 7], "color": AMBER},
                {"range": [7, MAX_SCORE], "color": GREEN},
            ],
        },
    ))
    fig.update_layout(
        height=170, margin=dict(l=20,r=20,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color":"#E5E7EB"},
    )
    return fig

def scan_stocks(symbols):
    results, errors = [], []
    total    = len(symbols)
    progress = st.progress(0, text="Starting scan...")
    for i, sym in enumerate(symbols):
        progress.progress((i+1)/total, text=f"Scanning {sym} ({i+1}/{total})...")
        resolved, df = resolve_ticker(sym)
        if df.empty:
            errors.append(sym)
            time.sleep(0.5)
            continue
        time.sleep(0.5)
        indicators     = compute_indicators(df)
        signal, score, _, _ = compute_signal(indicators)
        results.append({
            "Symbol": sym, "Name": NSE_STOCKS.get(sym, sym),
            "Price": indicators["price"], "Score": score, "Signal": signal,
        })
    progress.empty()
    return results, errors

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return df.resample(rule).agg(agg).dropna()

def multi_timeframe_view(daily_df, extended_df):
    weekly_df  = resample_ohlc(extended_df, "W")
    monthly_df = resample_ohlc(extended_df, "ME")
    rows = []
    for label, tf_df, min_bars in [("Daily",daily_df,60),("Weekly",weekly_df,60),("Monthly",monthly_df,24)]:
        if tf_df is None or len(tf_df) < min_bars:
            rows.append({"Timeframe":label,"Signal":"Not enough history","Score":"—","RSI":"—","Price":"—"})
            continue
        ind = compute_indicators(tf_df)
        sig, score, _, _ = compute_signal(ind)
        rows.append({"Timeframe":label,"Signal":sig,"Score":score,"RSI":ind["rsi"],"Price":ind["price"]})
    return pd.DataFrame(rows)

# ── AI narrative ──────────────────────────────────────────────────────────────
def generate_narrative(symbol, fundamentals, breakdown, signal, api_key) -> str:
    reasons = "; ".join(f"{row[0]}: {row[3]}" for row in breakdown)
    prompt  = f"""
You are an educational stock-analysis assistant. Based ONLY on the data below,
write a short, plain-English analysis of {symbol} for a beginner investor.

Rule-based signal: {signal}
Reasons: {reasons}
Fundamentals: {fundamentals}

Structure:
1. One-sentence summary
2. 3 bullet points: what looks GOOD
3. 3 bullet points: what looks CONCERNING
4. One closing line: remind the reader this is educational, not financial advice.

Keep it under 200 words.
"""
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"(AI narrative unavailable: {e})"

# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
st.title("📊 AI Stock Screener")
st.caption("Enhanced · NSE/BSE · Weighted scoring · Supertrend · RSI divergence · ATR filter · 52w breakout · Gemini AI")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze a Stock", "📋 Screen a List", "📈 Backtest"])

# ═══════════════════════ TAB 1 ═══════════════════════════════════════════════
with tab1:

    # ── Autocomplete search ──────────────────────────────────────────────────
    st.markdown("#### Search Stock")

    # Use session state so suggestion clicks populate the input
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""
    if "selected_symbol" not in st.session_state:
        st.session_state["selected_symbol"] = ""

    search_input = st.text_input(
        "Type NSE symbol or company name",
        value=st.session_state["search_query"],
        placeholder="e.g.  RELIANCE  or  Tata Consul...",
        key="tab1_search",
    )

    # Show live suggestions as the user types
    suggestions = get_suggestions(search_input) if search_input else []

    if suggestions and search_input:
        st.markdown("**Suggestions — tap to select:**")
        # Show suggestions as clickable buttons in a row
        cols = st.columns(min(len(suggestions), 3))
        for idx, (sym, name) in enumerate(suggestions):
            with cols[idx % 3]:
                label = f"{sym}\n{name[:20]}{'…' if len(name)>20 else ''}"
                if st.button(label, key=f"sugg_{sym}", use_container_width=True):
                    st.session_state["search_query"]   = sym
                    st.session_state["selected_symbol"] = sym
                    st.rerun()

    # Resolve final symbol: either from suggestion click or direct input
    final_symbol = st.session_state.get("selected_symbol") or search_input.strip().upper()

    analyze_clicked = st.button("Analyze 📊", type="primary", use_container_width=True)

    # Clear selected symbol after analyze so next search starts fresh
    if analyze_clicked:
        st.session_state["selected_symbol"] = ""

    raw_input = final_symbol if analyze_clicked and final_symbol else None

    if raw_input:
        with st.spinner("Fetching data..."):
            symbol, df = resolve_ticker(raw_input)

        if df.empty:
            st.error(
                "Couldn't fetch data. Either the symbol is wrong, or the data provider "
                "is briefly rate-limiting this shared server — wait a minute and try again."
            )
        else:
            try:
                fundamentals       = fetch_fundamentals(symbol)
                indicators         = compute_indicators(df)
                signal, score, breakdown, atr_muted = compute_signal(indicators)

                with st.container(border=True):
                    st.subheader(f"{fundamentals.get('longName', symbol)} ({symbol})")

                    if signal == "BUY":
                        st.success(f"🟢 **BUY** — weighted score {score} / {MAX_SCORE}")
                    elif signal == "SELL":
                        st.error(f"🔴 **SELL** — weighted score {score} / {MAX_SCORE}")
                    else:
                        st.warning(f"🟡 **HOLD** — weighted score {score} / {MAX_SCORE}")

                    if atr_muted:
                        st.warning("⚠️ **ATR volatility spike detected** — score was muted by 50%. "
                                   "The market is unusually volatile; treat signals with extra caution.")

                    if indicators.get("rsi_divergence") == "bearish":
                        st.warning("📉 **Bearish RSI divergence** — price making new highs but momentum is weakening. Watch out.")
                    elif indicators.get("rsi_divergence") == "bullish":
                        st.info("📈 **Bullish RSI divergence** — price falling but momentum improving. Possible reversal.")

                    if indicators.get("breakout_52w"):
                        st.success("🚀 **52-Week Breakout!** Price at yearly high with strong volume — powerful signal.")

                    st.plotly_chart(make_gauge(score), use_container_width=True,
                                    config={"displayModeBar": False})

                    # Quality gate
                    if fundamentals:
                        qchecks = run_quality_gate(fundamentals)
                        if qchecks:
                            passed  = sum(1 for c in qchecks if c[2])
                            total_q = len(qchecks)
                            if passed >= 5:
                                st.success(f"✅ Quality grade: {passed}/{total_q} — fundamentally strong")
                            elif passed >= 3:
                                st.warning(f"⚠️ Quality grade: {passed}/{total_q} — average quality")
                            else:
                                st.error(f"❌ Quality grade: {passed}/{total_q} — weak fundamentals")

                            if signal == "BUY" and passed >= 5:
                                st.info("💡 **High-conviction setup:** Technical signal AND fundamentals agree.")
                            elif signal == "BUY" and passed < 3:
                                st.warning("⚠️ **Caution:** Technical signal says BUY but fundamentals are weak.")
                            elif signal == "HOLD" and passed >= 5:
                                st.info("💡 **Quality stock waiting for signal:** Strong business, not yet the right technical entry.")

                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Price", f"₹{indicators['price']}")
                        st.metric("RSI (14)", indicators["rsi"])
                        st.caption("🟢 >55 buy · ⚪ 45–55 neutral · 🔴 <45 weak")
                    with col2:
                        st.metric("MACD", indicators["macd"])
                        st.metric("Signal Line", indicators["macd_signal"])
                        sup_label = "🟢 Bullish" if indicators.get("supertrend_bull") else "🔴 Bearish"
                        st.metric("Supertrend", sup_label)

                    st.plotly_chart(make_price_chart(df), use_container_width=True,
                                    config={"displayModeBar": False})
                    st.caption(
                        "🟣 Shaded band = Bollinger Bands · dashed lines = 52w high/low · "
                        "dots = Supertrend (🟢 bullish / 🔴 bearish) · bottom bars = daily volume"
                    )

                    with st.expander("📐 Why this rating? — full score breakdown"):
                        st.dataframe(
                            pd.DataFrame(breakdown,
                                         columns=["Indicator","Value","Signal","Why","Weight/Points"]),
                            hide_index=True, use_container_width=True,
                        )
                        st.caption(
                            f"Score range: -{MAX_SCORE} to +{MAX_SCORE} (weighted). "
                            "BUY ≥ 7 AND above 200 SMA · SELL ≤ -7 · else HOLD."
                        )

                    with st.expander("💰 Fundamentals"):
                        if fundamentals:
                            st.dataframe(format_fundamentals(fundamentals),
                                         hide_index=True, use_container_width=True)
                        else:
                            st.caption("Fundamentals temporarily unavailable.")

                    with st.expander("🏆 Quality Gate — full checklist"):
                        if fundamentals:
                            qchecks = run_quality_gate(fundamentals)
                            for name, val, passed, range_hint, explanation in qchecks:
                                icon = "✅" if passed else "❌"
                                col_a, _ = st.columns([3,1])
                                col_a.markdown(f"**{icon} {name}**: {val}")
                                col_a.caption(f"{explanation}\n*Range: {range_hint}*")
                            st.caption("5–6 passed = quality stock ✅ · 3–4 = average ⚠️ · 0–2 = avoid ❌")

                    with st.expander("📅 Daily vs Weekly vs Monthly view"):
                        if st.button("Load multi-timeframe view", key="load_mtf"):
                            with st.spinner("Fetching extended history..."):
                                ext_df = fetch_extended_history(symbol)
                            if ext_df.empty:
                                st.error("Couldn't fetch extended history — try again shortly.")
                            else:
                                mtf = multi_timeframe_view(df, ext_df)
                                st.dataframe(mtf, hide_index=True, use_container_width=True)

                    st.subheader("🤖 AI Summary")
                    if api_key:
                        with st.spinner("Generating AI narrative..."):
                            narrative = generate_narrative(symbol, fundamentals, breakdown, signal, api_key)
                        st.write(narrative)
                    else:
                        st.info("Enter a free Gemini API key in the sidebar to get an AI-written plain-English summary.")

            except Exception as e:
                st.error(f"Something went wrong while analyzing: {e}")
                st.caption("This is usually temporary — try again in a minute.")

    else:
        st.divider()
        st.markdown("#### Quick picks")
        cols = st.columns(5)
        for i, sym in enumerate(QUICK_PICKS):
            with cols[i % 5]:
                if st.button(sym, key=f"qp_{sym}", use_container_width=True):
                    st.session_state["search_query"]   = sym
                    st.session_state["selected_symbol"] = sym
                    st.rerun()
        st.caption("Tap any quick pick above, or type in the search box.")

# ═══════════════════════ TAB 2 ═══════════════════════════════════════════════
with tab2:
    st.subheader("📋 Screen multiple stocks at once")
    st.caption(
        "Runs the weighted RSI/MACD/Supertrend/trend score across a list of stocks. "
        "No AI calls here — fast and won't touch your Gemini quota."
    )

    chosen = st.multiselect(
        "Choose stocks to screen",
        options=sorted(NSE_STOCKS.keys()),
        default=QUICK_PICKS,
        format_func=lambda s: f"{s} — {NSE_STOCKS[s]}",
    )

    if st.button("Scan list 🔍", use_container_width=True) and chosen:
        results, errors = scan_stocks(chosen)
        if results:
            df_results = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
            n_bull = (df_results["Signal"] == "BUY").sum()
            n_hold = (df_results["Signal"] == "HOLD").sum()
            n_bear = (df_results["Signal"] == "SELL").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 Bullish", n_bull)
            m2.metric("🟡 Neutral", n_hold)
            m3.metric("🔴 Bearish", n_bear)

            display_df = df_results[["Symbol","Name","Price","Score","Signal"]]
            try:
                styled = (
                    display_df.style
                    .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-8, vmax=8)
                    .format({"Price": "₹{:.2f}"})
                )
                st.dataframe(styled, hide_index=True, use_container_width=True)
            except Exception:
                st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption(f"Sorted most bullish → most bearish. Max score now ±{MAX_SCORE} (weighted).")
        if errors:
            st.caption(f"Couldn't fetch: {', '.join(errors)}")

# ═══════════════════════ TAB 3 ═══════════════════════════════════════════════
with tab3:
    st.subheader("📈 Backtest the signal")
    st.caption(
        "Weighted 9-factor strategy (200 SMA · Supertrend · EMA alignment · Fast EMA · "
        "RSI · RSI divergence · MACD · Volume · 52w breakout) vs ~5 years of history. "
        "BUY requires score ≥7 AND price above 200 SMA."
    )
    st.warning(
        "⚠️ No transaction costs, taxes, or slippage modeled. This is for learning "
        "how backtesting works — not a trading system.",
        icon="⚠️",
    )
    st.info(
        "🔬 **Out-of-sample split:** First 60% of history = in-sample · "
        "Last 40% = true out-of-sample test (the blue dashed line).",
        icon="🔬",
    )

    bt_search = st.text_input(
        "Enter NSE/BSE symbol to backtest",
        placeholder="e.g. RELIANCE, TCS, HDFCBANK",
        key="bt_search",
    )
    run_bt = st.button("Run backtest 📊", type="primary", use_container_width=True)

    if run_bt:
        bt_raw = bt_search.strip() if bt_search.strip() else None
        if not bt_raw:
            st.error("Enter an NSE/BSE symbol first.")
        else:
            with st.spinner("Fetching ~5 years of history and running the backtest..."):
                bt_symbol, _ = resolve_ticker(bt_raw)
                ext_df = fetch_extended_history(bt_symbol) if bt_symbol else pd.DataFrame()

            if ext_df.empty:
                st.error("Couldn't fetch enough history — try again shortly.")
            else:
                result = run_backtest(ext_df)
                if result[0] is None:
                    st.error("Not enough history available for a meaningful backtest.")
                else:
                    summary_df, cum_strategy, cum_buyhold, cum_oos_strat, oos_split_date = result
                    hybrid_result = run_hybrid_backtest(ext_df)

                    cum_hybrid = None
                    n_trades   = 0
                    avg_hold   = 0
                    if hybrid_result[0] is not None:
                        cum_hybrid, _, n_trades, avg_hold = hybrid_result

                    st.markdown(f"#### Results for {bt_symbol}")
                    st.caption(f"Out-of-sample period starts: **{oos_split_date}** (blue dashed line = OOS performance)")

                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                    st.caption(
                        "'Occurrences' = days in backtest with that signal. "
                        "'Avg return' = avg price change over the holding period. "
                        "'Win rate' = % of time the return was positive."
                    )

                    total_strategy_return = (cum_strategy.iloc[-1] - 1) * 100
                    total_buyhold_return  = (cum_buyhold.iloc[-1]  - 1) * 100
                    total_hybrid_return   = (cum_hybrid.iloc[-1]   - 1) * 100 if cum_hybrid is not None else None
                    total_oos_return      = (cum_oos_strat.iloc[-1] - 1) * 100 if cum_oos_strat is not None else None

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Original strategy", f"{total_strategy_return:.1f}%")
                    c2.metric("Buy & hold",         f"{total_buyhold_return:.1f}%")
                    if total_hybrid_return is not None:
                        c3.metric("🆕 Hybrid strategy", f"{total_hybrid_return:.1f}%")
                    if total_oos_return is not None:
                        c4.metric("🔬 Out-of-sample", f"{total_oos_return:.1f}%",
                                  help=f"Strategy return on unseen data from {oos_split_date}")

                    st.plotly_chart(
                        make_backtest_chart(cum_strategy, cum_buyhold, cum_hybrid, cum_oos_strat),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                    st.caption(
                        "🟡 Original = exits on every signal change · "
                        "⬜ Buy & hold = never sells · "
                        "🟢 Hybrid = enters on BUY, only exits on hard breakdown · "
                        "🔵 Out-of-sample = strategy on data it never 'saw'"
                    )

                    if total_hybrid_return is not None:
                        st.info(
                            f"**Hybrid stats:** {n_trades} trades, "
                            f"avg hold ~{avg_hold} trading days (~{round(avg_hold/21)} months)"
                        )

                    # Plain-English verdict
                    st.subheader("🧠 What do these results mean?")
                    buy_rows   = summary_df[summary_df["Signal"] == "BUY"]
                    buy_wr_1m  = buy_rows["Win rate % (~1 month)"].values[0] if len(buy_rows) else None
                    buy_ret_1m = buy_rows["Avg return (~1 month)"].values[0] if len(buy_rows) else None
                    buy_occ    = buy_rows["Occurrences"].values[0] if len(buy_rows) else None

                    verdict_lines = []
                    if buy_occ is not None:
                        verdict_lines.append(
                            f"**Signal frequency:** BUY signal on **{buy_occ}** "
                            f"out of ~{len(ext_df)} days tested "
                            f"(~{round(buy_occ/max(len(ext_df),1)*100,1)}% of days). "
                            "Fewer signals = more selective."
                        )
                    if buy_wr_1m is not None:
                        if buy_wr_1m >= 60:
                            verdict_lines.append(f"**1-month win rate: {buy_wr_1m}% ✅** — good historical hit rate.")
                        elif buy_wr_1m >= 50:
                            verdict_lines.append(f"**1-month win rate: {buy_wr_1m}% ⚠️** — just above 50%, not strong alone.")
                        else:
                            verdict_lines.append(f"**1-month win rate: {buy_wr_1m}% ❌** — BUY signals preceded losses more often.")
                    if buy_ret_1m is not None:
                        sign = "positive" if buy_ret_1m > 0 else "negative"
                        verdict_lines.append(
                            f"**Avg 1-month return after BUY: {buy_ret_1m}%** — "
                            f"on average, price moved {abs(buy_ret_1m)}% in a {sign} direction."
                        )
                    if total_oos_return is not None:
                        oos_bh_return = (cum_oos_strat.index[0], )  # placeholder
                        verdict_lines.append(
                            f"**🔬 Out-of-sample return: {total_oos_return:.1f}%** — "
                            "this is the most honest number: performance on data the strategy never trained on. "
                            "Compare this to the full-period number to check for overfitting."
                        )
                    if total_strategy_return > total_buyhold_return:
                        verdict_lines.append(f"**Strategy vs Buy & Hold: Strategy won ✅** — {total_strategy_return:.1f}% vs {total_buyhold_return:.1f}%.")
                    else:
                        verdict_lines.append(f"**Strategy vs Buy & Hold: Buy & Hold won ❌** — {total_buyhold_return:.1f}% vs {total_strategy_return:.1f}%. Too many exits missed the uptrend.")
                    if total_hybrid_return is not None and total_hybrid_return > total_strategy_return:
                        verdict_lines.append(f"**🆕 Hybrid improved over original** — {total_hybrid_return:.1f}% vs {total_strategy_return:.1f}%.")

                    verdict_lines.append("⚠️ **Remember:** No costs, taxes, or slippage modeled. Real returns would be lower. Past results never guarantee future performance.")

                    for line in verdict_lines:
                        st.markdown(f"- {line}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Educational tool only. Not financial advice. Technical indicators are "
    "simplified heuristics and do not account for news, risk, or macro conditions."
)
st.caption("📊 AI Stock Screener · built with Streamlit, yfinance & Gemini")
