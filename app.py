"""
AI Stock Screener (Educational Trial) - Indian Markets (NSE/BSE)
------------------------------------------------------------------
Combines free technical/fundamental data (yfinance) with a free
Gemini API call to generate a plain-English Buy/Hold/Sell narrative.
Also supports scanning a list of stocks into Bullish/Neutral/Bearish
buckets using the same rule-based score (no AI calls, so it's fast).

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

# Brand palette -- keep in sync with .streamlit/config.toml
GREEN = "#22C55E"
AMBER = "#FBBF24"
RED = "#EF4444"

st.set_page_config(page_title="AI Stock Screener", page_icon="📊", layout="centered")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

NSE_STOCKS = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "INFY": "Infosys", "WIPRO": "Wipro", "HCLTECH": "HCL Technologies",
    "TECHM": "Tech Mahindra", "LTIM": "LTIMindtree", "MPHASIS": "Mphasis",
    "PERSISTENT": "Persistent Systems",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank", "SBIN": "State Bank of India",
    "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank",
    "INDUSINDBK": "IndusInd Bank", "PNB": "Punjab National Bank",
    "BANKBARODA": "Bank of Baroda", "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv", "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance", "ICICIPRULI": "ICICI Prudential Life",
    "ICICIGI": "ICICI Lombard General Insurance", "SBICARD": "SBI Cards",
    "ONGC": "Oil and Natural Gas Corp", "IOC": "Indian Oil Corporation",
    "BPCL": "Bharat Petroleum", "NTPC": "NTPC Limited",
    "POWERGRID": "Power Grid Corporation", "COALINDIA": "Coal India",
    "ADANIGREEN": "Adani Green Energy", "ADANIPOWER": "Adani Power",
    "TATAPOWER": "Tata Power", "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports & SEZ",
    "MARUTI": "Maruti Suzuki", "TATAMOTORS": "Tata Motors",
    "M&M": "Mahindra & Mahindra", "BAJAJ-AUTO": "Bajaj Auto",
    "HEROMOTOCO": "Hero MotoCorp", "EICHERMOT": "Eicher Motors",
    "TVSMOTOR": "TVS Motor Company",
    "HINDUNILVR": "Hindustan Unilever", "ITC": "ITC Limited",
    "NESTLEIND": "Nestle India", "BRITANNIA": "Britannia Industries",
    "DABUR": "Dabur India", "GODREJCP": "Godrej Consumer Products",
    "TATACONSUM": "Tata Consumer Products", "MARICO": "Marico Limited",
    "SUNPHARMA": "Sun Pharmaceutical", "DRREDDY": "Dr. Reddy's Laboratories",
    "CIPLA": "Cipla", "DIVISLAB": "Divi's Laboratories",
    "APOLLOHOSP": "Apollo Hospitals", "LUPIN": "Lupin Limited", "BIOCON": "Biocon",
    "TATASTEEL": "Tata Steel", "JSWSTEEL": "JSW Steel",
    "HINDALCO": "Hindalco Industries", "VEDL": "Vedanta Limited",
    "JINDALSTEL": "Jindal Steel & Power",
    "ULTRACEMCO": "UltraTech Cement", "SHREECEM": "Shree Cement",
    "AMBUJACEM": "Ambuja Cements", "ACC": "ACC Limited",
    "BHARTIARTL": "Bharti Airtel", "IDEA": "Vodafone Idea",
    "LT": "Larsen & Toubro", "GRASIM": "Grasim Industries", "DLF": "DLF Limited",
    "TITAN": "Titan Company", "ASIANPAINT": "Asian Paints", "HAVELLS": "Havells India",
    "BEL": "Bharat Electronics", "SIEMENS": "Siemens India",
    "PIDILITIND": "Pidilite Industries", "ZOMATO": "Eternal (Zomato)",
    "NYKAA": "FSN E-Commerce (Nykaa)", "PAYTM": "One97 Communications (Paytm)",
    "IRCTC": "Indian Railway Catering & Tourism", "TRENT": "Trent Limited",
}

QUICK_PICKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "TATAMOTORS"]


# ---------- Sidebar ----------
st.sidebar.header("⚙️ Setup")


def _secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


api_key = os.environ.get("GEMINI_API_KEY") or _secret_key()
if api_key:
    st.sidebar.success("✅ AI summaries are enabled — nothing to set up!")
else:
    api_key = st.sidebar.text_input(
        "Gemini API key (free, no card — get one at aistudio.google.com)",
        type="password",
    )
    st.sidebar.caption("Your key stays in this session only, never saved or shared.")

st.sidebar.divider()
st.sidebar.caption("Educational, rule-based tool. Not financial advice.")

with st.sidebar.expander("📖 Indicator cheat sheet"):
    st.markdown(
        """
**200 SMA (structural gate)**
- Price above = safe zone, buy signals valid
- Price below = avoid all buys (structurally broken)

**EMA trend (20 vs 50)**
- 20 EMA above 50 EMA → uptrend
- 20 EMA below 50 EMA → downtrend

**Fast EMA (9 vs 20)**
- 9 EMA crosses above 20 EMA → entry trigger
- 9 EMA crosses below 20 EMA → exit trigger

**RSI (14)** — range 0–100
- Above 55 → momentum building, trend-continuation buy
- 45–55 → neutral
- Below 45 → momentum weak/bearish

**MACD vs Signal line**
- MACD above Signal → bullish crossover
- MACD below Signal → bearish crossover

**Volume vs 10-day average**
- ≥1.5x average → high conviction move
- <0.8x average → weak, don't trust the move
        """
    )


# ---------- Data fetching ----------
@st.cache_data(ttl=1800)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Retries briefly, then gives up quietly instead of crashing the app."""
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
        "longName", "currentPrice", "trailingPE", "forwardPE",
        "priceToBook", "returnOnEquity", "debtToEquity",
        "dividendYield", "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
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
    """Only called when the user opts into weekly/monthly view or a backtest --
    keeps the default single-stock flow from making extra requests."""
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
    raw = raw.strip().upper().replace(".NS", "").replace(".BO", "")
    for suffix in [".NS", ".BO"]:
        symbol = raw + suffix
        df = fetch_price_history(symbol)
        if not df.empty:
            return symbol, df
    return None, pd.DataFrame()


def format_fundamentals(f: dict) -> pd.DataFrame:
    labels = {
        "trailingPE": "P/E Ratio (Trailing)", "forwardPE": "P/E Ratio (Forward)",
        "priceToBook": "Price-to-Book", "returnOnEquity": "Return on Equity",
        "debtToEquity": "Debt-to-Equity", "dividendYield": "Dividend Yield",
        "fiftyTwoWeekHigh": "52-Week High (₹)", "fiftyTwoWeekLow": "52-Week Low (₹)",
    }
    rows = []
    if f.get("marketCap"):
        rows.append(("Market Cap (₹ Cr)", round(f["marketCap"] / 1e7, 1)))
    for key, label in labels.items():
        val = f.get(key)
        if val is not None:
            rows.append((label, round(val, 2) if isinstance(val, (int, float)) else val))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------- Indicators ----------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["Close"]
    volume = df["Volume"]

    rsi             = compute_rsi(close)
    macd_line, sig_line = compute_macd(close)

    ema9   = close.ewm(span=9,   adjust=False).mean()
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    sma200 = close.rolling(200).mean()

    vol_avg10 = volume.rolling(10).mean()

    def _f(s): return round(float(s.iloc[-1]), 2) if pd.notna(s.iloc[-1]) else None

    return {
        "price":      round(float(close.iloc[-1]), 2),
        "rsi":        _f(rsi),
        "macd":       _f(macd_line),
        "macd_signal":_f(sig_line),
        "ema9":       _f(ema9),
        "ema20":      _f(ema20),
        "ema50":      _f(ema50),
        "sma200":     _f(sma200),
        "volume":     int(volume.iloc[-1]),
        "vol_avg10":  int(vol_avg10.iloc[-1]) if pd.notna(vol_avg10.iloc[-1]) else None,
    }


# ---- Scoring rules (from the two strategy documents) ----
# Each rule contributes +1 (bullish) or -1 (bearish) or 0 (neutral).
# Max possible score = +6, min = -6.
# BUY requires ≥ 4, SELL requires ≤ -4, else HOLD.
# The 200 SMA structural filter is a hard gate:
#   if price is BELOW 200 SMA the strategy says "never buy" -- no BUY signal at all.

def compute_signal(ind: dict):
    breakdown = []
    score = 0

    # ── 1. STRUCTURAL FILTER: Price vs 200 SMA ──────────────────────────────
    above_200 = ind["sma200"] is not None and ind["price"] > ind["sma200"]
    if ind["sma200"] is not None:
        if above_200:
            score += 1
            breakdown.append((
                "200 SMA structural filter", f"₹{ind['price']} vs ₹{ind['sma200']}",
                "🟢 Bullish", "Price above 200 SMA — long-term uptrend intact",
                "Price ABOVE = safe to trade  ·  Price BELOW = avoid all buy signals",
            ))
        else:
            score -= 1
            breakdown.append((
                "200 SMA structural filter", f"₹{ind['price']} vs ₹{ind['sma200']}",
                "🔴 Bearish", "Price below 200 SMA — structurally compromised, avoid buys",
                "Price ABOVE = safe to trade  ·  Price BELOW = avoid all buy signals",
            ))

    # ── 2. TREND DIRECTION: 20 EMA vs 50 EMA ────────────────────────────────
    if ind["ema20"] is not None and ind["ema50"] is not None:
        if ind["ema20"] > ind["ema50"]:
            score += 1
            breakdown.append((
                "EMA trend (20 vs 50)", f"₹{ind['ema20']} vs ₹{ind['ema50']}",
                "🟢 Bullish", "20 EMA above 50 EMA — short-term momentum upward",
                "20 EMA above 50 EMA = uptrend  ·  below = downtrend",
            ))
        else:
            score -= 1
            breakdown.append((
                "EMA trend (20 vs 50)", f"₹{ind['ema20']} vs ₹{ind['ema50']}",
                "🔴 Bearish", "20 EMA below 50 EMA — short-term momentum downward",
                "20 EMA above 50 EMA = uptrend  ·  below = downtrend",
            ))

    # ── 3. FAST MOMENTUM: 9 EMA vs 21 EMA (Doc 2 Setup 1) ───────────────────
    if ind["ema9"] is not None and ind["ema20"] is not None:
        if ind["ema9"] > ind["ema20"]:
            score += 1
            breakdown.append((
                "Fast EMA (9 vs 20)", f"₹{ind['ema9']} vs ₹{ind['ema20']}",
                "🟢 Bullish", "9 EMA crossed above 20 EMA — quick momentum burst",
                "9 EMA above 20 EMA = entry trigger  ·  below = exit trigger",
            ))
        else:
            score -= 1
            breakdown.append((
                "Fast EMA (9 vs 20)", f"₹{ind['ema9']} vs ₹{ind['ema20']}",
                "🔴 Bearish", "9 EMA below 20 EMA — momentum fading",
                "9 EMA above 20 EMA = entry trigger  ·  below = exit trigger",
            ))

    # ── 4. RSI MOMENTUM CONFIRMATION (Doc 1: RSI > 55 = momentum buy) ───────
    if ind["rsi"] is not None:
        if ind["rsi"] > 55:
            score += 1
            breakdown.append((
                "RSI (14)", ind["rsi"], "🟢 Bullish",
                "Above 55 — momentum is building (trend-continuation buy zone)",
                "0–100  ·  >55 momentum buy  ·  45–55 neutral  ·  <45 weak/bearish",
            ))
        elif ind["rsi"] < 45:
            score -= 1
            breakdown.append((
                "RSI (14)", ind["rsi"], "🔴 Bearish",
                "Below 45 — momentum weakening",
                "0–100  ·  >55 momentum buy  ·  45–55 neutral  ·  <45 weak/bearish",
            ))
        else:
            breakdown.append((
                "RSI (14)", ind["rsi"], "⚪ Neutral",
                "Between 45–55 — no strong momentum signal",
                "0–100  ·  >55 momentum buy  ·  45–55 neutral  ·  <45 weak/bearish",
            ))

    # ── 5. MACD CROSSOVER (both docs agree) ──────────────────────────────────
    if ind["macd"] is not None and ind["macd_signal"] is not None:
        if ind["macd"] > ind["macd_signal"]:
            score += 1
            breakdown.append((
                "MACD crossover", f"{ind['macd']} vs {ind['macd_signal']}",
                "🟢 Bullish", "MACD above signal line — bullish crossover confirmed",
                "MACD > Signal = bullish  ·  MACD < Signal = bearish",
            ))
        else:
            score -= 1
            breakdown.append((
                "MACD crossover", f"{ind['macd']} vs {ind['macd_signal']}",
                "🔴 Bearish", "MACD below signal line — bearish crossover",
                "MACD > Signal = bullish  ·  MACD < Signal = bearish",
            ))

    # ── 6. VOLUME CONFIRMATION (Doc 1: ≥ 1.5x 10-period avg) ────────────────
    if ind["vol_avg10"] is not None and ind["vol_avg10"] > 0:
        vol_ratio = ind["volume"] / ind["vol_avg10"]
        if vol_ratio >= 1.5:
            score += 1
            breakdown.append((
                "Volume confirmation", f"{vol_ratio:.1f}x avg",
                "🟢 Bullish", f"Volume is {vol_ratio:.1f}x the 10-day average — strong conviction",
                "≥1.5x avg = high conviction  ·  1–1.5x = normal  ·  <1x = weak",
            ))
        elif vol_ratio < 0.8:
            score -= 1
            breakdown.append((
                "Volume confirmation", f"{vol_ratio:.1f}x avg",
                "🔴 Bearish", "Volume well below average — move lacks conviction",
                "≥1.5x avg = high conviction  ·  1–1.5x = normal  ·  <1x = weak",
            ))
        else:
            breakdown.append((
                "Volume confirmation", f"{vol_ratio:.1f}x avg",
                "⚪ Neutral", "Volume near average — no strong confirmation",
                "≥1.5x avg = high conviction  ·  1–1.5x = normal  ·  <1x = weak",
            ))

    max_score = 6
    # Hard structural gate from Doc 2: never show BUY when below 200 SMA
    if score >= 4 and above_200:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, breakdown


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def multi_timeframe_view(daily_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    """Daily uses the regular 1y data; Weekly/Monthly need the longer history
    so 50/200-period moving averages have enough bars to mean something."""
    weekly_df = resample_ohlc(extended_df, "W")
    monthly_df = resample_ohlc(extended_df, "ME")

    rows = []
    for label, tf_df, min_bars in [("Daily", daily_df, 60), ("Weekly", weekly_df, 60), ("Monthly", monthly_df, 24)]:
        if tf_df is None or len(tf_df) < min_bars:
            rows.append({"Timeframe": label, "Signal": "Not enough history", "Score": "—", "RSI": "—", "Price": "—"})
            continue
        ind = compute_indicators(tf_df)
        sig, score, _ = compute_signal(ind)
        rows.append({"Timeframe": label, "Signal": sig, "Score": score, "RSI": ind["rsi"], "Price": f"₹{ind['price']}"})
    return pd.DataFrame(rows)


def compute_score_series(df: pd.DataFrame):
    """Vectorized version of compute_signal — same 6-factor rules applied to
    every historical row so the backtest uses exactly the same logic as the
    live signal. Score range: -6 to +6. BUY = ≥4 AND above 200 SMA."""
    close  = df["Close"]
    volume = df["Volume"]

    rsi        = compute_rsi(close)
    macd_line, sig_line = compute_macd(close)
    ema9       = close.ewm(span=9,   adjust=False).mean()
    ema20      = close.ewm(span=20,  adjust=False).mean()
    ema50      = close.ewm(span=50,  adjust=False).mean()
    sma200     = close.rolling(200).mean()
    vol_avg10  = volume.rolling(10).mean()

    # 1. 200 SMA structural filter
    sma200_pts = np.where(close > sma200, 1, -1)
    sma200_pts = np.where(sma200.isna(), 0, sma200_pts)
    above_200  = close > sma200

    # 2. EMA trend: 20 vs 50
    ema_trend_pts = np.where(ema20 > ema50, 1, -1)
    ema_trend_pts = np.where(ema20.isna() | ema50.isna(), 0, ema_trend_pts)

    # 3. Fast EMA: 9 vs 20
    fast_ema_pts = np.where(ema9 > ema20, 1, -1)
    fast_ema_pts = np.where(ema9.isna() | ema20.isna(), 0, fast_ema_pts)

    # 4. RSI > 55 momentum buy
    rsi_pts = np.select([rsi > 55, rsi < 45], [1, -1], default=0)
    rsi_pts = np.where(rsi.isna(), 0, rsi_pts)

    # 5. MACD crossover
    macd_pts = np.where(macd_line > sig_line, 1, -1)
    macd_pts = np.where(macd_line.isna() | sig_line.isna(), 0, macd_pts)

    # 6. Volume >= 1.5x 10-day avg
    vol_ratio  = volume / vol_avg10.replace(0, np.nan)
    vol_pts    = np.select([vol_ratio >= 1.5, vol_ratio < 0.8], [1, -1], default=0)
    vol_pts    = np.where(vol_avg10.isna(), 0, vol_pts)

    score = pd.Series(
        sma200_pts + ema_trend_pts + fast_ema_pts + rsi_pts + macd_pts + vol_pts,
        index=df.index,
    )
    # Hard gate: never BUY below 200 SMA
    signal = pd.Series(
        np.select(
            [(score >= 4) & above_200, score <= -4],
            ["BUY", "SELL"],
            default="HOLD",
        ),
        index=df.index,
    )
    return score, signal


def run_backtest(df: pd.DataFrame, holding_days=(5, 20, 60)):
    """Long-only: act on YESTERDAY's signal (no lookahead bias). No transaction
    costs, slippage, or taxes are modeled -- this is for learning, not trading."""
    df = df.copy()
    score, signal = compute_score_series(df)

    warmup = 200  # first 200 rows can't have a valid 200-period average yet
    if len(df) <= warmup + max(holding_days):
        return None, None, None

    close = df["Close"]
    daily_return = close.pct_change()

    acted_signal = signal.shift(1)  # yesterday's signal drives today's action
    strategy_return = daily_return.where(acted_signal == "BUY", 0.0)

    bt = pd.DataFrame({"Signal": signal, "DailyReturn": daily_return}, index=df.index).iloc[warmup:]
    for h in holding_days:
        bt[f"fwd_{h}d"] = close.shift(-h) / close - 1

    # One row per signal type, with avg forward return + win rate for each holding period
    table_rows = []
    for sig in ["BUY", "HOLD", "SELL"]:
        subset = bt[bt["Signal"] == sig]
        table_rows.append({"Signal": sig, "Occurrences": len(subset)})
    summary_df = pd.DataFrame(table_rows)
    for h, label in zip(holding_days, ["~1 week", "~1 month", "~3 months"]):
        avg_returns, win_rates = [], []
        for sig in ["BUY", "HOLD", "SELL"]:
            valid = bt[bt["Signal"] == sig][f"fwd_{h}d"].dropna()
            avg_returns.append(round(valid.mean() * 100, 2) if len(valid) else None)
            win_rates.append(round((valid > 0).mean() * 100, 1) if len(valid) else None)
        summary_df[f"Avg return ({label})"] = avg_returns
        summary_df[f"Win rate % ({label})"] = win_rates

    # Equity curves, both rebased to 1.0 at the start of the tested window
    strat_window = strategy_return.iloc[warmup:].fillna(0)
    hold_window = daily_return.iloc[warmup:].fillna(0)
    cum_strategy = (1 + strat_window).cumprod()
    cum_buyhold = (1 + hold_window).cumprod()

    return summary_df, cum_strategy, cum_buyhold


def make_backtest_chart(cum_strategy: pd.Series, cum_buyhold: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_strategy.index, y=cum_strategy, name="Signal strategy (long only on BUY)",
                              line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=cum_buyhold.index, y=cum_buyhold, name="Just buy & hold",
                              line=dict(color="#9CA3AF", width=2, dash="dot")))
    fig.update_layout(
        height=280,
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


def make_gauge(score: int) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 34}},
        gauge={
            "axis": {"range": [-6, 6], "tickwidth": 1},
            "bar": {"color": "#E5E7EB", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-6, -4], "color": RED},
                {"range": [-4, 4],  "color": AMBER},
                {"range": [4, 6],   "color": GREEN},
            ],
        },
    ))
    fig.update_layout(
        height=170,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5E7EB"},
    )
    return fig


def scan_stocks(symbols):
    """Rule-based only -- no AI calls -- so it stays fast and free even for many stocks."""
    results, errors = [], []
    total = len(symbols)
    progress = st.progress(0, text="Starting scan...")
    for i, sym in enumerate(symbols):
        progress.progress((i + 1) / total, text=f"Scanning {sym} ({i + 1}/{total})...")
        resolved, df = resolve_ticker(sym)
        if df.empty:
            errors.append(sym)
            time.sleep(0.5)
            continue
        time.sleep(0.5)
        indicators = compute_indicators(df)
        signal, score, _ = compute_signal(indicators)
        results.append({
            "Symbol": sym, "Name": NSE_STOCKS.get(sym, sym),
            "Price": indicators["price"], "Score": score, "Signal": signal,
        })
    progress.empty()
    return results, errors


# ---------- AI narrative ----------
def generate_narrative(symbol, fundamentals, breakdown, signal, api_key) -> str:
    reasons = "; ".join(f"{row[0]}: {row[3]}" for row in breakdown)
    prompt = f"""
You are an educational stock-analysis assistant. Based ONLY on the data below,
write a short, plain-English analysis of {symbol} for a beginner investor.

Rule-based signal: {signal}
Reasons behind that signal: {reasons}
Fundamentals: {fundamentals}

Structure your response as:
1. One-sentence summary
2. 3 bullet points: what looks GOOD
3. 3 bullet points: what looks CONCERNING
4. One closing line reminding the reader this is educational, not financial advice,
   and that they should do their own research.
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


# ---------- UI ----------
st.title("📊 AI Stock Screener")
st.caption("Educational trial · NSE/BSE · RSI + MACD + Moving Averages + Gemini summary")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze a Stock", "📋 Screen a List", "🧪 Backtest"])

# ===== TAB 1: single-stock deep dive =====
with tab1:
    search_input = st.text_input(
        "Enter NSE/BSE symbol (e.g. RELIANCE, TCS, HDFCBANK)",
        placeholder="Type a ticker symbol and press Analyze",
    )
    analyze_clicked = st.button("Analyze 📊", type="primary", use_container_width=True)
    raw_input = search_input.strip() if analyze_clicked and search_input.strip() else None

    if raw_input:
        with st.spinner("Fetching data..."):
            symbol, df = resolve_ticker(raw_input)

        if df.empty:
            st.error(
                "Couldn't fetch data right now. Either the symbol is wrong, or the data "
                "provider is briefly rate-limiting this shared server — wait a minute or "
                "two and try again."
            )
        else:
            try:
                fundamentals = fetch_fundamentals(symbol)
                indicators = compute_indicators(df)
                signal, score, breakdown = compute_signal(indicators)

                with st.container(border=True):
                    st.subheader(f"{fundamentals.get('longName', symbol)} ({symbol})")
                    if signal == "BUY":
                        st.success(f"🟢 **BUY** — signal score {score} / 6")
                    elif signal == "SELL":
                        st.error(f"🔴 **SELL** — signal score {score} / 6")
                    else:
                        st.warning(f"🟡 **HOLD** — signal score {score} / 6")
                    st.plotly_chart(make_gauge(score), use_container_width=True, config={"displayModeBar": False})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Price", f"₹{indicators['price']}")
                        st.metric("RSI (14)", indicators["rsi"])
                        st.caption("🟢 >55 momentum buy · ⚪ 45–55 neutral · 🔴 <45 weak")
                    with col2:
                        st.metric("MACD", indicators["macd"])
                        st.metric("Signal Line", indicators["macd_signal"])
                        st.caption("🟢 MACD above Signal = bullish · 🔴 MACD below Signal = bearish")

                    st.plotly_chart(make_price_chart(df), use_container_width=True, config={"displayModeBar": False})
                    st.caption(
                        "🟣 Shaded band = Bollinger Bands (volatility) · dashed lines = 52-week high/low · "
                        "bottom bars = daily volume (green = up day, red = down day)"
                    )

                with st.expander("📐 Why this rating? — see the basis for the score"):
                    st.dataframe(
                        pd.DataFrame(breakdown, columns=["Indicator", "Value", "Signal", "Why", "Typical Range"]),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption(
                        "Score ranges -6 (all bearish) to +6 (all bullish). "
                        "≥4 AND above 200 SMA → BUY  ·  ≤-4 → SELL  ·  else → HOLD. "
                        "Strategy based on trend-continuation rules: 200 SMA gate, EMA alignment, "
                        "RSI>55 momentum, MACD crossover, volume confirmation."
                    )

                with st.expander("💰 Fundamentals"):
                    if fundamentals:
                        st.dataframe(format_fundamentals(fundamentals), hide_index=True, use_container_width=True)
                    else:
                        st.caption("Fundamentals temporarily unavailable (data provider rate limit). Price chart and technical score above are unaffected.")

                with st.expander("📅 Daily vs Weekly vs Monthly view"):
                    st.caption(
                        "Same scoring rules, applied to weekly and monthly candles instead of daily. "
                        "Fetches a bit more history the first time (~5 years), then it's cached."
                    )
                    if st.button("Load weekly/monthly view", key="load_mtf"):
                        with st.spinner("Fetching extended history..."):
                            ext_df = fetch_extended_history(symbol)
                        if ext_df.empty:
                            st.error("Couldn't fetch extended history right now — try again shortly.")
                        else:
                            mtf = multi_timeframe_view(df, ext_df)
                            st.dataframe(mtf, hide_index=True, use_container_width=True)
                            st.caption(
                                "Short-term (Daily) and long-term (Monthly) signals can disagree — that's normal. "
                                "A stock can look weak short-term while still being fine for a long-term hold, or vice versa."
                            )

                st.subheader("🤖 AI Summary")
                if api_key:
                    with st.spinner("Generating AI narrative..."):
                        narrative = generate_narrative(symbol, fundamentals, breakdown, signal, api_key)
                    st.write(narrative)
                else:
                    st.info("Enter a free Gemini API key in the sidebar to get an AI-written summary.")
            except Exception as e:
                st.error(f"Something went wrong while analyzing this stock: {e}")
                st.caption("This is usually temporary — try again in a minute.")
    else:
        st.divider()
        st.markdown("#### 👋 Get started in 3 taps")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1️⃣ Pick**")
            st.caption("Tap a quick pick or search any NSE/BSE stock")
        with c2:
            st.markdown("**2️⃣ Analyze**")
            st.caption("See RSI, MACD, trend score & fundamentals")
        with c3:
            st.markdown("**3️⃣ Understand**")
            st.caption("Read the plain-English AI takeaway")

# ===== TAB 2: bulk bullish/bearish screener =====
with tab2:
    st.subheader("📋 Screen multiple stocks at once")
    st.caption(
        "Runs the same rule-based RSI/MACD/trend score across a list of stocks and "
        "sorts them into Bullish, Neutral, and Bearish buckets. No AI calls here, "
        "so it's fast and won't touch your Gemini quota."
    )

    chosen = st.multiselect(
        "Choose stocks to screen (start with a small list — each one is a live fetch)",
        options=sorted(NSE_STOCKS.keys()),
        default=QUICK_PICKS,
        format_func=lambda s: f"{s} — {NSE_STOCKS[s]}",
    )

    if st.button("Scan list 🔎", use_container_width=True) and chosen:
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

            display_df = df_results[["Symbol", "Name", "Price", "Score", "Signal"]]
            try:
                styled = (
                    display_df.style
                    .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-4, vmax=4)
                    .format({"Price": "₹{:.2f}"})
                )
                st.dataframe(styled, hide_index=True, use_container_width=True)
            except Exception:
                # Fallback if styling isn't supported in this environment
                st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.caption("Sorted most bullish → most bearish. Green = higher score, red = lower score.")

        if errors:
            st.caption(f"Couldn't fetch data for: {', '.join(errors)}")

# ===== TAB 3: backtest the scoring rule against history =====
with tab3:
    st.subheader("🧪 Backtest the signal")
    st.caption(
        "Runs the new 6-factor strategy (200 SMA gate · EMA alignment · Fast EMA · "
        "RSI>55 · MACD crossover · Volume) against ~5 years of history and checks "
        "what actually happened afterward. BUY requires score ≥4 AND price above 200 SMA."
    )
    st.warning(
        "⚠️ No transaction costs, taxes, or slippage are modeled. Past performance on "
        "historical data does NOT guarantee future results. This is for learning how "
        "backtesting works, not a trading system.",
        icon="⚠️",
    )

    bt_search = st.text_input(
        "Enter NSE/BSE symbol to backtest (e.g. RELIANCE, TCS, HDFCBANK)",
        placeholder="Any NSE/BSE stock — type the symbol",
        key="bt_search",
    )
    run_bt = st.button("Run backtest 🧪", type="primary", use_container_width=True)

    if run_bt:
        bt_raw = bt_search.strip() if bt_search.strip() else None

        if not bt_raw:
            st.error("Enter an NSE/BSE symbol first.")
        else:
            with st.spinner("Fetching ~5 years of history and running the backtest..."):
                bt_symbol, _ = resolve_ticker(bt_raw)
                ext_df = fetch_extended_history(bt_symbol) if bt_symbol else pd.DataFrame()

            if ext_df.empty:
                st.error("Couldn't fetch enough history for this symbol right now — try again shortly.")
            else:
                summary_df, cum_strategy, cum_buyhold = run_backtest(ext_df)
                if summary_df is None:
                    st.error("Not enough history available for a meaningful backtest on this stock.")
                else:
                    st.markdown(f"#### Results for {bt_symbol}")
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                    st.caption(
                        "'Occurrences' = how many days in the backtest had that signal. "
                        "'Avg return' = average price change over the holding period after that signal. "
                        "'Win rate' = % of the time the return was positive."
                    )

                    total_strategy_return = (cum_strategy.iloc[-1] - 1) * 100
                    total_buyhold_return = (cum_buyhold.iloc[-1] - 1) * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Strategy total return", f"{total_strategy_return:.1f}%")
                    c2.metric("Buy & hold total return", f"{total_buyhold_return:.1f}%")

                    st.plotly_chart(make_backtest_chart(cum_strategy, cum_buyhold),
                                     use_container_width=True, config={"displayModeBar": False})
                    st.caption(
                        "Strategy = only invested on days after a BUY signal, in cash otherwise. "
                        "Compare it to simply buying and holding the whole time."
                    )

                    # ── Plain-English verdict ─────────────────────────────
                    st.subheader("🗣️ What do these results actually mean?")
                    buy_rows = summary_df[summary_df["Signal"] == "BUY"]
                    beat = total_strategy_return > total_buyhold_return
                    buy_wr_1m = buy_rows["Win rate % (~1 month)"].values[0] if len(buy_rows) else None
                    buy_ret_1m = buy_rows["Avg return (~1 month)"].values[0] if len(buy_rows) else None
                    buy_occ = buy_rows["Occurrences"].values[0] if len(buy_rows) else 0

                    verdict_lines = []
                    verdict_lines.append(
                        f"**Signal frequency:** The strategy produced a BUY signal on **{buy_occ} days** "
                        f"out of the ~{len(ext_df)} trading days tested (~{round(buy_occ/max(len(ext_df),1)*100)}% of the time). "
                        "Fewer signals = more selective = each signal carries more weight."
                    )
                    if buy_wr_1m is not None:
                        if buy_wr_1m >= 60:
                            verdict_lines.append(
                                f"**1-month win rate: {buy_wr_1m}% ✅** — historically, more than 6 out of 10 BUY signals "
                                "were followed by a gain one month later. That's a meaningful edge."
                            )
                        elif buy_wr_1m >= 50:
                            verdict_lines.append(
                                f"**1-month win rate: {buy_wr_1m}% ⚠️** — just above 50%. Marginally better than a coin flip. "
                                "Not strong enough to rely on alone."
                            )
                        else:
                            verdict_lines.append(
                                f"**1-month win rate: {buy_wr_1m}% ❌** — below 50%. The BUY signal was followed by a loss "
                                "more often than a gain. The strategy didn't work well on this stock historically."
                            )
                    if buy_ret_1m is not None:
                        sign = "positive" if buy_ret_1m > 0 else "negative"
                        verdict_lines.append(
                            f"**Avg 1-month return after BUY: {buy_ret_1m}%** — on average the stock moved "
                            f"{abs(buy_ret_1m)}% in a {sign} direction in the month after a BUY signal."
                        )
                    if beat:
                        verdict_lines.append(
                            f"**Strategy vs Buy & Hold: Strategy won ✅** — the signal-based approach returned "
                            f"{total_strategy_return:.1f}% vs {total_buyhold_return:.1f}% for simply holding. "
                            "It added value on this stock over this time period."
                        )
                    else:
                        verdict_lines.append(
                            f"**Strategy vs Buy & Hold: Buy & Hold won ❌** — simply holding returned "
                            f"{total_buyhold_return:.1f}% vs {total_strategy_return:.1f}% for the strategy. "
                            "The signals caused you to miss some of the uptrend by being in cash too often."
                        )
                    verdict_lines.append(
                        "⚠️ **Remember:** This is one stock over one historical window. "
                        "Test 5–10 different stocks before drawing any conclusions. "
                        "Past results never guarantee future performance."
                    )
                    for line in verdict_lines:
                        st.markdown(f"- {line}")

st.divider()
st.caption(
    "⚠️ Educational tool only. Not financial advice. Technical indicators are "
    "simplified heuristics and do not account for news, risk, or macro conditions."
)
st.caption("📊 AI Stock Screener · built with Streamlit, yfinance & Gemini")
