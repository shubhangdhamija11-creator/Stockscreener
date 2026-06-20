"""
AI Stock Screener (Educational Trial) - Indian Markets (NSE/BSE)
------------------------------------------------------------------
Combines free technical/fundamental data (yfinance) with a free
Gemini API call to generate a plain-English Buy/Hold/Sell narrative.

EDUCATIONAL USE ONLY -- NOT FINANCIAL ADVICE.
The buy/hold/sell signal below comes from a simple rule-based
calculation on common technical indicators. It does not account for
news, macro conditions, deeper fundamentals, or your personal risk
tolerance. Always do your own research before investing.
"""

import os
import requests
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Stock Screener (Educational)", layout="centered")

GEMINI_MODEL = "gemini-2.5-flash"  # check ai.google.dev for current free-tier model names
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ---------- Sidebar: API key ----------
st.sidebar.header("Setup")
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API key (free, no card -- get one at aistudio.google.com)",
    type="password",
)
st.sidebar.caption("Your key stays in this session only, never saved or shared.")


# ---------- Data fetching ----------
@st.cache_data(ttl=900)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1y")
    return df


@st.cache_data(ttl=900)
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info or {}
    keys = [
        "longName", "currentPrice", "trailingPE", "forwardPE",
        "priceToBook", "returnOnEquity", "debtToEquity",
        "dividendYield", "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    ]
    return {k: info.get(k) for k in keys}


def resolve_ticker(raw: str):
    """Try NSE (.NS) first, then BSE (.BO); return the working symbol + history."""
    raw = raw.strip().upper().replace(".NS", "").replace(".BO", "")
    for suffix in [".NS", ".BO"]:
        symbol = raw + suffix
        df = fetch_price_history(symbol)
        if not df.empty:
            return symbol, df
    return None, pd.DataFrame()


# ---------- Indicator calculations ----------
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
    close = df["Close"]
    rsi = compute_rsi(close)
    macd_line, signal_line = compute_macd(close)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    return {
        "price": round(float(close.iloc[-1]), 2),
        "rsi": round(float(rsi.iloc[-1]), 2) if pd.notna(rsi.iloc[-1]) else None,
        "macd": round(float(macd_line.iloc[-1]), 2),
        "macd_signal": round(float(signal_line.iloc[-1]), 2),
        "sma50": round(float(sma50.iloc[-1]), 2) if pd.notna(sma50.iloc[-1]) else None,
        "sma200": round(float(sma200.iloc[-1]), 2) if pd.notna(sma200.iloc[-1]) else None,
    }


def rule_based_signal(ind: dict):
    score = 0
    if ind["rsi"] is not None:
        if ind["rsi"] < 30:
            score += 1
        elif ind["rsi"] > 70:
            score -= 1
    score += 1 if ind["macd"] > ind["macd_signal"] else -1
    if ind["sma50"] is not None:
        score += 1 if ind["price"] > ind["sma50"] else -1
    if ind["sma200"] is not None:
        score += 1 if ind["price"] > ind["sma200"] else -1

    if score >= 2:
        return "BUY", score
    elif score <= -2:
        return "SELL", score
    return "HOLD", score


# ---------- AI narrative ----------
def generate_narrative(symbol, fundamentals, indicators, signal, api_key) -> str:
    prompt = f"""
You are an educational stock-analysis assistant. Based ONLY on the data below,
write a short, plain-English analysis of {symbol} for a beginner investor.

Data:
- Technical: {indicators}
- Fundamentals: {fundamentals}
- Rule-based signal: {signal}

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
st.title("📊 AI Stock Screener — Educational Trial")
st.caption("NSE/BSE stocks · RSI + MACD + Moving Averages · Gemini-generated summary")

raw_input = st.text_input("Enter NSE/BSE ticker symbol (e.g. RELIANCE, TCS, INFY)")

if st.button("Analyze") and raw_input:
    with st.spinner("Fetching data..."):
        symbol, df = resolve_ticker(raw_input)

    if df.empty:
        st.error("Couldn't find data for that ticker. Try the exact NSE/BSE symbol.")
    else:
        fundamentals = fetch_fundamentals(symbol)
        indicators = compute_indicators(df)
        signal, score = rule_based_signal(indicators)

        st.subheader(f"{fundamentals.get('longName', symbol)} ({symbol})")
        badge = {"BUY": "🟢 BUY", "HOLD": "🟡 HOLD", "SELL": "🔴 SELL"}[signal]
        st.markdown(f"### {badge}  *(rule-based, score: {score})*")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Price", f"₹{indicators['price']}")
            st.metric("RSI (14)", indicators["rsi"])
        with col2:
            st.metric("MACD", indicators["macd"])
            st.metric("Signal Line", indicators["macd_signal"])

        st.line_chart(df["Close"], height=200)

        st.subheader("Fundamentals")
        st.table(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]))

        st.subheader("AI Summary")
        if api_key:
            with st.spinner("Generating AI narrative..."):
                narrative = generate_narrative(symbol, fundamentals, indicators, signal, api_key)
            st.write(narrative)
        else:
            st.info("Enter a free Gemini API key in the sidebar to get an AI-written summary.")

st.divider()
st.caption(
    "⚠️ Educational tool only. Not financial advice. Technical indicators are "
    "simplified heuristics and do not account for news, risk, or macro conditions."
)
