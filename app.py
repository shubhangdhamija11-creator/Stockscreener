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
import pandas as pd
import streamlit as st
import yfinance as yf

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
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API key (free, no card — get one at aistudio.google.com)",
    type="password",
)
st.sidebar.caption("Your key stays in this session only, never saved or shared.")
st.sidebar.divider()
st.sidebar.caption("Educational, rule-based tool. Not financial advice.")


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


def compute_signal(ind: dict):
    breakdown = []
    score = 0
    if ind["rsi"] is not None:
        if ind["rsi"] < 30:
            score += 1
            breakdown.append(("RSI (14)", ind["rsi"], "🟢 Bullish", "Below 30 — may be oversold"))
        elif ind["rsi"] > 70:
            score -= 1
            breakdown.append(("RSI (14)", ind["rsi"], "🔴 Bearish", "Above 70 — may be overbought"))
        else:
            breakdown.append(("RSI (14)", ind["rsi"], "⚪ Neutral", "Between 30-70 — no extreme"))

    if ind["macd"] > ind["macd_signal"]:
        score += 1
        breakdown.append(("MACD", f"{ind['macd']} vs {ind['macd_signal']}", "🟢 Bullish", "MACD above signal — upward momentum"))
    else:
        score -= 1
        breakdown.append(("MACD", f"{ind['macd']} vs {ind['macd_signal']}", "🔴 Bearish", "MACD below signal — downward momentum"))

    if ind["sma50"] is not None:
        if ind["price"] > ind["sma50"]:
            score += 1
            breakdown.append(("Price vs 50-day avg", f"₹{ind['price']} vs ₹{ind['sma50']}", "🟢 Bullish", "Above 50-day average — short-term uptrend"))
        else:
            score -= 1
            breakdown.append(("Price vs 50-day avg", f"₹{ind['price']} vs ₹{ind['sma50']}", "🔴 Bearish", "Below 50-day average — short-term downtrend"))

    if ind["sma200"] is not None:
        if ind["price"] > ind["sma200"]:
            score += 1
            breakdown.append(("Price vs 200-day avg", f"₹{ind['price']} vs ₹{ind['sma200']}", "🟢 Bullish", "Above 200-day average — long-term uptrend"))
        else:
            score -= 1
            breakdown.append(("Price vs 200-day avg", f"₹{ind['price']} vs ₹{ind['sma200']}", "🔴 Bearish", "Below 200-day average — long-term downtrend"))

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, score, breakdown


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

tab1, tab2 = st.tabs(["🔍 Analyze a Stock", "📋 Screen a List"])

# ===== TAB 1: single-stock deep dive =====
with tab1:
    st.subheader("Quick picks")
    quick_pick_clicked = None
    cols = st.columns(4)
    for i, sym in enumerate(QUICK_PICKS):
        if cols[i % 4].button(sym, use_container_width=True, key=f"qp_{sym}"):
            quick_pick_clicked = sym

    st.subheader("Or search")
    stock_options = [f"{name} ({sym})" for sym, name in sorted(NSE_STOCKS.items(), key=lambda x: x[1])]
    selected_option = st.selectbox(
        "Type a company name or symbol — matching results filter as you type",
        options=["— Select a stock —"] + stock_options,
        index=0,
    )
    custom_symbol = st.text_input("Not listed? Enter the exact NSE/BSE symbol here")
    analyze_clicked = st.button("Analyze 📊", type="primary", use_container_width=True)

    raw_input = None
    if quick_pick_clicked:
        raw_input = quick_pick_clicked
    elif analyze_clicked:
        if custom_symbol.strip():
            raw_input = custom_symbol.strip()
        elif selected_option != "— Select a stock —":
            match = re.search(r"\(([^)]+)\)$", selected_option)
            raw_input = match.group(1) if match else None

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
                        st.success(f"🟢 **BUY** — signal score {score} / 4")
                    elif signal == "SELL":
                        st.error(f"🔴 **SELL** — signal score {score} / 4")
                    else:
                        st.warning(f"🟡 **HOLD** — signal score {score} / 4")
                    st.progress((score + 4) / 8)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Price", f"₹{indicators['price']}")
                        st.metric("RSI (14)", indicators["rsi"])
                    with col2:
                        st.metric("MACD", indicators["macd"])
                        st.metric("Signal Line", indicators["macd_signal"])

                    st.line_chart(df["Close"], height=200)

                with st.expander("📐 Why this rating? — see the basis for the score"):
                    st.dataframe(
                        pd.DataFrame(breakdown, columns=["Indicator", "Value", "Signal", "Why"]),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption("Score ranges -4 (all bearish) to +4 (all bullish). ≥2 → BUY, ≤-2 → SELL, else → HOLD.")

                with st.expander("💰 Fundamentals"):
                    if fundamentals:
                        st.dataframe(format_fundamentals(fundamentals), hide_index=True, use_container_width=True)
                    else:
                        st.caption("Fundamentals temporarily unavailable (data provider rate limit). Price chart and technical score above are unaffected.")

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
            df_results = pd.DataFrame(results)
            bullish = df_results[df_results["Signal"] == "BUY"].sort_values("Score", ascending=False)
            neutral = df_results[df_results["Signal"] == "HOLD"]
            bearish = df_results[df_results["Signal"] == "SELL"].sort_values("Score")

            st.success(f"🟢 Bullish — {len(bullish)} stock(s)")
            st.dataframe(bullish[["Symbol", "Name", "Price", "Score"]], hide_index=True, use_container_width=True)

            st.warning(f"🟡 Neutral / Hold — {len(neutral)} stock(s)")
            st.dataframe(neutral[["Symbol", "Name", "Price", "Score"]], hide_index=True, use_container_width=True)

            st.error(f"🔴 Bearish — {len(bearish)} stock(s)")
            st.dataframe(bearish[["Symbol", "Name", "Price", "Score"]], hide_index=True, use_container_width=True)

        if errors:
            st.caption(f"Couldn't fetch data for: {', '.join(errors)}")

st.divider()
st.caption(
    "⚠️ Educational tool only. Not financial advice. Technical indicators are "
    "simplified heuristics and do not account for news, risk, or macro conditions."
            )
