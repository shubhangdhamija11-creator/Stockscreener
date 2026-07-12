"""
crypto_data.py
Drop-in replacement for yfinance, but for crypto pairs via Binance's
public REST API. No API key required for market data (klines/candles).

Usage:
    from crypto_data import fetch_klines
    df = fetch_klines("BTCUSDT", interval="1d", limit=200)
"""

import requests
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

# Binance interval strings: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
def fetch_klines(symbol: str, interval: str = "1d", limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a crypto pair from Binance.

    symbol: e.g. "BTCUSDT", "ETHUSDT" (no slash, uppercase, quote asset attached)
    interval: candle timeframe, e.g. "1h", "4h", "1d"
    limit: number of candles to pull (max 1000 per Binance's API limit)

    Returns a DataFrame with columns matching what your yfinance-based
    screener already expects: Open, High, Low, Close, Volume, indexed by
    datetime — so your existing scoring functions (RSI, MACD, EMA, etc.)
    should work on this without modification.
    """
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise ValueError(f"No data returned for {symbol}. Check the symbol is valid on Binance.")

    df = pd.DataFrame(raw, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df.index.name = "Date"

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    return df[["Open", "High", "Low", "Close", "Volume"]]


def get_top_usdt_pairs(limit: int = 30) -> list:
    """
    Returns the top N USDT trading pairs by 24h quote volume — useful for
    scanning the most liquid/active pairs instead of hardcoding a symbol list.
    """
    url = "https://api.binance.com/api/v3/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    usdt_pairs = [d for d in data if d["symbol"].endswith("USDT")]
    usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

    return [d["symbol"] for d in usdt_pairs[:limit]]


if __name__ == "__main__":
    # quick sanity check
    df = fetch_klines("BTCUSDT", interval="1d", limit=10)
    print(df.tail())
    print("\nTop 10 USDT pairs by volume:")
    print(get_top_usdt_pairs(10))
