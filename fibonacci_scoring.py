"""
fibonacci_scoring.py
Shared Fibonacci retracement logic for both the stock screener (NSE/BSE)
and the crypto bot. Works on any OHLCV DataFrame with High/Low/Close
columns, regardless of source (yfinance or Binance).

Design choice: Fib levels are used as a CONFLUENCE bonus added to your
existing momentum score, not as a standalone signal. See conversation
notes on why raw Fib retracement alone is noisy.
"""

import pandas as pd

FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]


def detect_swing_high_low(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """
    Auto-detects the most recent swing high and swing low over a lookback
    window. For crypto (more volatile), a shorter lookback (20-30) tends
    to track meaningful swings better than the 50-100 you'd use for stocks.

    Returns (swing_high, swing_low, high_idx, low_idx)
    """
    window = df.tail(lookback)
    swing_high = window["High"].max()
    swing_low = window["Low"].min()
    high_idx = window["High"].idxmax()
    low_idx = window["Low"].idxmin()
    return swing_high, swing_low, high_idx, low_idx


def calculate_fib_levels(swing_high: float, swing_low: float, uptrend: bool = True) -> dict:
    """
    Calculates standard Fibonacci retracement levels between a swing high
    and swing low.

    uptrend=True: measures retracement DOWN from high (for pullback-buy setups)
    uptrend=False: measures retracement UP from low (for pullback-sell/short setups)
    """
    diff = swing_high - swing_low
    levels = {}

    for ratio in FIB_RATIOS:
        if uptrend:
            levels[ratio] = swing_high - (diff * ratio)
        else:
            levels[ratio] = swing_low + (diff * ratio)

    return levels


def fib_confluence_score(current_price: float, fib_levels: dict, tolerance_pct: float = 1.0) -> dict:
    """
    Checks if current_price is sitting near a Fib level (within tolerance_pct%).
    Returns a score bonus (0-2 points) and which level it's near, so you can
    plug this straight into your existing scoring function, e.g.:

        score += fib_result["bonus"]

    The 50%/61.8% levels get a higher bonus since they're the most-watched
    "golden pocket" levels traders react to.
    """
    result = {"bonus": 0, "near_level": None, "level_price": None}

    for ratio, level_price in fib_levels.items():
        pct_diff = abs(current_price - level_price) / level_price * 100
        if pct_diff <= tolerance_pct:
            # golden pocket (50%-61.8%) weighted higher than shallow/deep levels
            bonus = 2 if ratio in (0.5, 0.618) else 1
            if bonus > result["bonus"]:
                result["bonus"] = bonus
                result["near_level"] = ratio
                result["level_price"] = round(level_price, 4)

    return result


def get_fib_signal(df: pd.DataFrame, lookback: int = 50, tolerance_pct: float = 1.0, uptrend: bool = True) -> dict:
    """
    One-call convenience wrapper: pass in your OHLCV df, get back everything
    needed to fold into your existing BUY/HOLD/SELL scoring logic.

    Example integration in your existing score function:

        fib = get_fib_signal(df, lookback=30 if is_crypto else 50)
        score += fib["bonus"]
        if fib["near_level"]:
            reasons.append(f"Price near {fib['near_level']*100:.1f}% Fib retracement")
    """
    swing_high, swing_low, high_idx, low_idx = detect_swing_high_low(df, lookback)
    fib_levels = calculate_fib_levels(swing_high, swing_low, uptrend)
    current_price = df["Close"].iloc[-1]
    confluence = fib_confluence_score(current_price, fib_levels, tolerance_pct)

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_levels": {f"{r*100:.1f}%": round(p, 4) for r, p in fib_levels.items()},
        "current_price": current_price,
        "bonus": confluence["bonus"],
        "near_level": confluence["near_level"],
        "level_price": confluence["level_price"],
    }


if __name__ == "__main__":
    # quick sanity check with dummy data
    import numpy as np
    dates = pd.date_range("2026-01-01", periods=60)
    prices = np.linspace(100, 150, 60) + np.random.randn(60) * 2
    df = pd.DataFrame({
        "Open": prices, "High": prices + 2, "Low": prices - 2,
        "Close": prices, "Volume": 1000
    }, index=dates)

    signal = get_fib_signal(df, lookback=30)
    print(signal)
