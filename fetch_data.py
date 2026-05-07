import yfinance as yf
import pandas as pd
from cpr_levels import add_cpr_to_intraday


def fetch_and_save(symbol="TSLA", period="60d", interval="5m", output="data.csv"):
    ticker = yf.Ticker(symbol)

    # --- 5m intraday data (yfinance allows 60 days — 8x more training data than 1m/7d) ---
    print(f"Downloading {symbol} 5m intraday data (last {period})...")
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No intraday data returned for {symbol}. Market may be closed.")

    df_5m = df.between_time("09:30", "15:55")  # last 5m bar starts at 15:55
    df_5m.index.name = "time"
    df_5m.columns = [c.lower() for c in df_5m.columns]
    df_5m = df_5m[["open", "high", "low", "close", "volume"]]
    df_3m = df_5m  # alias — keeping variable name consistent with rest of pipeline

    # --- Daily data for CPR level calculation ---
    print(f"Downloading {symbol} daily data for CPR pivot levels...")
    daily = ticker.history(period="30d", interval="1d", auto_adjust=True)
    if daily.empty:
        raise ValueError(f"No daily data returned for {symbol}.")

    # Merge prior-day CPR levels into every 3-min bar
    df_3m = add_cpr_to_intraday(df_3m, daily)

    df_3m.to_csv(output)
    print(f"Saved {len(df_3m)} 3-min bars with CPR levels [{df_3m.index[0].date()} to {df_3m.index[-1].date()}]")
    return df_3m
