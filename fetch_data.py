import yfinance as yf
import pandas as pd


def fetch_and_save(symbol="TSLA", period="7d", interval="1m", output="data.csv"):
    print(f"Downloading {symbol} {interval} data (last {period})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Market may be closed.")

    # Resample 1m -> 3m
    df_3m = df.resample("3min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # Keep only regular market hours 09:30–15:57 ET
    df_3m = df_3m.between_time("09:30", "15:57")

    df_3m.index.name = "time"
    df_3m.columns = [c.lower() for c in df_3m.columns]
    df_3m = df_3m[["open", "high", "low", "close", "volume"]]
    df_3m.to_csv(output)

    first_day = df_3m.index[0]
    last_day = df_3m.index[-1]
    print(f"Saved {len(df_3m)} 3-min bars  [{first_day.date()} to {last_day.date()}]")
    return df_3m
