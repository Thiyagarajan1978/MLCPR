import yfinance as yf
import pandas as pd
from cpr_levels import add_cpr_to_intraday


def fetch_and_save(symbol="TSLA", period="730d", interval="1h", output="data.csv"):
    ticker = yf.Ticker(symbol)

    # 1h bars — yfinance allows up to 730 days at this resolution (vs 60 days at 5m)
    print(f"Downloading {symbol} 1h intraday data (last {period})...")
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No intraday data returned for {symbol}. Market may be closed.")

    # Regular session only: 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 (7 bars/day)
    df_1h = df.between_time("09:30", "15:30")
    df_1h.columns = [c.lower() for c in df_1h.columns]
    df_1h = df_1h[["open", "high", "low", "close", "volume"]]
    # Strip timezone → naive Eastern Time so CSV timestamps are always "HH:MM" as seen on chart
    if df_1h.index.tz is not None:
        df_1h.index = df_1h.index.tz_convert("America/New_York").tz_localize(None)
    df_1h.index.name = "time"

    # Daily data for CPR calculation — need 2y to cover all intraday dates
    print(f"Downloading {symbol} daily data for CPR pivot levels...")
    daily = ticker.history(period="2y", interval="1d", auto_adjust=True)
    if daily.empty:
        raise ValueError(f"No daily data returned for {symbol}.")

    df_1h = add_cpr_to_intraday(df_1h, daily)

    df_1h.to_csv(output)
    print(f"Saved {len(df_1h)} 1h bars with CPR levels "
          f"[{df_1h.index[0].date()} to {df_1h.index[-1].date()}]")
    return df_1h
