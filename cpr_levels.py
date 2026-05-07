import pandas as pd


def compute_cpr(high, low, close):
    pp = (high + low + close) / 3
    bc = (high + low) / 2          # Bottom Central Pivot
    tc = (2 * pp) - bc             # Top Central Pivot
    # When close < midpoint, tc < bc (inverted CPR). Always keep top > bottom.
    cpr_top    = round(max(tc, bc), 4)
    cpr_bottom = round(min(tc, bc), 4)
    r1 = (2 * pp) - low
    r2 = pp + (high - low)
    s1 = (2 * pp) - high
    s2 = pp - (high - low)
    return {
        "pp":         round(pp, 4),
        "cpr_top":    cpr_top,
        "cpr_bottom": cpr_bottom,
        "r1":         round(r1, 4),
        "r2":         round(r2, 4),
        "s1":         round(s1, 4),
        "s2":         round(s2, 4),
        "cpr_width":  round(cpr_top - cpr_bottom, 4),
    }


def add_cpr_to_intraday(df_3m, daily_df):
    daily = daily_df.copy()
    daily.columns = [c.lower() for c in daily.columns]

    # Build date -> (high, low, close) map using timezone-agnostic date objects
    daily_by_date = {}
    for idx, row in daily.iterrows():
        d = pd.Timestamp(idx).date()
        daily_by_date[d] = (float(row["high"]), float(row["low"]), float(row["close"]))

    sorted_daily_dates = sorted(daily_by_date.keys())

    # Unique trading dates in intraday data
    intraday_dates = sorted(set(pd.Timestamp(idx).date() for idx in df_3m.index))

    # For each trading date, compute CPR from the prior session's daily bar
    cpr_by_trade_date = {}
    for trade_date in intraday_dates:
        prior_dates = [d for d in sorted_daily_dates if d < trade_date]
        if not prior_dates:
            continue
        h, l, c = daily_by_date[prior_dates[-1]]
        cpr_by_trade_date[trade_date] = compute_cpr(h, l, c)

    # Stamp CPR levels onto every 3-min bar
    df_out = df_3m.copy()
    bar_dates = [pd.Timestamp(idx).date() for idx in df_3m.index]
    cpr_cols = ["pp", "cpr_top", "cpr_bottom", "r1", "r2", "s1", "s2", "cpr_width"]
    for col in cpr_cols:
        df_out[col] = [cpr_by_trade_date.get(d, {}).get(col, float("nan")) for d in bar_dates]

    dropped = len(df_out) - len(df_out.dropna(subset=["pp"]))
    if dropped:
        print(f"  Dropped {dropped} bars with no prior-day CPR data.")
    return df_out.dropna(subset=["pp"])
