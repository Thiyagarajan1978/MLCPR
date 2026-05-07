import pandas as pd


def build_features(df):
    # --- Basic price structure ---
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["trend"] = df["close"].diff()
    df["volatility"] = df["range"].rolling(10).mean()

    # --- Volume context ---
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(10).mean().replace(0, 1)
        df["vol_ratio"] = df["volume"] / vol_ma

    # --- CPR-relative features (the real edge) ---
    if "cpr_bottom" in df.columns and "cpr_top" in df.columns:
        pp = df["pp"]

        # Distance from key levels, normalised by PP so they're scale-invariant
        df["dist_cpr_bottom"] = (df["close"] - df["cpr_bottom"]) / pp
        df["dist_cpr_top"]    = (df["close"] - df["cpr_top"])    / pp
        df["dist_pp"]         = (df["close"] - pp)               / pp
        df["dist_r1"]         = (df["close"] - df["r1"])         / pp
        df["dist_s1"]         = (df["close"] - df["s1"])         / pp

        # Narrow vs wide CPR day (key for regime detection)
        df["cpr_width_pct"] = df["cpr_width"] / pp

        # Where is price relative to CPR right now?
        df["above_cpr"]  = (df["close"] > df["cpr_top"]).astype(int)
        df["inside_cpr"] = ((df["close"] >= df["cpr_bottom"]) &
                            (df["close"] <= df["cpr_top"])).astype(int)
        df["below_cpr"]  = (df["close"] < df["cpr_bottom"]).astype(int)

        # CPR cross signals (the reclaim / reject events)
        prev = df["close"].shift(1)
        df["cpr_bottom_reclaim"] = ((prev < df["cpr_bottom"]) &
                                    (df["close"] >= df["cpr_bottom"])).astype(int)
        df["cpr_top_reclaim"]    = ((prev < df["cpr_top"]) &
                                    (df["close"] >= df["cpr_top"])).astype(int)
        df["cpr_bottom_reject"]  = ((prev > df["cpr_bottom"]) &
                                    (df["close"] < df["cpr_bottom"])).astype(int)
        df["cpr_top_reject"]     = ((prev > df["cpr_top"]) &
                                    (df["close"] < df["cpr_top"])).astype(int)

        # Phase D — expanded signal universe
        # CPR breakout: price exits CPR from inside (prev bar was inside CPR)
        # These catch "inside-CPR" days that previously had zero signals.
        # Target: R1 (up breakout) or S1 (down breakout) — the next pivot destination.
        prev_inside = ((prev >= df["cpr_bottom"]) & (prev <= df["cpr_top"]))
        df["cpr_top_breakout"] = (prev_inside & (df["close"] > df["cpr_top"])).astype(int)
        df["cpr_bot_breakout"] = (prev_inside & (df["close"] < df["cpr_bottom"])).astype(int)

        # S1/R1 mean-reversion flags: help model learn that S1 bounces and R1 rejects
        # carry stronger momentum than ordinary below/above CPR bars.
        df["s1_bounce"] = ((prev < df["s1"]) & (df["close"] >= df["s1"])).astype(int)
        df["r1_reject"]  = ((prev > df["r1"]) & (df["close"] <= df["r1"])).astype(int)

        # Raw pivot price columns are kept here — labels.py needs cpr_top/cpr_bottom/r1/s1
        # to compute CPR-target labels. main.py and walk_forward.py exclude them from X.

    # --- Phase B: Time-of-day and session regime features ---
    timestamps = pd.DatetimeIndex(df.index)

    # bars_since_open: hourly bars elapsed since 09:30
    # Regular session = 7 bars (09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30)
    session_start_minutes = 9 * 60 + 30   # 570 minutes from midnight
    # Use pd.Series so .clip() and boolean ops work correctly
    bar_minutes = pd.Series(
        timestamps.hour * 60 + timestamps.minute, index=df.index, dtype=int
    )
    df["bars_since_open"] = ((bar_minutes - session_start_minutes) // 60).clip(0, 6)

    session_total_bars = 7
    df["time_of_day"] = (df["bars_since_open"] / (session_total_bars - 1)).clip(0.0, 1.0)

    # Binary session-window flags
    df["first_hour"] = (bar_minutes == 570).astype(int)   # 09:30 bar only
    df["last_hour"]  = (bar_minutes >= 870).astype(int)   # 14:30 and 15:30 bars

    # session_gap_pct: (today open - yesterday close) / yesterday close
    # Positive = gap up (bullish regime), Negative = gap down (bearish regime)
    daily_open     = df.groupby(timestamps.date)["open"].transform("first")
    prev_day_close = df["close"].groupby(timestamps.date).last().shift(1)
    date_to_prev_close = prev_day_close.to_dict()
    df["session_gap_pct"] = [
        (daily_open.iloc[i] - date_to_prev_close.get(timestamps[i].date(), float("nan")))
        / date_to_prev_close.get(timestamps[i].date(), float("nan"))
        if timestamps[i].date() in date_to_prev_close else float("nan")
        for i in range(len(df))
    ]

    # --- Phase C: Prior-day regime features ---
    # All backward-looking (prior session data only) — no leakage.
    # Teaches the model that SHORT setups only work in bearish-regime days.
    if "cpr_bottom" in df.columns and "cpr_top" in df.columns:
        ts           = pd.DatetimeIndex(df.index)
        unique_dates = sorted(set(t.date() for t in ts))

        # Build per-day summary from intraday bars
        daily_info = {}
        for d in unique_dates:
            day_mask = [t.date() == d for t in ts]
            day_df   = df[day_mask]
            daily_info[d] = {
                "last_close":    float(day_df["close"].iloc[-1]),
                "pp":            float(day_df["pp"].iloc[0]),
                "cpr_top":       float(day_df["cpr_top"].iloc[0]),
                "cpr_bottom":    float(day_df["cpr_bottom"].iloc[0]),
                "pct_above_cpr": float((day_df["close"] > day_df["cpr_top"]).mean()),
                "pct_below_cpr": float((day_df["close"] < day_df["cpr_bottom"]).mean()),
            }

        # Derive features relative to prior session
        regime_by_date = {}
        for i, d in enumerate(unique_dates):
            if i == 0:
                continue
            prev = daily_info[unique_dates[i - 1]]
            curr = daily_info[d]
            # Where did prior day close vs its own CPR?
            if prev["last_close"] > prev["cpr_top"]:
                regime = 1.0    # bullish close (price above CPR)
            elif prev["last_close"] < prev["cpr_bottom"]:
                regime = -1.0   # bearish close (price below CPR)
            else:
                regime = 0.0    # neutral (closed inside CPR band)

            regime_by_date[d] = {
                "prior_day_regime":    regime,
                "pp_rising":           1.0 if curr["pp"] > prev["pp"] else 0.0,
                "prior_pct_above_cpr": prev["pct_above_cpr"],
                "prior_pct_below_cpr": prev["pct_below_cpr"],
            }

        bar_dates_c = [t.date() for t in ts]
        for col in ["prior_day_regime", "pp_rising",
                    "prior_pct_above_cpr", "prior_pct_below_cpr"]:
            df[col] = [regime_by_date.get(d, {}).get(col, float("nan")) for d in bar_dates_c]

    return df.dropna()
