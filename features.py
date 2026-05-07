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

        # Raw pivot price columns are kept here — labels.py needs cpr_top/cpr_bottom
        # to compute CPR-target labels. main.py and walk_forward.py exclude them from X.

    return df.dropna()
