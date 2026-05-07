import numpy as np


def create_cpr_labels(df, horizon=4):
    """
    CPR-target labels — encodes the actual trade objective.

    Signal type   | Condition                    | Target (must reach within horizon bars)
    --------------|------------------------------|----------------------------------------
    LONG          | below_cpr or cpr_bot_reclaim | cpr_top (reclaim) / cpr_bottom (position)
    SHORT         | above_cpr or cpr_top_reject  | cpr_bottom (reject) / cpr_top (position)
    BKT-LONG      | cpr_top_breakout             | R1  (price broke out above CPR, heading to R1)
    BKT-SHORT     | cpr_bot_breakout             | S1  (price broke down below CPR, heading to S1)

    inside_cpr-only bars that don't trigger a breakout: excluded.
    """
    n           = len(df)
    closes      = df["close"].values
    cpr_tops    = df["cpr_top"].values
    cpr_bottoms = df["cpr_bottom"].values
    r1s         = df["r1"].values
    s1s         = df["s1"].values
    below       = df["below_cpr"].values.astype(int)
    above       = df["above_cpr"].values.astype(int)
    reclaim     = df["cpr_bottom_reclaim"].values.astype(int)
    reject      = df["cpr_top_reject"].values.astype(int)
    bkt_up      = df["cpr_top_breakout"].values.astype(int) if "cpr_top_breakout" in df else np.zeros(n, int)
    bkt_dn      = df["cpr_bot_breakout"].values.astype(int) if "cpr_bot_breakout" in df else np.zeros(n, int)

    labels = np.full(n, -1, dtype=int)
    keep   = np.zeros(n, dtype=bool)

    for i in range(n - horizon):
        is_long  = bool(below[i] or reclaim[i] or bkt_up[i])
        is_short = bool(above[i] or reject[i]  or bkt_dn[i])

        if not is_long and not is_short:
            continue

        future = closes[i + 1: i + 1 + horizon]

        if is_long:
            if bkt_up[i]:
                target = r1s[i]           # breakout above CPR → target R1
            elif reclaim[i]:
                target = cpr_tops[i]      # reclaim bar → full CPR traverse
            else:
                target = cpr_bottoms[i]   # position below CPR → reach zone bottom
            labels[i] = 1 if np.any(future >= target) else 0
        else:
            if bkt_dn[i]:
                target = s1s[i]           # breakdown below CPR → target S1
            elif reject[i]:
                target = cpr_bottoms[i]   # reject bar → full CPR traverse
            else:
                target = cpr_tops[i]      # position above CPR → reach zone top
            labels[i] = 1 if np.any(future <= target) else 0

        keep[i] = True

    df = df.copy()
    df["label"] = labels
    df = df[keep].copy()
    df["label"] = df["label"].astype(int)
    return df
