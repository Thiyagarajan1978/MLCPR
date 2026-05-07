import numpy as np


def create_cpr_labels(df, horizon=10):
    """
    CPR-target labels — encodes the actual trade objective.

    LONG setup  (below_cpr=1 or cpr_bottom_reclaim=1):
        label = 1 if price reaches cpr_top within `horizon` bars, else 0.

    SHORT setup (above_cpr=1 or cpr_top_reject=1):
        label = 1 if price reaches cpr_bottom within `horizon` bars, else 0.

    inside_cpr only bars: excluded — no clear CPR trade setup.

    This replaces the naive "price up in 3 bars" label with the actual
    question: "does the CPR directional trade reach its structural target?"
    """
    n = len(df)
    closes      = df["close"].values
    cpr_tops    = df["cpr_top"].values
    cpr_bottoms = df["cpr_bottom"].values
    below       = df["below_cpr"].values.astype(int)
    above       = df["above_cpr"].values.astype(int)
    reclaim     = df["cpr_bottom_reclaim"].values.astype(int)
    reject      = df["cpr_top_reject"].values.astype(int)

    labels = np.full(n, -1, dtype=int)
    keep   = np.zeros(n, dtype=bool)

    for i in range(n - horizon):
        is_long  = bool(below[i] or reclaim[i])
        is_short = bool(above[i] or reject[i])

        if not is_long and not is_short:
            continue

        future = closes[i + 1: i + 1 + horizon]

        if is_long:
            # below_cpr: first target is reaching CPR Bottom (entry into the zone)
            # reclaim bar: already inside, target is CPR Top (full traverse)
            if reclaim[i]:
                target = cpr_tops[i]
            else:
                target = cpr_bottoms[i]
            labels[i] = 1 if np.any(future >= target) else 0
        else:
            # above_cpr: first target is reaching CPR Top (back into the zone)
            # reject bar: already inside, target is CPR Bottom (full traverse)
            if reject[i]:
                target = cpr_bottoms[i]
            else:
                target = cpr_tops[i]
            labels[i] = 1 if np.any(future <= target) else 0

        keep[i] = True

    df = df.copy()
    df["label"] = labels
    df = df[keep].copy()
    df["label"] = df["label"].astype(int)
    return df
