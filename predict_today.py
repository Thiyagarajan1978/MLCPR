import joblib
import pandas as pd
from datetime import date

from data_loader import load_data
from features import build_features


def _is_long(row):
    return (row.get("below_cpr",          0) == 1 or
            row.get("cpr_bottom_reclaim",  0) == 1 or
            row.get("cpr_top_breakout",    0) == 1)


def _is_short(row):
    return (row.get("above_cpr",           0) == 1 or
            row.get("cpr_top_reject",       0) == 1 or
            row.get("cpr_bot_breakout",     0) == 1)


def _signal_type(row):
    if row.get("cpr_top_breakout", 0) == 1:
        return "BKT-L"
    if row.get("cpr_bot_breakout", 0) == 1:
        return "BKT-S"
    if row.get("cpr_bottom_reclaim", 0) == 1:
        return "RCLM"
    if row.get("cpr_top_reject", 0) == 1:
        return "RJCT"
    if row.get("below_cpr", 0) == 1:
        return "BLW"
    if row.get("above_cpr", 0) == 1:
        return "ABV"
    return "?"


def predict_today(model_path="model.pkl", data_path="data.csv", symbol="TSLA"):
    payload      = joblib.load(model_path)
    long_model   = payload["long_model"]
    short_model  = payload["short_model"]
    feature_cols = payload["feature_cols"]

    df = load_data(data_path)
    df = build_features(df)

    last_date = df.index[-1].date()
    mask      = [idx.date() == last_date for idx in df.index]
    today_df  = df[mask]

    if today_df.empty:
        print("No data found for the most recent trading day.")
        return []

    setup_rows = [(idx, row) for idx, row in today_df.iterrows()
                  if _is_long(row) or _is_short(row)]

    if not setup_rows:
        print("No CPR trade setups found for today (price stayed inside CPR all day).")
        return []

    long_indices  = [idx for idx, row in setup_rows if _is_long(row)]
    short_indices = [idx for idx, row in setup_rows if _is_short(row)]

    long_proba  = {}
    short_proba = {}
    if long_indices:
        p = long_model.predict_proba(today_df.loc[long_indices, feature_cols])
        long_proba = {idx: p[i][1] for i, idx in enumerate(long_indices)}
    if short_indices:
        p = short_model.predict_proba(today_df.loc[short_indices, feature_cols])
        short_proba = {idx: p[i][1] for i, idx in enumerate(short_indices)}

    signals = []
    for idx, row in setup_rows:
        entry     = round(float(row["close"]), 2)
        cpr_top   = round(float(row["cpr_top"]), 2)
        cpr_bot   = round(float(row["cpr_bottom"]), 2)
        r1        = round(float(row["r1"]), 2)
        s1        = round(float(row["s1"]), 2)
        direction = "LONG" if _is_long(row) else "SHORT"
        sig_type  = _signal_type(row)

        is_reclaim    = row.get("cpr_bottom_reclaim", 0) == 1
        is_reject     = row.get("cpr_top_reject",     0) == 1
        is_bkt_up     = row.get("cpr_top_breakout",   0) == 1
        is_bkt_dn     = row.get("cpr_bot_breakout",   0) == 1

        if direction == "LONG":
            stop       = round(entry * 0.995, 2)
            target     = r1 if is_bkt_up else (cpr_top if is_reclaim else cpr_bot)
            confidence = long_proba.get(idx, 0.0)
        else:
            stop       = round(entry * 1.005, 2)
            target     = s1 if is_bkt_dn else (cpr_bot if is_reject else cpr_top)
            confidence = short_proba.get(idx, 0.0)

        risk   = abs(entry - stop)
        reward = abs(target - entry)
        rr     = round(reward / risk, 1) if risk > 0 else 0

        signals.append({
            "time":       idx.strftime("%H:%M"),
            "type":       sig_type,
            "direction":  direction,
            "entry":      entry,
            "stop":       stop,
            "target":     target,
            "rr":         rr,
            "confidence": confidence,
        })

    return signals


def print_signals(signals, symbol="TSLA", min_conf=0.50):
    if not signals:
        print("No CPR trade signals for today.")
        return

    today = date.today()
    sep   = "=" * 80
    high  = [s for s in signals if s["confidence"] >= min_conf]

    print(f"\n{sep}")
    print(f"  PHASE D  CPR TRADE SIGNALS (1h) - {symbol}  |  {today}")
    print(f"  Confidence = P(price reaches CPR/pivot target within 4 hourly bars)")
    print(f"  Types: BLW=below CPR  ABV=above CPR  RCLM=reclaim  RJCT=reject")
    print(f"         BKT-L=CPR breakout long  BKT-S=CPR breakdown short")
    print(sep)
    print(f"{'Time':<7} {'Type':<7} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
          f"{'Target':>8} {'R:R':>5} {'Conf':>7}")
    print("-" * 80)

    for s in signals:
        flag = " **" if s["confidence"] >= min_conf else ""
        print(
            f"{s['time']:<7} {s['type']:<7} {s['direction']:<6} "
            f"${s['entry']:>7.2f} ${s['stop']:>7.2f} ${s['target']:>7.2f} "
            f"{s['rr']:>4.1f}x {s['confidence']:>6.1%}{flag}"
        )

    print("-" * 80)
    print(f"Total setup bars : {len(signals)}")
    print(f"High-confidence (>={min_conf:.0%}): {len(high)}")

    if high:
        print(f"\n  BEST SETUPS (>={min_conf:.0%} confidence)")
        print(f"  {'Time':<7} {'Type':<7} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
              f"{'Target':>8} {'R:R':>5} {'Conf':>7}")
        print(f"  {'-'*66}")
        for s in high:
            print(
                f"  {s['time']:<7} {s['type']:<7} {s['direction']:<6} "
                f"${s['entry']:>7.2f} ${s['stop']:>7.2f} ${s['target']:>7.2f} "
                f"{s['rr']:>4.1f}x {s['confidence']:>6.1%}"
            )
    print(f"{sep}\n")
