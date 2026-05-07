"""
CPR Backtest  —  out-of-sample holdout.

Split strategy
  Train : first TRAIN_DAYS trading days  (~9 months at 200 days)
  Test  : all remaining trading days     (~3 months or more)

The model is trained ONLY on the train split, then scored on unseen test data.
This avoids the in-sample inflation seen when the live model (trained on all data)
is used to score its own training period.
"""

import pandas as pd
from tabulate import tabulate   # pip install tabulate

from data_loader import load_data
from features import build_features
from labels import create_cpr_labels
from model import train_long_short


_OHLC    = ["open", "high", "low", "close", "volume"]
_RAW_CPR = ["pp", "cpr_top", "cpr_bottom", "r1", "r2", "s1", "s2", "cpr_width"]
_EXCLUDE = set(["label"] + _OHLC + _RAW_CPR)

TRAIN_DAYS = 200   # ~9 months of trading days for training


def _is_long(row):
    return (row.get("below_cpr",         0) == 1 or
            row.get("cpr_bottom_reclaim", 0) == 1 or
            row.get("cpr_top_breakout",   0) == 1)


def _is_short(row):
    return (row.get("above_cpr",        0) == 1 or
            row.get("cpr_top_reject",   0) == 1 or
            row.get("cpr_bot_breakout", 0) == 1)


def run_backtest(data_path="data.csv"):
    df         = load_data(data_path)
    df         = build_features(df)
    df_labeled = create_cpr_labels(df.copy())

    all_dates = sorted(set(pd.Timestamp(idx).date() for idx in df_labeled.index))

    if len(all_dates) <= TRAIN_DAYS:
        print(f"Need more than {TRAIN_DAYS} trading days — got {len(all_dates)}.")
        return []

    train_cutoff = all_dates[TRAIN_DAYS - 1]
    test_dates   = all_dates[TRAIN_DAYS:]

    print(f"Train : {all_dates[0]}  to  {train_cutoff}  ({TRAIN_DAYS} days)")
    print(f"Test  : {test_dates[0]}  to  {test_dates[-1]}  ({len(test_dates)} days)")

    # ---- Train models on train split only --------------------------------
    train_mask = [pd.Timestamp(idx).date() <= train_cutoff for idx in df_labeled.index]
    train_df   = df_labeled[train_mask]

    feat_cols  = [c for c in train_df.columns if c not in _EXCLUDE]
    X_train    = train_df[feat_cols]

    long_mask  = (train_df.get("below_cpr", 0) == 1) | (train_df.get("cpr_bottom_reclaim", 0) == 1)
    short_mask = (train_df.get("above_cpr",  0) == 1) | (train_df.get("cpr_top_reject",    0) == 1)

    long_model, short_model = train_long_short(
        X_train[long_mask],  train_df["label"][long_mask],
        X_train[short_mask], train_df["label"][short_mask],
    )
    print(f"Trained on {long_mask.sum()} LONG + {short_mask.sum()} SHORT setup bars\n")

    # ---- Score test split -----------------------------------------------
    signals = []

    for trade_date in test_dates:
        mask   = [pd.Timestamp(idx).date() == trade_date for idx in df_labeled.index]
        day_df = df_labeled[mask]
        if day_df.empty:
            continue

        setup_rows    = [(idx, row) for idx, row in day_df.iterrows()
                         if _is_long(row) or _is_short(row)]
        long_indices  = [idx for idx, row in setup_rows if _is_long(row)]
        short_indices = [idx for idx, row in setup_rows if _is_short(row)]

        long_proba  = {}
        short_proba = {}
        if long_indices:
            p = long_model.predict_proba(day_df.loc[long_indices, feat_cols])
            long_proba = {idx: p[i][1] for i, idx in enumerate(long_indices)}
        if short_indices:
            p = short_model.predict_proba(day_df.loc[short_indices, feat_cols])
            short_proba = {idx: p[i][1] for i, idx in enumerate(short_indices)}

        for idx, row in setup_rows:
            direction  = "LONG" if _is_long(row) else "SHORT"
            entry      = round(float(row["close"]), 2)
            cpr_top    = round(float(row["cpr_top"]), 2)
            cpr_bot    = round(float(row["cpr_bottom"]), 2)
            r1         = round(float(row["r1"]), 2)
            s1         = round(float(row["s1"]), 2)
            is_reclaim = row.get("cpr_bottom_reclaim", 0) == 1
            is_reject  = row.get("cpr_top_reject",     0) == 1
            is_bkt_up  = row.get("cpr_top_breakout",   0) == 1
            is_bkt_dn  = row.get("cpr_bot_breakout",   0) == 1

            if direction == "LONG":
                stop   = round(entry * 0.995, 2)
                target = r1 if is_bkt_up else (cpr_top if is_reclaim else cpr_bot)
                conf   = long_proba.get(idx, 0.0)
            else:
                stop   = round(entry * 1.005, 2)
                target = s1 if is_bkt_dn else (cpr_bot if is_reject else cpr_top)
                conf   = short_proba.get(idx, 0.0)

            risk   = abs(entry - stop)
            reward = abs(target - entry)
            rr     = round(reward / risk, 1) if risk > 0 else 0.0

            signals.append({
                "date":       trade_date,
                "time":       pd.Timestamp(idx).strftime("%H:%M"),
                "direction":  direction,
                "entry":      entry,
                "stop":       stop,
                "target":     target,
                "rr":         rr,
                "confidence": conf,
                "hit":        int(row.get("label", -1)),
            })

    return signals


# ---- Reporting -----------------------------------------------------------

def _stats(trades):
    if not trades:
        return dict(n=0, wins=0, wr=0, ev=0, total_r=0, active=0)
    n      = len(trades)
    wins   = sum(1 for t in trades if t["hit"] == 1)
    ev     = sum(t["rr"] if t["hit"] == 1 else -1.0 for t in trades) / n
    total  = sum(t["rr"] if t["hit"] == 1 else -1.0 for t in trades)
    dates  = len(set(t["date"] for t in trades))
    return dict(n=n, wins=wins, wr=wins/n, ev=ev, total_r=total, active=dates)


def print_threshold_sweep(signals, test_days):
    """Show how win-rate and active days change at each confidence cut-off."""
    df  = pd.DataFrame(signals)
    sep = "=" * 80

    print(f"\n{sep}")
    print(f"  THRESHOLD SWEEP  |  test period: {test_days} trading days  "
          f"(out-of-sample)")
    print(sep)

    rows = []
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        subset = [s for s in signals if s["confidence"] >= thresh]
        st     = _stats(subset)
        rows.append([
            f">={thresh:.0%}",
            st["n"],
            f"{st['active']}/{test_days}",
            f"{st['active']/test_days:.0%}",
            f"{st['wins']}",
            f"{st['wr']:.1%}" if st["n"] else "-",
            f"{st['ev']:+.2f}R" if st["n"] else "-",
            f"{st['total_r']:+.1f}R" if st["n"] else "-",
        ])

    print(tabulate(rows,
                   headers=["Threshold", "Trades", "Active days", "Day%",
                             "Wins", "Win%", "Avg EV", "Total R"],
                   tablefmt="simple"))
    print()


def print_daily_log(signals, min_conf=0.55, min_rr=1.0):
    """Per-day table and full trade log at chosen threshold + R:R filter."""
    if not signals:
        print("No signals.")
        return

    df  = pd.DataFrame(signals)
    sep = "=" * 82

    flagged = [s for s in signals
               if s["confidence"] >= min_conf and s["rr"] >= min_rr]

    print(f"\n{sep}")
    print(f"  DAILY BREAKDOWN  |  conf >= {min_conf:.0%}  AND  R:R >= {min_rr:.1f}x"
          f"  (out-of-sample)")
    print(sep)
    print(f"  {'Date':<12} {'Setups':>7} {'Flagged':>8} {'L/S':>6} "
          f"{'Wins':>6} {'Win%':>6}  {'Avg EV':>8}")
    print(f"  {'-'*68}")

    for d in sorted(df["date"].unique()):
        day_all  = [s for s in signals if s["date"] == d]
        day_flag = [s for s in flagged if s["date"] == d]
        n_flag   = len(day_flag)
        n_long   = sum(1 for s in day_flag if s["direction"] == "LONG")
        n_short  = sum(1 for s in day_flag if s["direction"] == "SHORT")
        wins     = sum(1 for s in day_flag if s["hit"] == 1)
        win_pct  = f"{wins/n_flag:.0%}" if n_flag else "  -"
        if n_flag:
            ev = sum(s["rr"] if s["hit"] == 1 else -1.0 for s in day_flag) / n_flag
            ev_str = f"{ev:+.2f}R"
        else:
            ev_str = "   -"
        print(f"  {str(d):<12} {len(day_all):>7} {n_flag:>8} "
              f"{n_long}L/{n_short}S{'':<2} {wins:>6} {win_pct:>6}  {ev_str:>8}")

    print(f"  {'-'*68}")

    st = _stats(flagged)
    long_f  = [s for s in flagged if s["direction"] == "LONG"]
    short_f = [s for s in flagged if s["direction"] == "SHORT"]
    lwr = sum(1 for s in long_f  if s["hit"]==1)/len(long_f)  if long_f  else 0
    swr = sum(1 for s in short_f if s["hit"]==1)/len(short_f) if short_f else 0

    print(f"\n  Total setup bars    : {len(signals)}")
    print(f"  Qualifying trades   : {st['n']}  (LONG: {len(long_f)}  SHORT: {len(short_f)})")
    print(f"  Active trading days : {st['active']}/{df['date'].nunique()}")
    print(f"  Win rate            : {st['wr']:.1%}  "
          f"(LONG: {lwr:.1%}  SHORT: {swr:.1%})")
    print(f"  Avg EV per trade    : {st['ev']:+.2f}R")
    print(f"  Total P&L           : {st['total_r']:+.1f}R over {st['n']} trades")

    # Detailed trade log
    if flagged:
        print(f"\n  TRADE LOG  (conf >= {min_conf:.0%}, R:R >= {min_rr:.1f}x)")
        print(f"  {'Date':<12} {'Time':<6} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
              f"{'Target':>8} {'R:R':>5} {'Conf':>7} {'Result'}")
        print(f"  {'-'*74}")
        for s in sorted(flagged, key=lambda x: (x["date"], x["time"])):
            result = "HIT " if s["hit"] == 1 else "MISS"
            print(
                f"  {str(s['date']):<12} {s['time']:<6} {s['direction']:<6} "
                f"${s['entry']:>7.2f} ${s['stop']:>7.2f} ${s['target']:>7.2f} "
                f"{s['rr']:>4.1f}x {s['confidence']:>6.1%}  {result}"
            )
    print(f"\n{sep}\n")


if __name__ == "__main__":
    print("Loading and processing data...")
    sigs = run_backtest(data_path="data.csv")

    if sigs:
        test_days = len(set(s["date"] for s in sigs))
        print_threshold_sweep(sigs, test_days)
        # Show best threshold results: 55% confidence + 1.0x R:R
        print_daily_log(sigs, min_conf=0.55, min_rr=1.0)
