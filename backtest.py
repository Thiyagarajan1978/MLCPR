import joblib
import pandas as pd

from data_loader import load_data
from features import build_features
from labels import create_cpr_labels


_OHLC    = ["open", "high", "low", "close", "volume"]
_RAW_CPR = ["pp", "cpr_top", "cpr_bottom", "r1", "r2", "s1", "s2", "cpr_width"]


def _is_long(row):
    return row.get("below_cpr", 0) == 1 or row.get("cpr_bottom_reclaim", 0) == 1


def _is_short(row):
    return row.get("above_cpr", 0) == 1 or row.get("cpr_top_reject", 0) == 1


def run_backtest(model_path="model.pkl", data_path="data.csv", days=21, min_conf=0.60):
    payload      = joblib.load(model_path)
    long_model   = payload["long_model"]
    short_model  = payload["short_model"]
    feature_cols = payload["feature_cols"]

    df         = load_data(data_path)
    df         = build_features(df)
    df_labeled = create_cpr_labels(df.copy())   # adds label, drops inside-CPR bars

    all_dates      = sorted(set(pd.Timestamp(idx).date() for idx in df_labeled.index))
    backtest_dates = all_dates[-days:]

    signals = []

    for trade_date in backtest_dates:
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
            p = long_model.predict_proba(day_df.loc[long_indices, feature_cols])
            long_proba = {idx: p[i][1] for i, idx in enumerate(long_indices)}
        if short_indices:
            p = short_model.predict_proba(day_df.loc[short_indices, feature_cols])
            short_proba = {idx: p[i][1] for i, idx in enumerate(short_indices)}

        for idx, row in setup_rows:
            direction  = "LONG" if _is_long(row) else "SHORT"
            entry      = round(float(row["close"]), 2)
            cpr_top    = round(float(row["cpr_top"]), 2)
            cpr_bot    = round(float(row["cpr_bottom"]), 2)
            is_reclaim = row.get("cpr_bottom_reclaim", 0) == 1
            is_reject  = row.get("cpr_top_reject",     0) == 1

            if direction == "LONG":
                stop   = round(entry * 0.995, 2)
                target = cpr_top if is_reclaim else cpr_bot
                conf   = long_proba.get(idx, 0.0)
            else:
                stop   = round(entry * 1.005, 2)
                target = cpr_bot if is_reject else cpr_top
                conf   = short_proba.get(idx, 0.0)

            risk   = abs(entry - stop)
            reward = abs(target - entry)
            rr     = round(reward / risk, 1) if risk > 0 else 0.0
            hit    = int(row.get("label", -1))   # 1 = target reached within 10 bars

            signals.append({
                "date":       trade_date,
                "time":       pd.Timestamp(idx).strftime("%H:%M"),
                "direction":  direction,
                "entry":      entry,
                "stop":       stop,
                "target":     target,
                "rr":         rr,
                "confidence": conf,
                "hit":        hit,
                "flagged":    conf >= min_conf,
            })

    return signals


def print_backtest(signals, min_conf=0.60, days=21):
    if not signals:
        print("No signals in backtest window.")
        return

    df  = pd.DataFrame(signals)
    sep = "=" * 82

    print(f"\n{sep}")
    print(f"  PHASE C  CPR BACKTEST  |  Last {df['date'].nunique()} trading days  "
          f"|  min_conf >= {min_conf:.0%}")
    print(f"  hit = price reached CPR structural target within 10 bars (50 min)")
    print(sep)

    # ---- Per-day summary table ----
    print(f"\n  {'Date':<12} {'Setups':>7} {'Flagged':>8} {'L/S':>6} "
          f"{'Wins':>6} {'Win%':>6}  {'Avg EV':>8}")
    print(f"  {'-'*70}")

    for d in sorted(df["date"].unique()):
        day      = df[df["date"] == d]
        day_flag = day[day["flagged"]]
        n_flag   = len(day_flag)
        n_long   = int((day_flag["direction"] == "LONG").sum())
        n_short  = int((day_flag["direction"] == "SHORT").sum())
        ls_str   = f"{n_long}L/{n_short}S"
        wins     = int(day_flag["hit"].sum()) if n_flag > 0 else 0
        win_pct  = f"{wins/n_flag:.0%}" if n_flag > 0 else "  -"
        if n_flag > 0:
            ev = sum(row["rr"] if row["hit"] == 1 else -1.0
                     for _, row in day_flag.iterrows()) / n_flag
            ev_str = f"{ev:+.2f}R"
        else:
            ev_str = "   -"
        print(f"  {str(d):<12} {len(day):>7} {n_flag:>8} {ls_str:>6} "
              f"{wins:>6} {win_pct:>6}  {ev_str:>8}")

    print(f"  {'-'*70}")

    # ---- Overall stats ----
    flagged    = df[df["flagged"]]
    n_total    = len(flagged)
    n_wins     = int(flagged["hit"].sum())
    win_rate   = n_wins / n_total if n_total > 0 else 0.0
    ev_avg     = (sum(r["rr"] if r["hit"] == 1 else -1.0
                      for _, r in flagged.iterrows()) / n_total
                  if n_total > 0 else 0.0)

    long_f  = flagged[flagged["direction"] == "LONG"]
    short_f = flagged[flagged["direction"] == "SHORT"]
    long_wr  = long_f["hit"].mean()  if len(long_f)  > 0 else 0.0
    short_wr = short_f["hit"].mean() if len(short_f) > 0 else 0.0

    print(f"\n  Total setup bars scored   : {len(df)}")
    print(f"  High-confidence (>={min_conf:.0%}) : {n_total}  "
          f"(LONG: {len(long_f)}  SHORT: {len(short_f)})")
    print(f"  Win rate  : {win_rate:.1%}  "
          f"(LONG: {long_wr:.1%}  SHORT: {short_wr:.1%})")
    print(f"  Avg EV    : {ev_avg:+.2f}R  "
          f"({'positive edge' if ev_avg > 0 else 'no edge at this threshold'})")
    print(f"  Total P&L : "
          f"{sum(r['rr'] if r['hit']==1 else -1.0 for _, r in flagged.iterrows()):+.1f}R "
          f"across {n_total} trades")

    # ---- Detailed trade log ----
    if not flagged.empty:
        print(f"\n  DETAILED TRADE LOG  (high-confidence signals only)")
        print(f"  {'Date':<12} {'Time':<6} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
              f"{'Target':>8} {'R:R':>5} {'Conf':>7} {'Result'}")
        print(f"  {'-'*76}")
        for _, row in flagged.sort_values(["date", "time"]).iterrows():
            result = "HIT " if row["hit"] == 1 else "MISS"
            print(
                f"  {str(row['date']):<12} {row['time']:<6} {row['direction']:<6} "
                f"${row['entry']:>7.2f} ${row['stop']:>7.2f} ${row['target']:>7.2f} "
                f"{row['rr']:>4.1f}x {row['confidence']:>6.1%}  {result}"
            )

    print(f"\n{sep}\n")


def print_rr_filtered(signals, min_conf=0.60, min_rr=1.0):
    """Re-slice the same backtest results filtered to tradeable R:R setups."""
    df     = pd.DataFrame(signals)
    subset = df[(df["flagged"]) & (df["rr"] >= min_rr)]

    sep = "=" * 82
    print(f"\n{sep}")
    print(f"  HIGH QUALITY SETUPS  |  conf >= {min_conf:.0%}  AND  R:R >= {min_rr:.1f}x")
    print(sep)

    if subset.empty:
        print("  No setups meet both filters.")
        print(f"{sep}\n")
        return

    n       = len(subset)
    n_wins  = int(subset["hit"].sum())
    wr      = n_wins / n
    ev      = sum(r["rr"] if r["hit"] == 1 else -1.0 for _, r in subset.iterrows()) / n
    total_r = sum(r["rr"] if r["hit"] == 1 else -1.0 for _, r in subset.iterrows())

    print(f"  {'Date':<12} {'Time':<6} {'Dir':<6} {'Entry':>8} {'Stop':>8} "
          f"{'Target':>8} {'R:R':>5} {'Conf':>7} {'Result'}")
    print(f"  {'-'*76}")
    for _, row in subset.sort_values(["date", "time"]).iterrows():
        result = "HIT " if row["hit"] == 1 else "MISS"
        print(
            f"  {str(row['date']):<12} {row['time']:<6} {row['direction']:<6} "
            f"${row['entry']:>7.2f} ${row['stop']:>7.2f} ${row['target']:>7.2f} "
            f"{row['rr']:>4.1f}x {row['confidence']:>6.1%}  {result}"
        )
    print(f"  {'-'*76}")
    print(f"  Trades : {n}  |  Wins : {n_wins}  |  Win rate : {wr:.1%}  "
          f"|  Avg EV : {ev:+.2f}R  |  Total : {total_r:+.1f}R")
    print(f"{sep}\n")


if __name__ == "__main__":
    signals = run_backtest(days=21, min_conf=0.60)
    print_backtest(signals, min_conf=0.60, days=21)

    print("NOTE: win rate above is IN-SAMPLE (model trained on same 60 days).")
    print("      Honest out-of-sample accuracy = 79.8% (walk-forward validation).")
    print("      Many signals have R:R < 1x (target barely above/below entry).")
    print("      Filtered view below shows only setups with R:R >= 1.0x.\n")

    print_rr_filtered(signals, min_conf=0.60, min_rr=1.0)
