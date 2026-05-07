import joblib
import pandas as pd
from datetime import date

from data_loader import load_data
from features import build_features


def predict_today(model_path="model.pkl", data_path="data.csv", symbol="TSLA"):
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    df = load_data(data_path)
    df = build_features(df)

    # Use the most recent trading day in the file
    last_date = df.index[-1].date()
    mask = [idx.date() == last_date for idx in df.index]
    today_df = df[mask]

    if today_df.empty:
        print("No data found for the most recent trading day.")
        return []

    X_today = today_df[feature_cols]
    proba = model.predict_proba(X_today)
    preds = model.predict(X_today)

    signals = []
    for i, (idx, row) in enumerate(today_df.iterrows()):
        direction = "LONG" if preds[i] == 1 else "SHORT"
        conf = proba[i][preds[i]]
        entry = row["close"]

        if direction == "LONG":
            stop = round(entry * 0.995, 2)
            target = round(entry * 1.01, 2)
        else:
            stop = round(entry * 1.005, 2)
            target = round(entry * 0.99, 2)

        signals.append({
            "time": idx.strftime("%H:%M"),
            "direction": direction,
            "entry": round(entry, 2),
            "stop": stop,
            "target": target,
            "confidence": conf,
        })

    return signals


def print_signals(signals, symbol="TSLA", min_conf=0.60):
    if not signals:
        print("No signals generated.")
        return

    last_date = date.today()
    sep = "=" * 67

    print(f"\n{sep}")
    print(f"  ML TRADE SIGNALS - {symbol}  |  {last_date}")
    print(sep)
    print(f"{'Time':<7} {'Dir':<7} {'Entry':>8} {'Stop':>8} {'Target':>8} {'Conf':>7}  {'':>3}")
    print("-" * 67)

    high_conf = []
    for s in signals:
        flag = " **" if s["confidence"] >= min_conf else ""
        print(
            f"{s['time']:<7} {s['direction']:<7} "
            f"${s['entry']:>7.2f} ${s['stop']:>7.2f} ${s['target']:>7.2f} "
            f"{s['confidence']:>6.1%}{flag}"
        )
        if s["confidence"] >= min_conf:
            high_conf.append(s)

    print("-" * 67)
    print(f"Total bars: {len(signals)}   High-confidence (>={min_conf:.0%}): {len(high_conf)}")

    if high_conf:
        print(f"\n  HIGH-CONFIDENCE SIGNALS (>={min_conf:.0%})")
        print(f"  {'Time':<7} {'Dir':<7} {'Entry':>8} {'Stop':>8} {'Target':>8} {'Conf':>7}")
        print(f"  {'-'*57}")
        for s in high_conf:
            print(
                f"  {s['time']:<7} {s['direction']:<7} "
                f"${s['entry']:>7.2f} ${s['stop']:>7.2f} ${s['target']:>7.2f} "
                f"{s['confidence']:>6.1%}"
            )

    print(f"{sep}\n")
