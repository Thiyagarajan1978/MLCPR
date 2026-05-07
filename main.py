from fetch_data import fetch_and_save
from data_loader import load_data
from features import build_features
from labels import create_cpr_labels
from model import train, train_long_short
from walk_forward import run_walk_forward
from save_model import save_long_short
from predict_today import predict_today, print_signals

SYMBOL = "TSLA"

# Raw columns that must NOT enter the feature matrix:
#   OHLC    — absolute price levels cause memorisation, not generalisation
#   RAW_CPR — absolute pivot levels; model sees normalised distances instead
OHLC    = ["open", "high", "low", "close", "volume"]
RAW_CPR = ["pp", "cpr_top", "cpr_bottom", "r1", "r2", "s1", "s2", "cpr_width"]

print(f"\nFetching {SYMBOL} data (last 60 days, 5-min bars)...")
fetch_and_save(symbol=SYMBOL, output="data.csv")

print("\nLoading data...")
df = load_data("data.csv")

print("Building features...")
df = build_features(df)

print("Creating CPR-target labels...")
df_labeled = create_cpr_labels(df.copy())

exclude    = ["label"] + [c for c in OHLC + RAW_CPR if c in df_labeled.columns]
X          = df_labeled.drop(columns=exclude)
y          = df_labeled["label"]

long_mask  = ((df_labeled.get("below_cpr",        0) == 1) |
              (df_labeled.get("cpr_bottom_reclaim", 0) == 1) |
              (df_labeled.get("cpr_top_breakout",   0) == 1))
short_mask = ((df_labeled.get("above_cpr",         0) == 1) |
              (df_labeled.get("cpr_top_reject",     0) == 1) |
              (df_labeled.get("cpr_bot_breakout",   0) == 1))
X_long,  y_long  = X[long_mask],  y[long_mask]
X_short, y_short = X[short_mask], y[short_mask]

hit_rate   = float(y.mean())
print(f"CPR setups : {len(X)} bars  (LONG: {len(X_long)}  SHORT: {len(X_short)})")
print(f"Base rate  : {hit_rate:.1%}  "
      f"(LONG: {y_long.mean():.1%}  SHORT: {y_short.mean():.1%})")
print(f"Features   : {len(X.columns)}  {list(X.columns)}")

print("\nTraining separate LONG / SHORT models (Phase C)...")
long_model, short_model = train_long_short(X_long, y_long, X_short, y_short)

print("Saving models...")
save_long_short(long_model, short_model, feature_cols=X.columns)

print("\nRunning walk-forward validation (combined model)...")
results = run_walk_forward(df_labeled)
if results:
    avg  = sum(results) / len(results)
    span = max(results) - min(results)
    pct  = [f"{r:.0%}" for r in results]
    print(f"Windows : {pct}")
    print(f"Average : {avg:.1%}  |  Spread: {span:.0%}  "
          f"({'stable' if span < 0.20 else 'high variance'})")
    print(f"Base rate was {hit_rate:.1%}  — model lift: "
          f"{avg - hit_rate:+.1%}")
else:
    print("Not enough setup bars for walk-forward windows.")

print("\nGenerating today's CPR trade signals...")
signals = predict_today(model_path="model.pkl", data_path="data.csv", symbol=SYMBOL)
print_signals(signals, symbol=SYMBOL, min_conf=0.60)
