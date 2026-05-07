from fetch_data import fetch_and_save
from data_loader import load_data
from features import build_features
from labels import create_cpr_labels
from model import train, train_calibrated
from walk_forward import run_walk_forward
from save_model import save
from predict_today import predict_today, print_signals

SYMBOL = "TSLA"

# Raw columns that must NOT enter the feature matrix:
#   OHLC   — absolute price levels cause memorisation, not generalisation
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

exclude = ["label"] + [c for c in OHLC + RAW_CPR if c in df_labeled.columns]
X = df_labeled.drop(columns=exclude)
y = df_labeled["label"]

long_setups  = int((df["below_cpr"].sum() + df["cpr_bottom_reclaim"].sum()))
short_setups = int((df["above_cpr"].sum() + df["cpr_top_reject"].sum()))
hit_rate     = float(y.mean())
print(f"CPR setups : {len(X)} bars  (LONG-zone: {long_setups}  SHORT-zone: {short_setups})")
print(f"Base rate  : {hit_rate:.1%} of setups reach their CPR target")
print(f"Features   : {len(X.columns)}  {list(X.columns)}")

print("\nTraining calibrated model (for live prediction)...")
model = train_calibrated(X, y)

print("Saving model...")
save(model, feature_cols=X.columns)

print("\nRunning walk-forward validation...")
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
