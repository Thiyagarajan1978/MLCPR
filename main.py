from fetch_data import fetch_and_save
from data_loader import load_data
from features import build_features
from labels import create_labels
from model import train
from walk_forward import run_walk_forward
from save_model import save
from predict_today import predict_today, print_signals

SYMBOL = "TSLA"

print(f"\nFetching {SYMBOL} data (last 7 days, 3-min bars)...")
fetch_and_save(symbol=SYMBOL, period="7d", interval="1m", output="data.csv")

print("\nLoading data...")
df = load_data("data.csv")

print("Building features...")
df = build_features(df)

print("Creating labels...")
df_labeled = create_labels(df.copy())

# Raw OHLC prices are absolute levels — they cause memorization, not generalisation.
# Only derived / CPR-relative features go into the model.
OHLC = ["open", "high", "low", "close", "volume"]
X = df_labeled.drop(["label"] + OHLC, axis=1)
y = df_labeled["label"]

print(f"Dataset: {len(X)} bars | Features ({len(X.columns)}): {list(X.columns)}")

print("\nTraining model...")
model = train(X, y)

print("Saving model...")
save(model, feature_cols=X.columns)

print("\nRunning walk-forward validation...")
results = run_walk_forward(df_labeled)
pct = [f"{r:.1%}" for r in results]
avg = sum(results) / len(results) if results else 0
print(f"Windows : {pct}")
print(f"Average : {avg:.1%}  ({'stable' if max(results) - min(results) < 0.15 else 'unstable - check overfitting'})")

print("\nGenerating today's trade signals...")
signals = predict_today(model_path="model.pkl", data_path="data.csv", symbol=SYMBOL)
print_signals(signals, symbol=SYMBOL, min_conf=0.75)
