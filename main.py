from data_loader import load_data
from features import build_features
from labels import create_labels
from model import train
from walk_forward import run_walk_forward
from save_model import save

print("Loading data...")
df = load_data("data.csv")

print("Building features...")
df = build_features(df)

print("Creating labels...")
df = create_labels(df)

X = df.drop(["label"], axis=1)
y = df["label"]

print("Training model...")
model = train(X, y)

print("Saving model...")
save(model)

print("Running walk-forward...")
results = run_walk_forward(df)

print("Walk-forward accuracy:", results)
