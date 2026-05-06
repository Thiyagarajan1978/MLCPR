# MLCPR — CPR ML Walk-Forward System

A machine learning system for testing and validating CPR (Central Pivot Range) based price behavior using walk-forward validation.

> **For testing and validation only. Do NOT use directly in live trading.**

## Overview

This system trains a Random Forest classifier on 3-minute OHLC candle data and evaluates it using walk-forward validation windows. The goal is to measure whether the model has a stable predictive edge before integrating it into a live trading system.

## Setup

```bash
pip install -r requirements.txt
```

## Data Format

Create `data.csv` with the following columns:

```
time,open,high,low,close
2024-01-01 09:30,100,101,99,100
2024-01-01 09:33,100,102,99,101
...
```

- Use **3-minute candles**
- Recommended source: Databento historical data

## Run

```bash
python main.py
```

## Output

- Walk-forward accuracy printed per window
- `model.pkl` saved for downstream use

## Interpreting Results

Do not assume 60% accuracy = good system. Ask:

- Is accuracy **stable** across all windows?
- Is it **dropping over time** (overfitting)?
- Is it near **random (~50%)**?

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — orchestrates full pipeline |
| `data_loader.py` | Loads and parses `data.csv` |
| `features.py` | Engineers features: range, body, trend, volatility |
| `labels.py` | Creates forward-looking binary labels |
| `model.py` | Trains a Random Forest classifier |
| `walk_forward.py` | Runs rolling train/test windows |
| `save_model.py` | Persists trained model to `model.pkl` |

## Next Steps

After validating results:

1. Add CPR-aware features (pivot levels, reclaim behavior)
2. Test across different symbols and volatility regimes
3. Integrate `model.pkl` into the live CPR scalper system
