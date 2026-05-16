#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click

from src.data.collector import collect as do_collect, load
from src.models.ensemble import EnsembleModel
from src.backtest.engine import run as do_backtest, random_baseline
from src.evaluation.evaluator import print_report, print_comparison

ROLLING_WINDOWS = [6, 12, 18, 24]


@click.group()
def cli():
    """今彩539 深度學習預測系統"""


@cli.command()
@click.option("--years", default=2, show_default=True, help="Years of history to fetch")
def collect(years):
    """Download / update historical draw data from Taiwan Lottery API."""
    do_collect(years=years)


@cli.command()
def train():
    """Train the ensemble model on all available history."""
    history = load()
    print(f"[INFO] Loaded {len(history)} draws. Training...")
    model = EnsembleModel()
    model.fit(history)
    model.lstm.save()
    print("[OK] Model trained and saved.")


@cli.command()
def predict():
    """Predict 5 numbers for the next draw."""
    history = load()
    model = EnsembleModel()
    model.fit(history)
    numbers = model.top5(history)
    last = history[-1]
    print(f"\n  Last draw : Period {last['period']}  {last['date']} ({last['weekday'][:3]})")
    print(f"              Numbers: {last['numbers']}")
    print(f"\n  Predicted next draw: {numbers}")
    print()


@cli.command()
@click.option("--years", default=None, type=int, help="Limit backtest to last N years of data")
def backtest(years):
    """Run expanding window walk-forward backtest."""
    history = load()
    if years:
        from datetime import datetime, timedelta
        cutoff = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        history = [r for r in history if r["date"] >= cutoff]
    print(f"[INFO] Running expanding window backtest on {len(history)} draws...")
    results = do_backtest(history=history, verbose=True)
    print_report(results)


@cli.command()
def compare():
    """Run all 6 methods and print comparison table."""
    import json

    history = load()
    n = len(history)
    print(f"[INFO] Loaded {n} draws. Running 5 backtests + random baseline...\n")

    all_results = {}

    # 1. Expanding window (load from saved file if exists)
    saved = "backtest_results.json"
    if os.path.exists(saved):
        print(f"[INFO] Loading expanding window results from {saved}")
        with open(saved, encoding="utf-8") as f:
            all_results["Expanding"] = json.load(f)
    else:
        print("[INFO] Running expanding window backtest...")
        all_results["Expanding"] = do_backtest(history=history, window=None, verbose=True, label="Expanding")
        with open(saved, "w", encoding="utf-8") as f:
            json.dump(all_results["Expanding"], f, ensure_ascii=False)

    # 2-5. Rolling windows
    for w in ROLLING_WINDOWS:
        fname = f"backtest_results_rolling_{w}.json"
        if os.path.exists(fname):
            print(f"[INFO] Loading rolling-{w} results from {fname}")
            with open(fname, encoding="utf-8") as f:
                all_results[f"Rolling-{w}"] = json.load(f)
        else:
            print(f"[INFO] Running rolling-{w} backtest...")
            res = do_backtest(history=history, window=w, verbose=True, label=f"Rolling-{w}")
            all_results[f"Rolling-{w}"] = res
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)

    # 6. Random baseline
    n_preds = len(all_results["Expanding"])
    all_results["Random"] = random_baseline(n_preds)

    print_comparison(all_results)


if __name__ == "__main__":
    cli()
