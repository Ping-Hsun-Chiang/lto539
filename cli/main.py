#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click

from src.data.collector import collect as do_collect, load
from src.models.ensemble import EnsembleModel
from src.backtest.engine import run as do_backtest
from src.evaluation.evaluator import print_report


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
    """Run walk-forward backtest and print hit distribution."""
    history = load()
    if years:
        from datetime import datetime, timedelta
        cutoff = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        history = [r for r in history if r["date"] >= cutoff]
    print(f"[INFO] Running walk-forward backtest on {len(history)} draws...")
    results = do_backtest(history=history, verbose=True)
    print_report(results)


if __name__ == "__main__":
    cli()
