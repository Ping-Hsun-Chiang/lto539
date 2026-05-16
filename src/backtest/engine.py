"""
Walk-Forward Backtest Engine.

For each draw T (starting from START_IDX):
  1. Train on history[0 : T]
  2. Predict 5 numbers for draw T+1
  3. Compare with actual draw T+1
  4. Record match count (0-5)
"""

from collections import Counter

from tqdm import tqdm

from src.data.collector import load
from src.models.ensemble import EnsembleModel

START_IDX = 29  # need at least 30 draws before first prediction


def run(history: list[dict] | None = None, verbose: bool = True) -> list[dict]:
    if history is None:
        history = load()

    results = []
    total = len(history) - START_IDX - 1

    iterator = range(START_IDX, len(history) - 1)
    if verbose:
        iterator = tqdm(iterator, total=total, desc="Backtesting", unit="draw")

    for t in iterator:
        train_data = history[: t + 1]
        actual = history[t + 1]["numbers"]

        model = EnsembleModel()
        model.fit(train_data)
        predicted = model.top5(train_data)

        hits = len(set(predicted) & set(actual))
        results.append({
            "period": history[t + 1]["period"],
            "date": history[t + 1]["date"],
            "weekday": history[t + 1]["weekday"],
            "predicted": predicted,
            "actual": actual,
            "hits": hits,
        })

    return results


def summarise(results: list[dict]) -> dict:
    dist = Counter(r["hits"] for r in results)
    total = len(results)
    summary = {}
    for k in range(6):
        count = dist.get(k, 0)
        summary[k] = {"count": count, "pct": count / total * 100 if total else 0}
    return summary
