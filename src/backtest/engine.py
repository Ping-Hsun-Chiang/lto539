"""
Walk-Forward Backtest Engine.

Supports two modes:
  - Expanding window (window=None): train on all history up to T
  - Rolling window  (window=N)    : train on last N draws before T
"""

from collections import Counter
from math import comb

from tqdm import tqdm

from src.data.collector import load
from src.models.ensemble import EnsembleModel

START_IDX = 29  # need at least 30 draws before first prediction


def run(
    history: list[dict] | None = None,
    window: int | None = None,
    verbose: bool = True,
    label: str = "",
) -> list[dict]:
    """
    window=None  → expanding (cumulative) window
    window=N     → rolling window of last N draws
    """
    if history is None:
        history = load()

    results = []
    total = len(history) - START_IDX - 1
    desc = label or ("Expanding" if window is None else f"Rolling-{window}")

    iterator = range(START_IDX, len(history) - 1)
    if verbose:
        iterator = tqdm(iterator, total=total, desc=desc, unit="draw")

    for t in iterator:
        if window is None:
            train_data = history[: t + 1]
        else:
            start = max(0, t + 1 - window)
            train_data = history[start : t + 1]

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


def random_baseline(n_predictions: int = 616) -> dict:
    """
    Theoretical hit distribution for random 5-from-39 guess.
    Uses hypergeometric distribution: P(X=k) = C(5,k)*C(34,5-k)/C(39,5)
    """
    total_ways = comb(39, 5)
    dist = {}
    for k in range(6):
        ways = comb(5, k) * comb(34, 5 - k)
        pct = ways / total_ways * 100
        dist[k] = {"count": round(pct * n_predictions / 100), "pct": pct}
    return dist


def summarise(results: list[dict]) -> dict:
    dist = Counter(r["hits"] for r in results)
    total = len(results)
    summary = {}
    for k in range(6):
        count = dist.get(k, 0)
        summary[k] = {"count": count, "pct": count / total * 100 if total else 0}
    return summary
