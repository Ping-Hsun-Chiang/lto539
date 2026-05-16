"""
9 traditional lottery feature sets for 今彩539 (numbers 1-39).

Each feature function takes `history` (list of draw dicts, oldest-first)
and returns a score array of shape (39,) indexed by number-1.
Higher score = model thinks this number is more likely.
"""

import math
from collections import defaultdict

import numpy as np

NUMBERS = list(range(1, 40))
WINDOWS = [6, 12, 18, 24]  # 1w / 2w / 3w / 4w


def _presence_matrix(history: list[dict]) -> np.ndarray:
    """Binary matrix (T x 39): 1 if number appeared in draw t."""
    mat = np.zeros((len(history), 39), dtype=np.float32)
    for t, draw in enumerate(history):
        for n in draw["numbers"]:
            mat[t, n - 1] = 1.0
    return mat


# ── Feature 1: Hot / Cold (frequency in each window) ─────────────────────────
def hot_cold(history: list[dict]) -> np.ndarray:
    mat = _presence_matrix(history)
    scores = np.zeros(39, dtype=np.float32)
    for w in WINDOWS:
        window = mat[-w:] if len(history) >= w else mat
        freq = window.sum(axis=0)
        # normalise within window
        freq = freq / (freq.sum() + 1e-9)
        scores += freq
    return scores / len(WINDOWS)


# ── Feature 2: Missing value (periods since last appearance) ──────────────────
def missing_value(history: list[dict]) -> np.ndarray:
    last_seen = {}
    for t, draw in enumerate(history):
        for n in draw["numbers"]:
            last_seen[n] = t
    T = len(history)
    missing = np.array([T - last_seen.get(n, -1) - 1 for n in NUMBERS], dtype=np.float32)
    # longer missing → higher score (number is "due")
    missing = missing / (missing.max() + 1e-9)
    return missing


# ── Feature 3: Average appearance cycle ───────────────────────────────────────
def avg_cycle(history: list[dict]) -> np.ndarray:
    appearances = defaultdict(list)
    for t, draw in enumerate(history):
        for n in draw["numbers"]:
            appearances[n].append(t)

    scores = np.zeros(39, dtype=np.float32)
    T = len(history)
    for n in NUMBERS:
        app = appearances[n]
        if len(app) < 2:
            # never or once seen → neutral
            scores[n - 1] = 0.5
            continue
        gaps = [app[i + 1] - app[i] for i in range(len(app) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        periods_since = T - app[-1] - 1
        # score: how close to avg_gap the current wait is
        deviation = abs(periods_since - avg_gap)
        scores[n - 1] = 1.0 / (1.0 + deviation)
    return scores


# ── Feature 4: Co-occurrence (which numbers appear together) ──────────────────
def co_occurrence(history: list[dict]) -> np.ndarray:
    """
    For numbers seen in the last 24 draws, compute how often each other
    number appeared with them → aggregate into a single score.
    """
    window = history[-24:] if len(history) >= 24 else history
    co_mat = np.zeros((39, 39), dtype=np.float32)
    for draw in window:
        nums = draw["numbers"]
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    co_mat[nums[i] - 1, nums[j] - 1] += 1

    # last draw's numbers as seed
    last_nums = history[-1]["numbers"]
    scores = np.zeros(39, dtype=np.float32)
    for n in last_nums:
        scores += co_mat[n - 1]
    scores = scores / (scores.sum() + 1e-9)
    return scores


# ── Feature 5: Conditional probability (if A last draw → B this draw) ─────────
def conditional_prob(history: list[dict]) -> np.ndarray:
    if len(history) < 2:
        return np.ones(39, dtype=np.float32) / 39

    trigger_counts = defaultdict(lambda: defaultdict(int))
    trigger_total = defaultdict(int)
    for t in range(len(history) - 1):
        prev = history[t]["numbers"]
        curr = history[t + 1]["numbers"]
        for a in prev:
            trigger_total[a] += 1
            for b in curr:
                trigger_counts[a][b] += 1

    last_nums = history[-1]["numbers"]
    scores = np.zeros(39, dtype=np.float32)
    for a in last_nums:
        total = trigger_total[a]
        if total == 0:
            continue
        for b in NUMBERS:
            scores[b - 1] += trigger_counts[a][b] / total
    scores = scores / (scores.sum() + 1e-9)
    return scores


# ── Feature 6: Weekday pattern ────────────────────────────────────────────────
def weekday_pattern(history: list[dict]) -> np.ndarray:
    next_weekday = history[-1]["weekday"]  # approximate: same weekday repeats
    # actually predict next draw's weekday
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    try:
        idx = weekday_order.index(next_weekday)
        next_wd = weekday_order[(idx + 1) % len(weekday_order)]
    except ValueError:
        next_wd = next_weekday

    freq = np.zeros(39, dtype=np.float32)
    count = 0
    for draw in history:
        if draw["weekday"] == next_wd:
            for n in draw["numbers"]:
                freq[n - 1] += 1
            count += 1
    if count == 0:
        return np.ones(39, dtype=np.float32) / 39
    return freq / (freq.sum() + 1e-9)


# ── Feature 7: Consecutive number analysis ────────────────────────────────────
def consecutive(history: list[dict]) -> np.ndarray:
    """Score numbers that tend to appear consecutively with recent hot numbers."""
    window = history[-12:] if len(history) >= 12 else history
    freq = np.zeros(39, dtype=np.float32)
    for draw in window:
        nums = sorted(draw["numbers"])
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                freq[nums[i] - 1] += 1
                freq[nums[i + 1] - 1] += 1
    return freq / (freq.sum() + 1e-9)


# ── Feature 8: Tail digit (last digit 0-9) distribution ──────────────────────
def tail_digit(history: list[dict]) -> np.ndarray:
    window = history[-24:] if len(history) >= 24 else history
    tail_freq = np.zeros(10, dtype=np.float32)
    for draw in window:
        for n in draw["numbers"]:
            tail_freq[n % 10] += 1
    tail_freq = tail_freq / (tail_freq.sum() + 1e-9)

    scores = np.zeros(39, dtype=np.float32)
    for n in NUMBERS:
        scores[n - 1] = tail_freq[n % 10]
    return scores / (scores.sum() + 1e-9)


# ── Feature 9: Head digit (tens digit: 0-head 1-9, 1-head 10-19, …) ──────────
def head_digit(history: list[dict]) -> np.ndarray:
    window = history[-24:] if len(history) >= 24 else history
    head_freq = np.zeros(4, dtype=np.float32)  # 0頭 1頭 2頭 3頭
    for draw in window:
        for n in draw["numbers"]:
            head_freq[(n - 1) // 10] += 1
    head_freq = head_freq / (head_freq.sum() + 1e-9)

    scores = np.zeros(39, dtype=np.float32)
    for n in NUMBERS:
        scores[n - 1] = head_freq[(n - 1) // 10]
    return scores / (scores.sum() + 1e-9)


# ── Aggregate all features into one score vector ──────────────────────────────
FEATURE_FNS = [
    hot_cold,
    missing_value,
    avg_cycle,
    co_occurrence,
    conditional_prob,
    weekday_pattern,
    consecutive,
    tail_digit,
    head_digit,
]

FEATURE_WEIGHTS = [1.0] * len(FEATURE_FNS)  # equal weight; tunable


def compute_scores(history: list[dict]) -> np.ndarray:
    """Return weighted sum of all feature scores, shape (39,)."""
    scores = np.zeros(39, dtype=np.float32)
    for fn, w in zip(FEATURE_FNS, FEATURE_WEIGHTS):
        s = fn(history)
        scores += w * s
    return scores / (scores.sum() + 1e-9)


def build_feature_matrix(history: list[dict], start_idx: int = 29) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix for model training.
    Returns:
        X: (T-start_idx, 39*len(FEATURE_FNS)) feature vectors
        y: (T-start_idx, 39) binary label vectors
    """
    X, y = [], []
    for t in range(start_idx, len(history)):
        past = history[:t]
        row = np.concatenate([fn(past) for fn in FEATURE_FNS])
        label = np.zeros(39, dtype=np.float32)
        for n in history[t]["numbers"]:
            label[n - 1] = 1.0
        X.append(row)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
