import numpy as np
from src.features.traditional import compute_scores


class RuleBasedModel:
    """Scores numbers 1-39 purely from traditional feature aggregation."""

    def predict_proba(self, history: list[dict]) -> np.ndarray:
        return compute_scores(history)

    def top5(self, history: list[dict]) -> list[int]:
        scores = self.predict_proba(history)
        indices = np.argsort(scores)[::-1][:5]
        return sorted([int(i + 1) for i in indices])
