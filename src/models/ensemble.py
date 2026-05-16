import numpy as np

from src.models.rule_based import RuleBasedModel

WEIGHTS = {"rule": 0.4, "lstm": 0.6}


class EnsembleModel:
    def __init__(self, use_lstm: bool = True):
        self.rule = RuleBasedModel()
        self.lstm = None
        if use_lstm:
            try:
                from src.models.lstm_model import LSTMModel
                self.lstm = LSTMModel()
            except ImportError:
                pass  # torch not available, fall back to rule-based only

    def fit(self, history: list[dict]) -> None:
        if self.lstm is not None:
            self.lstm.fit(history)

    def predict_proba(self, history: list[dict]) -> np.ndarray:
        rule_s = self.rule.predict_proba(history)
        if self.lstm is not None:
            lstm_s = self.lstm.predict_proba(history)
            combined = WEIGHTS["rule"] * rule_s + WEIGHTS["lstm"] * lstm_s
        else:
            combined = rule_s
        return combined / (combined.sum() + 1e-9)

    def top5(self, history: list[dict]) -> list[int]:
        scores = self.predict_proba(history)
        indices = np.argsort(scores)[::-1][:5]
        return sorted([int(i + 1) for i in indices])
