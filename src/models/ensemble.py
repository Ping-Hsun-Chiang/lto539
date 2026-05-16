import numpy as np

from src.models.lstm_model import LSTMModel
from src.models.rule_based import RuleBasedModel

# weight: rule-based gets 0.4, LSTM gets 0.6
WEIGHTS = {"rule": 0.4, "lstm": 0.6}


class EnsembleModel:
    def __init__(self):
        self.rule = RuleBasedModel()
        self.lstm = LSTMModel()

    def fit(self, history: list[dict]) -> None:
        self.lstm.fit(history)

    def predict_proba(self, history: list[dict]) -> np.ndarray:
        rule_s = self.rule.predict_proba(history)
        lstm_s = self.lstm.predict_proba(history)
        combined = WEIGHTS["rule"] * rule_s + WEIGHTS["lstm"] * lstm_s
        return combined / (combined.sum() + 1e-9)

    def top5(self, history: list[dict]) -> list[int]:
        scores = self.predict_proba(history)
        indices = np.argsort(scores)[::-1][:5]
        return sorted([int(i + 1) for i in indices])
