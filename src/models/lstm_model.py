import os

import numpy as np
import torch
import torch.nn as nn

from src.features.traditional import FEATURE_FNS, build_feature_matrix

FEATURE_DIM = 39 * len(FEATURE_FNS)  # 39 * 9 = 351
SEQ_LEN = 10  # use last 10 draws as sequence
HIDDEN = 128
LAYERS = 2
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../../models/saved/lstm.pt")


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, HIDDEN, LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN, 39)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


class LSTMModel:
    def __init__(self, lr: float = 1e-3, epochs: int = 30):
        self.lr = lr
        self.epochs = epochs
        self.net = _Net()
        self.trained = False

    def _build_sequences(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        seqs, labels_out = [], []
        # We need SEQ_LEN feature rows to make a prediction for row SEQ_LEN
        for i in range(SEQ_LEN, len(X)):
            seqs.append(X[i - SEQ_LEN:i])
            labels_out.append(X[i])  # X rows pair with y — handled externally
        return np.array(seqs, dtype=np.float32)

    def fit(self, history: list[dict]) -> None:
        if len(history) < 30 + SEQ_LEN:
            self.trained = False
            return

        X, y = build_feature_matrix(history, start_idx=29)
        if len(X) <= SEQ_LEN:
            self.trained = False
            return

        seqs, ys = [], []
        for i in range(SEQ_LEN, len(X)):
            seqs.append(X[i - SEQ_LEN:i])
            ys.append(y[i])

        X_t = torch.tensor(np.array(seqs), dtype=torch.float32)
        y_t = torch.tensor(np.array(ys), dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            opt.step()

        self.trained = True

    def predict_proba(self, history: list[dict]) -> np.ndarray:
        if not self.trained or len(history) < SEQ_LEN + 29:
            return np.ones(39, dtype=np.float32) / 39

        X, _ = build_feature_matrix(history, start_idx=29)
        if len(X) < SEQ_LEN:
            return np.ones(39, dtype=np.float32) / 39

        seq = torch.tensor(X[-SEQ_LEN:][np.newaxis], dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            proba = self.net(seq).numpy()[0]
        return proba / (proba.sum() + 1e-9)

    def top5(self, history: list[dict]) -> list[int]:
        scores = self.predict_proba(history)
        indices = np.argsort(scores)[::-1][:5]
        return sorted([int(i + 1) for i in indices])

    def save(self, path: str = SAVE_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str = SAVE_PATH) -> None:
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.trained = True
