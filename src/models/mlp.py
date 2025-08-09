from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MLPParams:
    input_dim: int
    hidden_dims: tuple = (512, 256, 128)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    batch_size: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers += [nn.Linear(prev, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)]
            prev = hd
        layers += [nn.Linear(prev, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPRegressorPair:
    def __init__(self, params: MLPParams):
        self.params = params
        self.model = _MLP(params.input_dim, params.hidden_dims, params.dropout).to(params.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    def fit(self, X: np.ndarray, y: np.ndarray):
        device = self.params.device
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).float().to(device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.params.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = self.params.device
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(device)
            pred = self.model(X_t).cpu().numpy()
        return pred


