from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_pair(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 2
    rx = rmse(y_true[:, 0], y_pred[:, 0])
    rz = rmse(y_true[:, 1], y_pred[:, 1])
    return {"rmse_x": rx, "rmse_z": rz, "rmse_avg": (rx + rz) / 2.0}


