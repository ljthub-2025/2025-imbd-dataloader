from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb


@dataclass
class XGBParams:
    # XGBoost>=2.0 建議以 device='cuda' 取代舊的 gpu_hist
    tree_method: str = "hist"
    device: str = "cuda"
    n_estimators: int = 800
    max_depth: int = 8
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42


class XGBRegressorPair:
    """
    兩個 XGBRegressor，分別預測 Disp. X 與 Disp. Z
    """

    def __init__(self, params: Optional[XGBParams] = None):
        self.params = params or XGBParams()
        self.model_x = xgb.XGBRegressor(**self.params.__dict__)
        self.model_z = xgb.XGBRegressor(**self.params.__dict__)

    def fit(self, X, y: np.ndarray):
        self.model_x.fit(X, y[:, 0])
        self.model_z.fit(X, y[:, 1])
        return self

    def predict(self, X) -> np.ndarray:
        # 將 numpy array 包成 DMatrix，避免 device mismatch 警告（顯式走 Booster.predict）
        try:
            dm = xgb.DMatrix(X)
            px = self.model_x.get_booster().predict(dm)
            pz = self.model_z.get_booster().predict(dm)
        except Exception:
            px = self.model_x.predict(X)
            pz = self.model_z.predict(X)
        return np.stack([px, pz], axis=1)


