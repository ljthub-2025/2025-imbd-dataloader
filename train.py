from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataloader import get_data
from src.features import build_exogenous_features, build_exogenous_ar_features, FeatureConfig
from src.models import XGBRegressorPair
try:
    from src.models.mlp import MLPRegressorPair, MLPParams  # type: ignore
    _TORCH_OK = True
except Exception:
    MLPRegressorPair = None  # type: ignore
    MLPParams = None  # type: ignore
    _TORCH_OK = False
from src.models.xgb import XGBParams
from src.utils import rmse_pair
from src.eval.rollout import rollout_exogenous_to_end, rollout_arx_to_end


TARGETS = ("Disp. X", "Disp. Z")


def prepare_xy(train_df: pd.DataFrame, val_df: pd.DataFrame, use_arx: bool, cfg: FeatureConfig):
    if use_arx:
        X_tr, y_tr = build_exogenous_ar_features(train_df, cfg)
        X_va, y_va = build_exogenous_ar_features(val_df, cfg)
    else:
        X_tr, y_tr = build_exogenous_features(train_df, cfg)
        X_va, y_va = build_exogenous_features(val_df, cfg)

    # 將包含 NaN 的前期行去除（由於滯後/rolling 產生）
    valid_tr = ~(X_tr.isna().any(axis=1) | y_tr.isna().any(axis=1))
    valid_va = ~(X_va.isna().any(axis=1) | y_va.isna().any(axis=1))
    X_tr, y_tr = X_tr.loc[valid_tr], y_tr.loc[valid_tr]
    X_va, y_va = X_va.loc[valid_va], y_va.loc[valid_va]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.values)
    X_va_scaled = scaler.transform(X_va.values)

    return X_tr_scaled, y_tr.values, X_va_scaled, y_va.values, scaler, list(X_tr.columns)


def fit_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, use_arx: bool, out_dir: Path):
    cfg = FeatureConfig(use_target_lags=use_arx)
    X_tr, y_tr, X_va, y_va, scaler, feat_names = prepare_xy(train_df, val_df, use_arx, cfg)

    # XGBoost
    xgb_model = XGBRegressorPair(XGBParams())
    xgb_model.fit(X_tr, y_tr)
    pred_xgb = xgb_model.predict(X_va)
    m_xgb = rmse_pair(y_va, pred_xgb)

    # MLP（如果可用）
    pred_mlp = None
    m_mlp = None
    if _TORCH_OK and MLPParams is not None and MLPRegressorPair is not None:
        mlp_params = MLPParams(input_dim=X_tr.shape[1])
        mlp_model = MLPRegressorPair(mlp_params)
        mlp_model.fit(X_tr, y_tr)
        pred_mlp = mlp_model.predict(X_va)
        m_mlp = rmse_pair(y_va, pred_mlp)

    # 簡單 blending
    if pred_mlp is not None:
        pred_blend = 0.5 * pred_xgb + 0.5 * pred_mlp
        m_blend = rmse_pair(y_va, pred_blend)
    else:
        pred_blend = pred_xgb
        m_blend = rmse_pair(y_va, pred_blend)

    # 依照真實測試型態，做暖身100步後一路到結尾的 rollout 評估
    # 選擇用 XGB 或 XGB+MLP 的集成
    mlp_for_roll = mlp_model if (_TORCH_OK and pred_mlp is not None) else None
    if mlp_for_roll is None:
        class _Dummy:
            def predict(self, X):
                return np.zeros((X.shape[0], 2), dtype=float)
        mlp_for_roll = _Dummy()

    roll_exo = rollout_exogenous_to_end(val_df, xgb_model, mlp_for_roll, scaler, feat_names, cfg, warmup=cfg.sequence_length)
    roll_arx = rollout_arx_to_end(val_df, xgb_model, mlp_for_roll, scaler, feat_names, cfg, warmup=cfg.sequence_length) if use_arx else None

    results = {
        "xgb": m_xgb,
        "mlp": m_mlp,
        "blend": m_blend,
        "rollout": roll_arx if use_arx else roll_exo,
    }

    # 保存模型與前處理
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / f"scaler_{'arx' if use_arx else 'exo'}.joblib")
    joblib.dump(feat_names, out_dir / f"features_{'arx' if use_arx else 'exo'}.joblib")
    joblib.dump(xgb_model, out_dir / f"xgb_{'arx' if use_arx else 'exo'}.joblib")
    joblib.dump(mlp_model, out_dir / f"mlp_{'arx' if use_arx else 'exo'}.joblib")
    (out_dir / f"metrics_{'arx' if use_arx else 'exo'}.json").write_text(pd.Series(results).to_json(indent=2, force_ascii=False))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="2025_dataset_0806/train")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num_k", type=int, default=1)
    parser.add_argument("--train_val_ratio", type=float, default=0.8)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    train_df, val_df, test_df = get_data(args.folder, k=args.k, num_k=args.num_k, train_val_ratio=args.train_val_ratio)

    out_dir = Path(args.out)
    print("[Exogenous-only] 訓練與驗證...")
    res_exo = fit_and_eval(train_df, val_df, use_arx=False, out_dir=out_dir)
    print("[ARX] 訓練與驗證...")
    res_arx = fit_and_eval(train_df, val_df, use_arx=True, out_dir=out_dir)

    print("驗證結果：")
    print("EXO:", res_exo)
    print("ARX:", res_arx)


if __name__ == "__main__":
    main()


