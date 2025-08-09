from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FeatureConfig, build_exogenous_features


TARGETS = ("Disp. X", "Disp. Z")


def _predict_exogenous(df: pd.DataFrame, models_dir: Path, warmup: int) -> pd.DataFrame:
    cfg = FeatureConfig(use_target_lags=False, sequence_length=warmup)
    scaler = joblib.load(models_dir / "scaler_exo.joblib")
    feat_names = joblib.load(models_dir / "features_exo.joblib")
    xgb_model = joblib.load(models_dir / "xgb_exo.joblib")
    mlp_path = models_dir / "mlp_exo.joblib"
    if mlp_path.exists():
        mlp_model = joblib.load(mlp_path)
    else:
        class _Dummy:
            def predict(self, X):
                return np.zeros((X.shape[0], 2), dtype=float)
        mlp_model = _Dummy()

    X_all, _ = build_exogenous_features(df, cfg)
    valid_mask = ~(X_all.isna().any(axis=1))
    X_all = X_all.loc[valid_mask]

    # 將特徵對齊訓練欄位與順序，缺欄以訓練均值填補
    scaler = joblib.load(models_dir / "scaler_exo.joblib")
    feat_means = {name: scaler.mean_[i] for i, name in enumerate(feat_names)}
    def align_df(Xdf: pd.DataFrame) -> pd.DataFrame:
        X_aligned = pd.DataFrame(index=Xdf.index, columns=feat_names)
        common = [c for c in feat_names if c in Xdf.columns]
        if common:
            X_aligned.loc[:, common] = Xdf[common]
        for c in feat_names:
            if X_aligned[c].isna().any():
                X_aligned[c] = X_aligned[c].fillna(feat_means.get(c, 0.0))
        return X_aligned

    keep_idx = []
    for fname, idx in df.loc[valid_mask].groupby(cfg.filename_column).groups.items():
        idx = list(idx)
        if len(idx) <= warmup:
            continue
        keep_idx.extend(idx[warmup:])
    keep_mask = X_all.index.isin(keep_idx)

    X_keep = align_df(X_all.loc[keep_mask])
    X_scaled = scaler.transform(X_keep.values)
    pred = 0.5 * xgb_model.predict(X_scaled) + 0.5 * mlp_model.predict(X_scaled)

    out_df = df.loc[keep_mask, [cfg.filename_column, cfg.time_column]].copy()
    out_df["pred_Disp_X"] = pred[:, 0]
    out_df["pred_Disp_Z"] = pred[:, 1]
    return out_df


def _predict_arx(df: pd.DataFrame, models_dir: Path, warmup: int) -> pd.DataFrame:
    from src.features import FeatureConfig  # local import to avoid cycles
    cfg = FeatureConfig(use_target_lags=True, sequence_length=warmup)
    scaler = joblib.load(models_dir / "scaler_arx.joblib")
    feat_names = joblib.load(models_dir / "features_arx.joblib")
    xgb_model = joblib.load(models_dir / "xgb_arx.joblib")
    mlp_path = models_dir / "mlp_arx.joblib"
    if mlp_path.exists():
        mlp_model = joblib.load(mlp_path)
    else:
        class _Dummy:
            def predict(self, X):
                return np.zeros((X.shape[0], 2), dtype=float)
        mlp_model = _Dummy()

    # 構建 base exogenous 特徵（不含目標滯後），滯後由遞迴生成
    base_X_df, target_df = build_exogenous_features(df, cfg)

    outputs = []
    for fname, g in df.groupby(cfg.filename_column):
        g = g.sort_values(cfg.time_column).copy()
        idx = g.index.to_list()
        if len(idx) <= warmup:
            continue
        base_feat = base_X_df.loc[idx]
        y_true = target_df.loc[idx, list(cfg.targets)].values

        history = {t: list(y_true[:warmup, i]) for i, t in enumerate(cfg.targets)}
        target_lag_cols = [f"{t}_lag{k}" for t in cfg.targets for k in cfg.lags]
        non_target_cols = [c for c in feat_names if c not in set(target_lag_cols)]

        rows = []
        for i in range(warmup, len(idx)):
            row = base_feat.iloc[i]
            feat_row = {}
            for c in non_target_cols:
                feat_row[c] = row.get(c, np.nan)
            for t in cfg.targets:
                for k in cfg.lags:
                    name = f"{t}_lag{k}"
                    vals = history[t]
                    feat_row[name] = vals[-k] if len(vals) >= k else np.nan
            x_vec = np.array([feat_row.get(c, np.nan) for c in feat_names], dtype=float)
            if np.isnan(x_vec).any():
                history[cfg.targets[0]].append(float(y_true[i, 0]))
                history[cfg.targets[1]].append(float(y_true[i, 1]))
                continue
            x_scaled = scaler.transform(x_vec.reshape(1, -1))
            p = 0.5 * xgb_model.predict(x_scaled) + 0.5 * mlp_model.predict(x_scaled)
            px, pz = float(p[0, 0]), float(p[0, 1])
            rows.append({cfg.filename_column: fname, cfg.time_column: g.iloc[i][cfg.time_column], "pred_Disp_X": px, "pred_Disp_Z": pz})
            history[cfg.targets[0]].append(px)
            history[cfg.targets[1]].append(pz)

        if rows:
            outputs.append(pd.DataFrame(rows))

    if outputs:
        return pd.concat(outputs, ignore_index=True)
    return pd.DataFrame(columns=["filename", "Time", "pred_Disp_X", "pred_Disp_Z"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--models", type=str, default="outputs")
    parser.add_argument("--use_arx", action="store_true")
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "filename" not in df.columns:
        df["filename"] = Path(args.csv).name

    if args.use_arx:
        out = _predict_arx(df, Path(args.models), warmup=args.warmup)
    else:
        out = _predict_exogenous(df, Path(args.models), warmup=args.warmup)
    print(out.head())


if __name__ == "__main__":
    main()


