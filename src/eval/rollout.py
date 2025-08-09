from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from src.features import FeatureConfig, build_exogenous_features
from src.utils import rmse_pair


def _group_sort(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    return df.sort_values([cfg.filename_column, cfg.time_column]).reset_index(drop=True)


def rollout_exogenous_to_end(
    df: pd.DataFrame,
    xgb_model,
    mlp_model,
    scaler,
    feat_names: List[str],
    cfg: FeatureConfig,
    warmup: int = 100,
) -> Dict:
    df = _group_sort(df, cfg)
    X_all, y_all = build_exogenous_features(df, cfg)
    valid_mask = ~(X_all.isna().any(axis=1) | y_all.isna().any(axis=1))
    df_valid = df.loc[valid_mask].copy()
    X_all = X_all.loc[valid_mask]
    y_all = y_all.loc[valid_mask]

    # 建立對齊訓練特徵欄位的轉換器
    feat_means = {name: scaler.mean_[i] for i, name in enumerate(feat_names)}

    def align_and_transform(X_df: pd.DataFrame) -> np.ndarray:
        # 重新索引到訓練的欄位順序，缺失欄位以訓練均值填補
        X_aligned = pd.DataFrame(index=X_df.index, columns=feat_names)
        common = [c for c in feat_names if c in X_df.columns]
        if common:
            X_aligned.loc[:, common] = X_df[common]
        # 用訓練均值填補剩餘 NaN
        for c in feat_names:
            if X_aligned[c].isna().any():
                X_aligned[c] = X_aligned[c].fillna(feat_means.get(c, 0.0))
        return scaler.transform(X_aligned.values)

    # 只在每個檔案內從 warmup 之後開始計算
    preds = []
    trues = []
    for fname, idx in df_valid.groupby(cfg.filename_column).groups.items():
        idx = list(idx)
        if len(idx) <= warmup:
            continue
        # 只取本檔案 warmup 之後的資料列，交集到 X_all 現有欄位
        sub_idx = idx[warmup:]
        X_sub = X_all.loc[sub_idx]
        X_sub_scaled = align_and_transform(X_sub)
        p_xgb = xgb_model.predict(X_sub_scaled)
        p_mlp = mlp_model.predict(X_sub_scaled)
        p = 0.5 * p_xgb + 0.5 * p_mlp
        y_sub = y_all.loc[sub_idx].values
        preds.append(p)
        trues.append(y_sub)

    if not preds:
        return {"rmse_x": np.nan, "rmse_z": np.nan, "rmse_avg": np.nan}
    P = np.concatenate(preds, axis=0)
    Y = np.concatenate(trues, axis=0)
    return rmse_pair(Y, P)


def rollout_arx_to_end(
    df: pd.DataFrame,
    xgb_model,
    mlp_model,
    scaler,
    feat_names: List[str],
    cfg: FeatureConfig,
    warmup: int = 100,
) -> Dict:
    # 將 ARX 問題轉為逐步遞迴：非目標特徵直接從 exogenous 特徵構建；
    # 目標的 lag 特徵在 warmup 使用真值，之後使用模型預測。
    df = _group_sort(df, cfg)
    base_X_df, target_df = build_exogenous_features(df, cfg)

    # 準備輸出收集器
    preds_all: List[np.ndarray] = []
    trues_all: List[np.ndarray] = []

    # 需要知道哪些欄位是目標滯後特徵
    target_lag_cols = []
    for t in cfg.targets:
        for k in cfg.lags:
            target_lag_cols.append(f"{t}_lag{k}")

    # 每個檔案獨立遞迴
    for fname, g in df.groupby(cfg.filename_column):
        g = g.sort_values(cfg.time_column).copy()
        idx = g.index.to_list()
        if len(idx) <= warmup:
            continue

        # 取 base features（不含目標滯後）對應本檔案行
        base_feat = base_X_df.loc[idx]
        # 取真實目標
        y_true = target_df.loc[idx, list(cfg.targets)].values

        # 建立容器：將按時間一步步生成特徵向量，對齊 feat_names 的順序
        feat_mat = []
        y_mat = []

        # 用 warmup 真值初始化歷史目標序列
        history = {t: list(y_true[:warmup, i]) for i, t in enumerate(cfg.targets)}

        # 找到非目標滯後的特徵名稱集合
        non_target_cols = [c for c in feat_names if c not in set(target_lag_cols)]

        # 從 warmup 開始一路遞迴到結尾
        preds_seq: List[List[float]] = []  # [[px, pz], ...]
        for i in range(warmup, len(idx)):
            row = base_feat.iloc[i]
            feat_row = {}
            # 填入非目標滯後特徵
            for c in non_target_cols:
                feat_row[c] = row.get(c, np.nan)
            # 填入目標滯後特徵（由 history 提供）
            for t_i, t in enumerate(cfg.targets):
                for k in cfg.lags:
                    name = f"{t}_lag{k}"
                    if len(history[t]) >= k:
                        feat_row[name] = history[t][-k]
                    else:
                        feat_row[name] = np.nan

            # 按 feat_names 順序轉為向量
            x_vec = np.array([feat_row.get(c, np.nan) for c in feat_names], dtype=float)
            # 跳過含 NaN 的步
            if np.isnan(x_vec).any():
                # 將對應真值仍往後推進（但不產生預測）
                for t_i, t in enumerate(cfg.targets):
                    history[t].append(float(y_true[i, t_i]))
                continue

            feat_mat.append(x_vec)
            y_mat.append(y_true[i])

            # 預跑 scaler + model 做一次預測，並將預測推入 history（供下一步的 lag 使用）
            x_scaled = scaler.transform(x_vec.reshape(1, -1))
            p_xgb = xgb_model.predict(x_scaled)
            p_mlp = mlp_model.predict(x_scaled)
            p = 0.5 * p_xgb + 0.5 * p_mlp
            px, pz = float(p[0, 0]), float(p[0, 1])
            preds_seq.append([px, pz])
            history[cfg.targets[0]].append(px)
            history[cfg.targets[1]].append(pz)

        if preds_seq and y_mat:
            preds_all.append(np.asarray(preds_seq, dtype=float))
            trues_all.append(np.asarray(y_mat, dtype=float))

    if not preds_all:
        return {"rmse_x": np.nan, "rmse_z": np.nan, "rmse_avg": np.nan}

    P = np.concatenate(preds_all, axis=0)
    Y = np.concatenate(trues_all, axis=0)
    return rmse_pair(Y, P)


