from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


EXOG_COLUMNS_BASE = [
    # PT & TC sensors
    *[f"PT{str(i).zfill(2)}" for i in range(1, 14)],
    *[f"TC{str(i).zfill(2)}" for i in range(1, 9)],
    # Motors
    "Spindle Motor",
    "X Motor",
    "Z Motor",
]


@dataclass
class FeatureConfig:
    sequence_length: int = 100
    lags: Tuple[int, ...] = (1, 3, 5, 10, 30)
    rolling_windows: Tuple[int, ...] = (5, 15, 60)
    use_target_lags: bool = False  # for ARX version
    targets: Tuple[str, str] = ("Disp. X", "Disp. Z")
    time_column: str = "Time"
    filename_column: str = "filename"


def _safe_group_sort(df: pd.DataFrame, time_col: str, filename_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if filename_col not in df.columns:
        raise ValueError(f"Missing filename column: {filename_col}")
    return df.sort_values([filename_col, time_col]).reset_index(drop=True)


def _add_time_delta(df: pd.DataFrame, time_col: str, filename_col: str) -> pd.DataFrame:
    df = df.copy()
    df["delta_t"] = df.groupby(filename_col)[time_col].diff().fillna(0.0)
    return df


def _make_lagged_features(df: pd.DataFrame, cols: List[str], lags: Tuple[int, ...], filename_col: str) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(filename_col)
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        gcol = grp[col]
        for k in lags:
            new_cols[f"{col}_lag{k}"] = gcol.shift(k)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _make_rolling_features(df: pd.DataFrame, cols: List[str], windows: Tuple[int, ...], filename_col: str) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(filename_col)
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        gcol = grp[col]
        for w in windows:
            new_cols[f"{col}_roll{w}_mean"] = gcol.transform(lambda s: s.rolling(w, min_periods=1).mean())
            new_cols[f"{col}_roll{w}_std"] = gcol.transform(lambda s: s.rolling(w, min_periods=1).std())
            new_cols[f"{col}_roll{w}_min"] = gcol.transform(lambda s: s.rolling(w, min_periods=1).min())
            new_cols[f"{col}_roll{w}_max"] = gcol.transform(lambda s: s.rolling(w, min_periods=1).max())
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _make_diff_features(df: pd.DataFrame, cols: List[str], filename_col: str) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(filename_col)
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        gcol = grp[col]
        new_cols[f"{col}_diff1"] = gcol.diff(1)
        new_cols[f"{col}_diff3"] = gcol.diff(3)
        new_cols[f"{col}_diff5"] = gcol.diff(5)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def build_exogenous_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    構建「不使用目標滯後」的表格特徵（Exogenous-only）。
    僅使用感測器與馬達等外生變數，避免推論時依賴未來的目標。

    返回 (features_df, targets_df)。
    """
    df = _safe_group_sort(df, cfg.time_column, cfg.filename_column)
    df = _add_time_delta(df, cfg.time_column, cfg.filename_column)

    use_cols = [c for c in EXOG_COLUMNS_BASE if c in df.columns]
    df_feat = _make_lagged_features(df, use_cols, cfg.lags, cfg.filename_column)
    df_feat = _make_rolling_features(df_feat, use_cols, cfg.rolling_windows, cfg.filename_column)
    df_feat = _make_diff_features(df_feat, use_cols, cfg.filename_column)

    target_df = df_feat.loc[:, list(cfg.targets)].copy()
    # 僅保留數值特徵，排除像是 filename 等非數值欄位
    feature_cols = [c for c in df_feat.columns if c not in set(cfg.targets)]
    feat_df = df_feat[feature_cols].copy()
    feat_df = feat_df.select_dtypes(include=[np.number])
    return feat_df, target_df


def build_exogenous_ar_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    構建「包含少量目標滯後（ARX）」的表格特徵。
    僅使用過去的目標值（例如 1/3/5 步滯後），推論時以已知的 100 步真值啟動，之後用模型預測回饋。
    返回 (features_df, targets_df)。
    """
    if not cfg.use_target_lags:
        cfg = FeatureConfig(
            sequence_length=cfg.sequence_length,
            lags=cfg.lags,
            rolling_windows=cfg.rolling_windows,
            use_target_lags=True,
            targets=cfg.targets,
            time_column=cfg.time_column,
            filename_column=cfg.filename_column,
        )

    feat_df, target_df = build_exogenous_features(df, cfg)
    for t in cfg.targets:
        if t in df.columns:
            for k in cfg.lags:
                feat_df[f"{t}_lag{k}"] = df.groupby(cfg.filename_column)[t].shift(k)
    # 再次保留數值欄位，避免任何非數值型別混入
    feat_df = feat_df.select_dtypes(include=[np.number])
    return feat_df, target_df


