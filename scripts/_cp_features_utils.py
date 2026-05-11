"""
Small helper module for the rxrx1 CellProfiler analysis.

Lifted (with minor trimming) from
/mnt/pvc/MorphGen/sc_perturb/evaluation/qualitative/analyze_cp_features.py
so `scripts/analyze_cp_features_rxrx1.py` can run standalone without
cross-repo imports.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def select_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    variance_thresh: float = 1e-5,
    z_thresh: float = 5.0,
) -> List[str]:
    """Drop low-variance features and features that contain extreme outliers (|z| > z_thresh)."""
    selector = VarianceThreshold(threshold=variance_thresh).fit(df[feature_columns])
    kept = [feature_columns[i] for i, keep in enumerate(selector.get_support()) if keep]
    zvals = np.abs(zscore(df[kept], nan_policy="omit"))
    outlier_mask = (zvals < z_thresh).all(axis=0)
    return [c for c, keep in zip(kept, outlier_mask) if keep]


def remove_highly_correlated_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    corr_thresh: float = 0.7,
) -> List[str]:
    """Drop one of each pair of features with |corr| > corr_thresh (order-preserving)."""
    corr = df[feature_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > corr_thresh).any()]
    return [c for c in feature_columns if c not in drop]


def preprocess_features_fit_on_real(
    df: pd.DataFrame,
    feature_columns: List[str],
    source_col: str = "Source",
    real_value: str = "real",
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on real rows only, then transform every row in-place."""
    real_mask = df[source_col] == real_value
    if real_mask.sum() == 0:
        raise ValueError(f"No rows with {source_col} == {real_value!r} to fit scaler.")
    scaler = StandardScaler().fit(df.loc[real_mask, feature_columns])
    out = df.copy()
    out[feature_columns] = scaler.transform(out[feature_columns])
    return out, scaler
