"""EDA helpers — every function returns a tidy DataFrame so notebooks stay short."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import f_classif

from .constants import COMBINED_TARGET, METADATA_COLUMNS, MISSING_SIGNAL_VALUE
from .data_loading import create_building_floor_target, get_wap_columns
from .preprocessing import get_non_constant_columns, replace_missing_signal_values


def summarize_dataset_shape(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    wap_cols = get_wap_columns(train)
    meta = [c for c in METADATA_COLUMNS if c in train.columns]
    return pd.DataFrame(
        {
            "split": ["train", "validation"],
            "rows": [len(train), len(valid)],
            "columns": [train.shape[1], valid.shape[1]],
            "wap_columns": [len(wap_cols), len(get_wap_columns(valid))],
            "metadata_columns": [len(meta), len([c for c in METADATA_COLUMNS if c in valid.columns])],
            "duplicated_rows": [int(train.duplicated().sum()), int(valid.duplicated().sum())],
            "nan_cells": [int(train.isna().sum().sum()), int(valid.isna().sum().sum())],
        }
    )


def summarize_missingness(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Counts of true NaNs vs. the encoded `100` sentinel in WAP columns."""
    out = []
    for name, df in [("train", train), ("validation", valid)]:
        wap = df[get_wap_columns(df)]
        total = wap.size
        sentinel = int((wap == MISSING_SIGNAL_VALUE).sum().sum())
        out.append(
            {
                "split": name,
                "total_wap_cells": total,
                "nan_cells": int(wap.isna().sum().sum()),
                "sentinel_100_cells": sentinel,
                "sentinel_100_pct": sentinel / total * 100.0,
            }
        )
    return pd.DataFrame(out)


def summarize_wap_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row detection statistics (how many WAPs were heard per scan)."""
    wap = df[get_wap_columns(df)]
    detected_per_row = (wap != MISSING_SIGNAL_VALUE).sum(axis=1)
    return pd.DataFrame(
        {
            "metric": [
                "rows",
                "mean_detected_per_row",
                "median_detected_per_row",
                "min_detected_per_row",
                "max_detected_per_row",
                "rows_with_zero_detected",
                "pct_detected_overall",
                "pct_sentinel_overall",
            ],
            "value": [
                len(df),
                float(detected_per_row.mean()),
                float(detected_per_row.median()),
                int(detected_per_row.min()),
                int(detected_per_row.max()),
                int((detected_per_row == 0).sum()),
                float((wap != MISSING_SIGNAL_VALUE).sum().sum() / wap.size * 100),
                float((wap == MISSING_SIGNAL_VALUE).sum().sum() / wap.size * 100),
            ],
        }
    )


def detected_per_row(df: pd.DataFrame) -> pd.Series:
    wap = df[get_wap_columns(df)]
    return (wap != MISSING_SIGNAL_VALUE).sum(axis=1).rename("detected_waps")


def summarize_wap_coverage(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Which WAPs are ever observed in train vs. validation?"""
    wap_cols = get_wap_columns(train)
    train_seen = set((train[wap_cols] != MISSING_SIGNAL_VALUE).any().loc[lambda s: s].index)
    valid_seen = set((valid[wap_cols] != MISSING_SIGNAL_VALUE).any().loc[lambda s: s].index)

    rows = [
        ("total_wap_columns", len(wap_cols)),
        ("seen_in_train", len(train_seen)),
        ("never_seen_in_train", len(wap_cols) - len(train_seen)),
        ("seen_in_validation", len(valid_seen)),
        ("seen_in_both", len(train_seen & valid_seen)),
        ("only_in_train", len(train_seen - valid_seen)),
        ("only_in_validation", len(valid_seen - train_seen)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_target_distribution(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Stack train+valid distributions of building/floor/building_floor."""
    out = []
    for name, df in [("train", train), ("validation", valid)]:
        bf = create_building_floor_target(df)
        for tgt_name, series in [
            ("BUILDINGID", df["BUILDINGID"].astype(int)),
            ("FLOOR", df["FLOOR"].astype(int)),
            (COMBINED_TARGET, bf.astype(str)),
        ]:
            counts = series.value_counts().sort_index()
            for cls, n in counts.items():
                out.append(
                    {
                        "split": name,
                        "target": tgt_name,
                        "class": str(cls),
                        "count": int(n),
                        "pct": float(n / len(series) * 100),
                    }
                )
    return pd.DataFrame(out)


def summarize_metadata_shift(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Compare metadata coverage between train and validation."""
    rows = []
    for col in ("USERID", "PHONEID", "SPACEID", "TIMESTAMP"):
        if col not in train.columns or col not in valid.columns:
            continue
        t = train[col]
        v = valid[col]
        if col == "TIMESTAMP":
            rows.append(
                {
                    "metric": f"{col}_min",
                    "train": int(t.min()),
                    "validation": int(v.min()),
                }
            )
            rows.append(
                {
                    "metric": f"{col}_max",
                    "train": int(t.max()),
                    "validation": int(v.max()),
                }
            )
        else:
            rows.append(
                {
                    "metric": f"{col}_unique",
                    "train": int(t.nunique()),
                    "validation": int(v.nunique()),
                }
            )
            rows.append(
                {
                    "metric": f"{col}_only_in_train",
                    "train": int(len(set(t) - set(v))),
                    "validation": 0,
                }
            )
            rows.append(
                {
                    "metric": f"{col}_only_in_validation",
                    "train": 0,
                    "validation": int(len(set(v) - set(t))),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# WAP correlation analysis
# ---------------------------------------------------------------------------


def compute_wap_correlations(
    X_train: pd.DataFrame,
    top_k_pairs: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary, top_pairs) for absolute pairwise correlation of WAPs.

    Operates on training only. Sentinel is replaced before correlation;
    constant columns are dropped (they have zero variance → undefined corr).
    """
    kept_cols = get_non_constant_columns(X_train)
    X_clean = replace_missing_signal_values(X_train[kept_cols])
    corr_arr = np.abs(X_clean.corr().to_numpy(copy=True))
    np.fill_diagonal(corr_arr, np.nan)
    corr = pd.DataFrame(corr_arr, index=kept_cols, columns=kept_cols)

    # upper triangle only
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    flat = upper.stack().rename("abs_corr")

    summary = pd.DataFrame(
        {
            "metric": [
                "n_features_after_constant_filter",
                "n_pairs",
                "mean_abs_corr",
                "median_abs_corr",
                "p95_abs_corr",
                "p99_abs_corr",
                "pairs_above_0.5",
                "pairs_above_0.7",
                "pairs_above_0.8",
            ],
            "value": [
                len(kept_cols),
                int(flat.size),
                float(flat.mean()),
                float(flat.median()),
                float(flat.quantile(0.95)),
                float(flat.quantile(0.99)),
                int((flat > 0.5).sum()),
                int((flat > 0.7).sum()),
                int((flat > 0.8).sum()),
            ],
        }
    )
    top_pairs = (
        flat.sort_values(ascending=False)
        .head(top_k_pairs)
        .reset_index()
        .rename(columns={"level_0": "wap_a", "level_1": "wap_b"})
    )
    return summary, top_pairs


def correlation_distribution(X_train: pd.DataFrame) -> np.ndarray:
    """Flat array of |corr| values for plotting a histogram."""
    kept = get_non_constant_columns(X_train)
    X_clean = replace_missing_signal_values(X_train[kept])
    corr = X_clean.corr().abs().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    return corr[iu]


# ---------------------------------------------------------------------------
# Feature relevance
# ---------------------------------------------------------------------------


def compute_anova_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 30,
) -> pd.DataFrame:
    """ANOVA F-score per WAP column, sorted descending.

    Sentinel is replaced first; constant columns dropped. NaN scores are
    treated as 0 (happens when a column has no variance for some classes).
    """
    kept = get_non_constant_columns(X_train)
    X_clean = replace_missing_signal_values(X_train[kept]).to_numpy()
    f_scores, p_values = f_classif(X_clean, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0)
    df = pd.DataFrame(
        {"feature": kept, "f_score": f_scores, "p_value": p_values}
    )
    df = df.sort_values("f_score", ascending=False).reset_index(drop=True)
    return df.head(top_k)


# ---------------------------------------------------------------------------
# Spatial / class-fingerprint helpers (notebook 01b)
# ---------------------------------------------------------------------------


def gps_bounds(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Per-split min/max of LONGITUDE/LATITUDE — used for the map sanity check.

    LON/LAT are projected metres in UJIIndoorLoc, not lat/lon degrees.
    """
    rows = []
    for name, df in [("train", train), ("validation", valid)]:
        rows.append(
            {
                "split": name,
                "lon_min": float(df["LONGITUDE"].min()),
                "lon_max": float(df["LONGITUDE"].max()),
                "lat_min": float(df["LATITUDE"].min()),
                "lat_max": float(df["LATITUDE"].max()),
                "rows": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def compute_class_centroids(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Mean RSSI per class after sentinel replacement (rows = classes, cols = WAPs).

    Note: cells include scans where the WAP was *not detected* (those count as
    -110 here). A near-`-110` cell therefore means "rarely detected", not
    "weak signal". Use `compute_wap_detection_rate_per_class` for the
    availability-only view.
    """
    kept = get_non_constant_columns(X_train)
    X_clean = replace_missing_signal_values(X_train[kept])
    df = X_clean.copy()
    df[COMBINED_TARGET] = pd.Series(y_train).astype(str).values
    centroids = df.groupby(COMBINED_TARGET, observed=True).mean()
    return centroids.sort_index()


def compute_wap_detection_rate_per_class(
    df: pd.DataFrame,
    target_col: str = COMBINED_TARGET,
) -> pd.DataFrame:
    """% of scans where each WAP is detected, broken down by class.

    `target_col` may be a name already present in `df`, or `COMBINED_TARGET`
    in which case the building_floor target is built on the fly.
    """
    wap_cols = get_wap_columns(df)
    detected = (df[wap_cols] != MISSING_SIGNAL_VALUE).astype(float)
    if target_col == COMBINED_TARGET and target_col not in df.columns:
        target = create_building_floor_target(df).astype(str)
    else:
        target = df[target_col].astype(str)
    detected = pd.concat(
        [detected, pd.Series(target.values, name=target_col, index=detected.index)],
        axis=1,
    )
    rates = detected.groupby(target_col, observed=True).mean()
    return rates.sort_index()


def compute_per_wap_train_valid_shift(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> pd.DataFrame:
    """One row per WAP: detection rate + mean detected RSSI in each split."""
    wap_cols = get_wap_columns(train)
    rows = []
    for col in wap_cols:
        t = train[col].to_numpy()
        v = valid[col].to_numpy()
        t_det = t != MISSING_SIGNAL_VALUE
        v_det = v != MISSING_SIGNAL_VALUE
        rows.append(
            {
                "wap": col,
                "train_detection_rate": float(t_det.mean()),
                "valid_detection_rate": float(v_det.mean()),
                # Mean over *detected* scans only (NaN if never detected) —
                # avoids the -110 sentinel dragging the mean down.
                "train_mean_rssi": float(t[t_det].mean()) if t_det.any() else np.nan,
                "valid_mean_rssi": float(v[v_det].mean()) if v_det.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_class_centroid_distances(centroids: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Euclidean distance between class centroids (symmetric, classes×classes)."""
    arr = centroids.to_numpy()
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq = (arr * arr).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * arr @ arr.T
    d2 = np.maximum(d2, 0.0)
    dist = np.sqrt(d2)
    return pd.DataFrame(dist, index=centroids.index, columns=centroids.index)


def clustered_corr_order(corr: pd.DataFrame) -> list[str]:
    """Hierarchical-clustering leaf order for a correlation matrix.

    Uses average linkage on `1 - |corr|` so highly-correlated columns end up
    adjacent — turns a noisy heatmap into a block-structured one.
    """
    arr = np.abs(corr.to_numpy(copy=True))
    np.fill_diagonal(arr, 1.0)
    # numerical safety: clip then make symmetric, in case input wasn't exactly symmetric
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr + arr.T) / 2.0
    dist = 1.0 - arr
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    order = leaves_list(z)
    return [corr.index[i] for i in order]
