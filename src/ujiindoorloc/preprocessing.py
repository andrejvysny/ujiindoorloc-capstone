"""Preprocessing pipelines for UJIIndoorLoc WAP fingerprints.

Two variants are exposed:
  * scaled   — replace 100 → -110, drop train-constant cols, StandardScaler.
                For LR / LDA / QDA / kNN / PCA.
  * tree     — replace 100 → -110, drop train-constant cols, no scaling.
                For Decision Tree / Random Forest.

All fitting happens on the training data only. The same transformer instance
is then `.transform`-ed onto validation data — never re-fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import MISSING_SIGNAL_VALUE, REPLACEMENT_SIGNAL_VALUE


def replace_missing_signal_values(
    X: pd.DataFrame | np.ndarray,
    missing_value: int = MISSING_SIGNAL_VALUE,
    replacement_value: int = REPLACEMENT_SIGNAL_VALUE,
) -> pd.DataFrame | np.ndarray:
    """Replace the WAP-not-detected sentinel with a below-floor RSSI value."""
    if isinstance(X, pd.DataFrame):
        return X.where(X != missing_value, replacement_value)
    arr = np.asarray(X, dtype=float).copy()
    arr[arr == missing_value] = replacement_value
    return arr


def get_non_constant_columns(X_train: pd.DataFrame) -> list[str]:
    """Return WAP columns that vary across the training set (after sentinel
    replacement). Selection is fit on TRAIN ONLY — never validation."""
    X_clean = replace_missing_signal_values(X_train)
    nunique = X_clean.nunique(dropna=False)
    return nunique[nunique > 1].index.tolist()


def apply_column_filter(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only the given columns in both splits, preserving order."""
    return X_train[columns].copy(), X_valid[columns].copy()


# ---------------------------------------------------------------------------
# sklearn-compatible transformers
# ---------------------------------------------------------------------------


class MissingSignalReplacer(BaseEstimator, TransformerMixin):
    """Replace the UJIIndoorLoc 100-sentinel with `replacement_value`.

    Stateless w.r.t. fitting (no parameters learned from data) but kept as a
    Transformer so it composes inside an sklearn Pipeline.
    """

    def __init__(
        self,
        missing_value: int = MISSING_SIGNAL_VALUE,
        replacement_value: int = REPLACEMENT_SIGNAL_VALUE,
    ) -> None:
        self.missing_value = missing_value
        self.replacement_value = replacement_value

    def fit(self, X, y=None):  # noqa: D401 — sklearn API
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.where(X != self.missing_value, self.replacement_value).to_numpy()
        arr = np.asarray(X, dtype=float).copy()
        arr[arr == self.missing_value] = self.replacement_value
        return arr


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns that are constant on the training set (after replacement)."""

    def __init__(self) -> None:
        self.keep_idx_: np.ndarray | None = None

    def fit(self, X, y=None):
        arr = np.asarray(X)
        # column is "non-constant" if it has more than 1 unique value
        keep = []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            if np.unique(col).size > 1:
                keep.append(j)
        self.keep_idx_ = np.array(keep, dtype=int)
        return self

    def transform(self, X):
        if self.keep_idx_ is None:
            raise RuntimeError("ConstantColumnDropper must be fit before transform.")
        arr = np.asarray(X)
        return arr[:, self.keep_idx_]


def build_scaled_preprocessor() -> Pipeline:
    """Pipeline for models that need scaled, finite features (LR, LDA, kNN, PCA)."""
    return Pipeline(
        steps=[
            ("replace_missing", MissingSignalReplacer()),
            ("drop_constant", ConstantColumnDropper()),
            ("scale", StandardScaler()),
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    """Pipeline for tree models — no scaling needed; trees handle raw RSSI fine."""
    return Pipeline(
        steps=[
            ("replace_missing", MissingSignalReplacer()),
            ("drop_constant", ConstantColumnDropper()),
        ]
    )


# ---------------------------------------------------------------------------
# Convenience: fit both variants once and return ready-to-use arrays
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    X_train_scaled: np.ndarray
    X_valid_scaled: np.ndarray
    X_train_tree: np.ndarray
    X_valid_tree: np.ndarray
    kept_wap_columns: list[str]
    scaled_pipeline: Pipeline
    tree_pipeline: Pipeline


def prepare_classification_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> PreparedData:
    """Fit both pipelines on training data and apply to both splits.

    Returns numpy arrays (sklearn-friendly), the kept WAP column names (for
    later reporting/explainability) and the fitted pipelines themselves.
    """
    scaled = build_scaled_preprocessor()
    tree = build_tree_preprocessor()

    Xt_scaled = scaled.fit_transform(X_train)
    Xv_scaled = scaled.transform(X_valid)

    Xt_tree = tree.fit_transform(X_train)
    Xv_tree = tree.transform(X_valid)

    kept_idx = scaled.named_steps["drop_constant"].keep_idx_
    kept = [X_train.columns[i] for i in kept_idx]

    return PreparedData(
        X_train_scaled=Xt_scaled,
        X_valid_scaled=Xv_scaled,
        X_train_tree=Xt_tree,
        X_valid_tree=Xv_tree,
        kept_wap_columns=kept,
        scaled_pipeline=scaled,
        tree_pipeline=tree,
    )
