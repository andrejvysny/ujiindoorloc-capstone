"""Model evaluation utilities — metrics, confusion matrices, batched runs."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class EvalResult:
    name: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    classes: list[str]
    fit_seconds: float
    predict_seconds: float
    classification_report: str


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _classes_from(y_train, y_valid) -> list[str]:
    """Stable, sorted union of class labels (as strings) across train+valid."""
    a = pd.Series(y_train).astype(str)
    b = pd.Series(y_valid).astype(str)
    return sorted(set(a.unique()) | set(b.unique()))


def evaluate_classifier(
    model: ClassifierMixin,
    X_train,
    y_train,
    X_valid,
    y_valid,
    name: str,
) -> EvalResult:
    """Fit on train, predict on validation, return metrics + diagnostics."""
    y_train_arr = np.asarray(y_train)
    y_valid_arr = np.asarray(y_valid)

    t0 = time.perf_counter()
    model.fit(X_train, y_train_arr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_valid)
    pred_s = time.perf_counter() - t1

    classes = _classes_from(y_train_arr, y_valid_arr)
    metrics = compute_metrics(y_valid_arr, y_pred)
    report = classification_report(
        y_valid_arr,
        y_pred,
        labels=classes,
        zero_division=0,
        digits=3,
    )

    return EvalResult(
        name=name,
        metrics=metrics,
        y_true=np.asarray(y_valid_arr),
        y_pred=np.asarray(y_pred),
        classes=classes,
        fit_seconds=fit_s,
        predict_seconds=pred_s,
        classification_report=report,
    )


def create_confusion_matrix_df(result: EvalResult) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    cm = confusion_matrix(result.y_true, result.y_pred, labels=result.classes)
    return pd.DataFrame(cm, index=result.classes, columns=result.classes)


def evaluate_many_models(
    specs: Iterable[tuple[str, ClassifierMixin, Any, Any]],
    y_train,
    y_valid,
) -> list[EvalResult]:
    """Run a batch of (name, model, X_train, X_valid) specs sequentially.

    X_train/X_valid are passed *per-spec* because different models require
    different preprocessing variants (scaled vs tree).
    """
    results: list[EvalResult] = []
    for name, model, X_tr, X_va in specs:
        results.append(evaluate_classifier(model, X_tr, y_train, X_va, y_valid, name))
    return results


def results_to_metrics_df(results: list[EvalResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                **r.metrics,
                "fit_seconds": r.fit_seconds,
                "predict_seconds": r.predict_seconds,
                "n_classes": len(r.classes),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )


def save_metrics_table(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_classification_report(result: EvalResult, dir_path: Path | str) -> Path:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    out = dir_path / f"{result.name}.txt"
    out.write_text(result.classification_report)
    return out


def classification_report_to_dataframe(
    y_true, y_pred, labels=None
) -> pd.DataFrame:
    """Convert sklearn classification_report output to a tidy DataFrame."""
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    rows = []
    for label, stats in report_dict.items():
        if label == "accuracy":
            continue
        if isinstance(stats, dict):
            rows.append(
                {
                    "label": label,
                    "precision": stats.get("precision"),
                    "recall": stats.get("recall"),
                    "f1-score": stats.get("f1-score"),
                    "support": stats.get("support"),
                }
            )

    df = pd.DataFrame(rows)

    per_class = df[df["label"].apply(lambda x: x not in ("macro avg", "weighted avg"))]
    averages = df[df["label"].isin(("macro avg", "weighted avg"))]

    per_class_sorted = per_class.sort_values("label").reset_index(drop=True)
    result_df = pd.concat([per_class_sorted, averages], ignore_index=True)

    return result_df


def select_best_model(metrics_df: pd.DataFrame) -> pd.Series:
    """Select best model based on balanced_accuracy, macro_f1, and simplicity.

    Required columns: model, balanced_accuracy, macro_f1.
    Optional: simplicity_rank (lower = simpler). If absent, use model name.
    """
    required = {"model", "balanced_accuracy", "macro_f1"}
    if not required.issubset(metrics_df.columns):
        missing = required - set(metrics_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    if metrics_df.empty:
        raise ValueError("metrics_df is empty")

    sort_cols = ["balanced_accuracy", "macro_f1"]
    if "simplicity_rank" in metrics_df.columns:
        sort_cols.append("simplicity_rank")
    else:
        metrics_df = metrics_df.copy()
        metrics_df["_name_sort"] = metrics_df["model"]
        sort_cols.append("_name_sort")

    ascending = [False, False, "simplicity_rank" in metrics_df.columns]
    sorted_df = metrics_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if "_name_sort" in sorted_df.columns:
        sorted_df = sorted_df.drop(columns=["_name_sort"])

    return sorted_df.iloc[0]
