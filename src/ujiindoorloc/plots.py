"""Plotting helpers — every function takes data + path, saves a PNG, returns Path."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluation import EvalResult, create_confusion_matrix_df

# Default look — keep it boring and readable.
sns.set_theme(style="whitegrid", context="notebook")


def _in_ipython() -> bool:
    """True iff we're running inside an IPython kernel (Jupyter notebook/lab)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _save(fig: plt.Figure, path: Path | str) -> Path:
    """Save the figure to PNG, also display it inline if inside Jupyter, then close.

    Inline display is required so notebook outputs embed the image. We close
    afterwards so memory doesn't accumulate when many plots are produced in
    one cell.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if _in_ipython():
        from IPython.display import display
        display(fig)
    plt.close(fig)
    return path


def plot_class_distribution(
    series: pd.Series,
    path: Path | str,
    title: str,
    xlabel: str = "class",
) -> Path:
    counts = series.astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(counts) + 4), 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    for t in ax.get_xticklabels():
        t.set_rotation(45)
        t.set_ha("right")
    return _save(fig, path)


def plot_detected_per_row_hist(
    detected: pd.Series,
    path: Path | str,
    title: str = "WAPs detected per scan",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(detected, bins=40, ax=ax, color="#4C72B0")
    ax.set_xlabel("# WAPs detected (signal != 100)")
    ax.set_ylabel("# scans")
    ax.set_title(title)
    return _save(fig, path)


def plot_top_features(
    df: pd.DataFrame,
    path: Path | str,
    title: str,
    feature_col: str = "feature",
    score_col: str = "f_score",
) -> Path:
    df = df.sort_values(score_col, ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(df))))
    ax.barh(df[feature_col], df[score_col], color="#4C72B0")
    ax.set_xlabel(score_col)
    ax.set_title(title)
    return _save(fig, path)


def plot_correlation_distribution(
    abs_corr_values: np.ndarray,
    path: Path | str,
    title: str = "Distribution of |correlation| between WAP features",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(abs_corr_values, bins=60, ax=ax, color="#4C72B0")
    ax.axvline(0.5, color="orange", linestyle="--", label="|r|=0.5")
    ax.axvline(0.8, color="red", linestyle="--", label="|r|=0.8")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("# pairs")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(
    result: EvalResult,
    path: Path | str,
    title: str | None = None,
    normalize: bool = False,
) -> Path:
    cm = create_confusion_matrix_df(result)
    if normalize:
        cm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(cm) + 2), max(5, 0.45 * len(cm) + 2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or f"Confusion matrix — {result.name}")
    return _save(fig, path)


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    path: Path | str,
    metrics: tuple[str, ...] = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"),
    title: str = "Model comparison",
) -> Path:
    long = metrics_df.melt(
        id_vars="model", value_vars=list(metrics), var_name="metric", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(metrics_df) + 4), 5))
    sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for t in ax.get_xticklabels():
        t.set_rotation(30)
        t.set_ha("right")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_pca_explained_variance(
    explained_variance_ratio: np.ndarray,
    path: Path | str,
    title: str = "PCA — explained variance",
) -> Path:
    cum = np.cumsum(explained_variance_ratio)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Per component")

    axes[1].plot(np.arange(1, len(cum) + 1), cum, color="#C44E52")
    axes[1].axhline(0.9, color="grey", linestyle="--", label="90%")
    axes[1].axhline(0.95, color="black", linestyle="--", label="95%")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title("Cumulative")
    axes[1].legend()
    fig.suptitle(title)
    return _save(fig, path)


def plot_2d_projection_scatter(
    projection: np.ndarray,
    y: pd.Series | np.ndarray,
    path: Path | str,
    title: str,
    x_label: str = "component_1",
    y_label: str = "component_2",
) -> Path:
    """Generic 2D projection scatter plot.

    Args:
        projection: Array with shape (n_samples, >=2).
        y: Class labels.
        path: Output path for PNG.
        title: Plot title.
        x_label: Label for x-axis (first projection column).
        y_label: Label for y-axis (second projection column).

    Returns:
        Path to saved PNG.
    """
    if projection.ndim != 2 or projection.shape[1] < 2:
        raise ValueError("projection must have at least 2 columns")
    df = pd.DataFrame({x_label: projection[:, 0], y_label: projection[:, 1], "class": pd.Series(y).astype(str)})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df, x=x_label, y=y_label, hue="class", s=10, alpha=0.6, linewidth=0, ax=ax
    )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, title="class")
    return _save(fig, path)


def plot_pca_2d_scatter(
    pcs: np.ndarray,
    y: pd.Series | np.ndarray,
    path: Path | str,
    title: str = "PCA — first 2 components",
) -> Path:
    return plot_2d_projection_scatter(pcs, y, path, title, x_label="PC1", y_label="PC2")


def plot_metric_vs_components(
    df: pd.DataFrame,
    path: Path | str,
    title: str = "Metric vs # components",
    metric: str = "balanced_accuracy",
    component_col: str = "n_components",
    model_col: str = "model",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sub in df.groupby(model_col):
        sub = sub.sort_values(component_col)
        ax.plot(sub[component_col], sub[metric], marker="o", label=name)
    ax.set_xlabel(component_col)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Spatial / heatmap plots (notebook 01b)
# ---------------------------------------------------------------------------


def plot_gps_scatter(
    df: pd.DataFrame,
    color_col: str,
    path: Path | str,
    title: str,
    sample: int | None = None,
    s: float = 6.0,
    alpha: float = 0.6,
) -> Path:
    """Scatter LON vs LAT, coloured by `color_col` (categorical or numeric).

    `sample` optionally limits the number of points for a faster figure.
    """
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df.assign(_c=df[color_col].astype(str)),
        x="LONGITUDE", y="LATITUDE", hue="_c",
        s=s, alpha=alpha, linewidth=0, ax=ax,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("LONGITUDE (m, projected)")
    ax.set_ylabel("LATITUDE (m, projected)")
    ax.legend(title=color_col, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    return _save(fig, path)


def plot_gps_facet_by_building(
    df: pd.DataFrame,
    color_col: str,
    path: Path | str,
    title: str,
    s: float = 6.0,
    alpha: float = 0.7,
) -> Path:
    """1×N facet (one panel per BUILDINGID) coloured by `color_col`.

    Useful to show that within a building the floors are NOT geographically
    separated — it is WiFi that distinguishes them.
    """
    buildings = sorted(df["BUILDINGID"].unique())
    fig, axes = plt.subplots(1, len(buildings), figsize=(5 * len(buildings), 5), sharey=True)
    if len(buildings) == 1:
        axes = [axes]
    for ax, b in zip(axes, buildings):
        sub = df[df["BUILDINGID"] == b].assign(_c=df[color_col].astype(str))
        sns.scatterplot(
            data=sub, x="LONGITUDE", y="LATITUDE", hue="_c",
            s=s, alpha=alpha, linewidth=0, ax=ax, legend=(ax is axes[-1]),
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"BUILDINGID = {b}")
        ax.set_xlabel("LONGITUDE")
        if ax is axes[0]:
            ax.set_ylabel("LATITUDE")
        else:
            ax.set_ylabel("")
    if axes[-1].get_legend() is not None:
        axes[-1].legend(title=color_col, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_gps_train_vs_valid(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    path: Path | str,
    title: str = "GPS overlap — training vs validation",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(train["LONGITUDE"], train["LATITUDE"], s=4, alpha=0.25,
               color="#4C72B0", label=f"train (n={len(train)})", linewidth=0)
    ax.scatter(valid["LONGITUDE"], valid["LATITUDE"], s=10, alpha=0.7,
               color="#C44E52", label=f"validation (n={len(valid)})", linewidth=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("LONGITUDE")
    ax.set_ylabel("LATITUDE")
    ax.legend()
    return _save(fig, path)


def plot_gps_density_hexbin(
    df: pd.DataFrame,
    path: Path | str,
    title: str = "Scan density across the campus",
    gridsize: int = 60,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(df["LONGITUDE"], df["LATITUDE"], gridsize=gridsize,
                   cmap="viridis", mincnt=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("LONGITUDE")
    ax.set_ylabel("LATITUDE")
    fig.colorbar(hb, ax=ax, label="# scans per cell")
    return _save(fig, path)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    path: Path | str,
    title: str,
    cluster_order: list[str] | None = None,
    annot: bool = False,
    cmap: str = "coolwarm",
    center: float = 0.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Path:
    """Heatmap of a correlation matrix.

    If `cluster_order` is given, rows/cols are reordered accordingly so a
    block structure (typically per-building / per-floor AP groups) becomes
    visible. Pass annot=True only on small (< 30) matrices.
    """
    if cluster_order is not None:
        corr = corr.loc[cluster_order, cluster_order]
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(16, max(6, n * 0.06 + 4)),
                                    min(14, max(5, n * 0.06 + 3))))
    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        annot=annot, fmt=".2f" if annot else "",
        cbar_kws={"label": "Pearson r"},
        xticklabels=False if n > 60 else True,
        yticklabels=False if n > 60 else True,
        square=True,
    )
    ax.set_title(title)
    return _save(fig, path)


def plot_class_fingerprint_heatmap(
    centroids: pd.DataFrame,
    path: Path | str,
    title: str = "Class-mean RSSI fingerprint (top WAPs)",
    cmap: str = "viridis",
) -> Path:
    """Rows = classes, cols = WAPs, values = mean RSSI (incl. -110 for not detected)."""
    fig, ax = plt.subplots(figsize=(max(8, 0.18 * centroids.shape[1] + 4),
                                    max(4, 0.35 * centroids.shape[0] + 2)))
    sns.heatmap(
        centroids, ax=ax, cmap=cmap, cbar_kws={"label": "mean RSSI (dBm; -110 = sentinel)"},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("WAP")
    ax.set_ylabel("class")
    ax.set_title(title)
    for t in ax.get_xticklabels():
        t.set_rotation(90)
        t.set_fontsize(7)
    return _save(fig, path)


def plot_detection_rate_heatmap(
    rates: pd.DataFrame,
    path: Path | str,
    title: str = "Per-class WAP detection rate (top WAPs)",
    cmap: str = "magma",
) -> Path:
    """Same shape as the fingerprint heatmap, but values are detection rate ∈ [0,1]."""
    fig, ax = plt.subplots(figsize=(max(8, 0.18 * rates.shape[1] + 4),
                                    max(4, 0.35 * rates.shape[0] + 2)))
    sns.heatmap(
        rates, ax=ax, cmap=cmap, vmin=0.0, vmax=1.0,
        cbar_kws={"label": "detection rate"},
        xticklabels=True, yticklabels=True,
    )
    ax.set_xlabel("WAP")
    ax.set_ylabel("class")
    ax.set_title(title)
    for t in ax.get_xticklabels():
        t.set_rotation(90)
        t.set_fontsize(7)
    return _save(fig, path)


def plot_class_centroid_distance_heatmap(
    distances: pd.DataFrame,
    path: Path | str,
    title: str = "Class centroid distance (predicted confusion structure)",
    cmap: str = "magma_r",
) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(distances) + 2),
                                    max(6, 0.45 * len(distances) + 2)))
    sns.heatmap(distances, ax=ax, cmap=cmap, annot=True, fmt=".0f",
                cbar_kws={"label": "Euclidean distance (RSSI space)"},
                square=True)
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("class")
    return _save(fig, path)


def plot_wap_shift_scatter(
    shift_df: pd.DataFrame,
    path: Path | str,
    title: str = "Per-WAP train ↔ validation shift",
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Detection rate
    axes[0].scatter(shift_df["train_detection_rate"], shift_df["valid_detection_rate"],
                    s=12, alpha=0.6, color="#4C72B0", linewidth=0)
    lo = 0.0
    hi = max(shift_df["train_detection_rate"].max(),
             shift_df["valid_detection_rate"].max(), 1e-3) * 1.05
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("train detection rate")
    axes[0].set_ylabel("validation detection rate")
    axes[0].set_title("Detection rate")
    axes[0].legend(loc="lower right")

    # Mean RSSI (drop NaNs — WAPs never detected in one of the splits)
    sub = shift_df.dropna(subset=["train_mean_rssi", "valid_mean_rssi"])
    axes[1].scatter(sub["train_mean_rssi"], sub["valid_mean_rssi"],
                    s=12, alpha=0.6, color="#C44E52", linewidth=0)
    if len(sub):
        lo2 = float(min(sub["train_mean_rssi"].min(), sub["valid_mean_rssi"].min()))
        hi2 = float(max(sub["train_mean_rssi"].max(), sub["valid_mean_rssi"].max()))
        axes[1].plot([lo2, hi2], [lo2, hi2], "k--", linewidth=1, label="y = x")
        axes[1].set_xlim(lo2 - 1, hi2 + 1)
        axes[1].set_ylim(lo2 - 1, hi2 + 1)
    axes[1].set_xlabel("train mean RSSI (dBm, detected only)")
    axes[1].set_ylabel("validation mean RSSI (dBm, detected only)")
    axes[1].set_title(f"Mean RSSI (n={len(sub)} WAPs detected in both)")
    axes[1].legend(loc="lower right")

    fig.suptitle(title)
    return _save(fig, path)


def plot_sparsity_heatmap(
    df: pd.DataFrame,
    path: Path | str,
    n_rows: int = 200,
    title: str = "Sparsity — sample × WAP detection mask",
) -> Path:
    """Subsampled binary heatmap: 1 = WAP detected, 0 = not detected."""
    from .data_loading import get_wap_columns

    sub = df.sample(n=min(n_rows, len(df)), random_state=0)
    wap = sub[get_wap_columns(df)]
    detected = (wap != 100).astype(int)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        detected.values, ax=ax, cmap="Greys", cbar_kws={"label": "detected (1) vs not (0)"},
        xticklabels=False, yticklabels=False,
    )
    ax.set_xlabel(f"WAP (n={detected.shape[1]})")
    ax.set_ylabel(f"sample (n={detected.shape[0]} of {len(df)})")
    ax.set_title(title)
    return _save(fig, path)
