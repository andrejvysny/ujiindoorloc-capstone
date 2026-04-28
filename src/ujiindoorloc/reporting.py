"""Tiny IO helpers so notebooks don't repeat path-building boilerplate."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR


def ensure_report_dirs() -> dict[str, Path]:
    """Make sure all output directories exist; return them as a dict."""
    dirs = {
        "figures": FIGURES_DIR,
        "tables": TABLES_DIR,
        "models": MODELS_DIR,
        "processed": PROCESSED_DATA_DIR,
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_table(df: pd.DataFrame, name: str, subdir: Path | str | None = None) -> Path:
    """Save DataFrame as CSV under reports/tables/[subdir]/<name>.csv."""
    base = TABLES_DIR if subdir is None else TABLES_DIR / subdir
    base.mkdir(parents=True, exist_ok=True)
    path = base / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def save_figure_path(name: str, subdir: Path | str | None = None) -> Path:
    """Just compute the figure path (creating dirs); plotting fns handle the write."""
    base = FIGURES_DIR if subdir is None else FIGURES_DIR / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base / (name if name.endswith(".png") else f"{name}.png")
