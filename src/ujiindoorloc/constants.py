"""Project-wide paths and magic numbers.

All paths are resolved relative to the *repo root* (the parent of `src/`),
so notebooks can import these regardless of their cwd.
"""
from __future__ import annotations

from pathlib import Path

# Repo root = parent of `src/`. constants.py lives at src/ujiindoorloc/constants.py.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Data
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRAIN_FILE: Path = RAW_DATA_DIR / "trainingData.csv"
VALIDATION_FILE: Path = RAW_DATA_DIR / "validationData.csv"

# Reports
REPORTS_DIR: Path = REPO_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"

# Models
MODELS_DIR: Path = REPO_ROOT / "models"

# Reproducibility
RANDOM_STATE: int = 42

# WAP signal handling — `100` is the UJIIndoorLoc sentinel for "WAP not detected".
# Real RSSI values are roughly in [-104, 0] dBm. We replace the sentinel with a
# value below the dynamic range so distance/scale-based models don't see it as
# the strongest signal possible.
MISSING_SIGNAL_VALUE: int = 100
REPLACEMENT_SIGNAL_VALUE: int = -110

# Column groups
WAP_PREFIX: str = "WAP"
METADATA_COLUMNS: tuple[str, ...] = (
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
)
TARGET_COLUMNS: tuple[str, ...] = ("BUILDINGID", "FLOOR")
COMBINED_TARGET: str = "building_floor"

# Columns we must NEVER use as predictors (location targets + metadata).
LEAKAGE_COLUMNS: tuple[str, ...] = (
    "LONGITUDE",
    "LATITUDE",
    "FLOOR",
    "BUILDINGID",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
)
