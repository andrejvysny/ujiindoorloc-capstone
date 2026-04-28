"""Load raw UJIIndoorLoc CSVs and split into features/targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import (
    COMBINED_TARGET,
    LEAKAGE_COLUMNS,
    TRAIN_FILE,
    VALIDATION_FILE,
    WAP_PREFIX,
)


@dataclass(frozen=True)
class RawData:
    train: pd.DataFrame
    valid: pd.DataFrame


def load_raw_data(
    train_path: Path | str = TRAIN_FILE,
    valid_path: Path | str = VALIDATION_FILE,
) -> RawData:
    """Load training + validation CSVs as raw DataFrames (no preprocessing)."""
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    return RawData(train=train, valid=valid)


def get_wap_columns(df: pd.DataFrame) -> list[str]:
    """Return the WAP* columns in their original order."""
    return [c for c in df.columns if c.startswith(WAP_PREFIX)]


def create_building_floor_target(df: pd.DataFrame) -> pd.Series:
    """Build the combined `B<bid>_F<floor>` multiclass target.

    Returned as a categorical series so the class set is stable across
    train/validation even when validation is missing some classes.
    """
    bid = df["BUILDINGID"].astype(int).astype(str)
    flr = df["FLOOR"].astype(int).astype(str)
    return ("B" + bid + "_F" + flr).astype("category").rename(COMBINED_TARGET)


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    target_name: str


def split_features_targets(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_name: str = COMBINED_TARGET,
) -> SplitData:
    """Split DataFrames into WAP-only features and the requested target.

    `target_name` may be:
      - "building_floor" (combined, the default + project main target),
      - "BUILDINGID" or "FLOOR" (raw column, helper targets only).
    """
    wap_cols = get_wap_columns(train_df)
    # Defensive: validation must use the SAME WAP columns in the same order.
    if get_wap_columns(valid_df) != wap_cols:
        raise ValueError("WAP columns differ between training and validation data.")

    # Leakage guard — make sure no leakage column slipped into the feature list.
    bad = set(wap_cols) & set(LEAKAGE_COLUMNS)
    if bad:
        raise ValueError(f"Leakage columns present in WAP feature list: {sorted(bad)}")

    X_train = train_df[wap_cols].copy()
    X_valid = valid_df[wap_cols].copy()

    if target_name == COMBINED_TARGET:
        y_train = create_building_floor_target(train_df)
        y_valid = create_building_floor_target(valid_df)
        # align categorical class set across train+valid so plots/CMs match
        all_classes = sorted(set(y_train.cat.categories) | set(y_valid.cat.categories))
        y_train = y_train.cat.set_categories(all_classes)
        y_valid = y_valid.cat.set_categories(all_classes)
    elif target_name in ("BUILDINGID", "FLOOR"):
        y_train = train_df[target_name].astype(int).rename(target_name)
        y_valid = valid_df[target_name].astype(int).rename(target_name)
    else:
        raise ValueError(f"Unsupported target_name: {target_name!r}")

    return SplitData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        target_name=target_name,
    )
