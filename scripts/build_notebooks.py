"""Generate the 6 capstone notebooks from cell lists.

Run with:  uv run python scripts/build_notebooks.py
Outputs:   notebooks/00_project_overview.ipynb,
           01_data_audit_and_eda.ipynb (numeric + spatial + heatmap EDA in one),
           02_preprocessing_and_targets.ipynb,
           03_scenario_2_model_comparison.ipynb,
           04_scenario_5_dimensionality_reduction.ipynb,
           05_final_comparison_and_interpretation.ipynb
           (each overwritten on every run).
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


def write_nb(name: str, cells: list[nbf.NotebookNode]) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ujiindoorloc-capstone)",
            "language": "python",
            "name": "ujiindoorloc-capstone",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    return out


# ---------------------------------------------------------------------------
# Common bootstrap snippet placed at the top of every analytical notebook
# ---------------------------------------------------------------------------

BOOTSTRAP = """
# --- bootstrap: make src/ importable when notebook started outside `uv run` ---
import sys
from pathlib import Path

_HERE = Path.cwd()
for parent in [_HERE, *_HERE.parents]:
    if (parent / "src" / "ujiindoorloc").is_dir():
        if str(parent / "src") not in sys.path:
            sys.path.insert(0, str(parent / "src"))
        break

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
"""

# ===========================================================================
# 00 — Project Overview
# ===========================================================================

nb00 = [
    md("""
    # 00 — Indoor Building & Floor Classification (UJIIndoorLoc)

    **Capstone project — FIIT OZNAL.**
    """),
    md("""
    ## Problem

    Indoor GPS is unreliable. WiFi RSSI fingerprints (signal strengths from
    nearby access points) can be used to estimate **where** a user is *inside*
    a building. We frame this as **multiclass classification** of the
    `(building, floor)` pair from the WAP signal vector.

    ## Dataset — UJIIndoorLoc (UCI)

    * 19,937 training scans + 1,111 validation scans
    * 520 WAP RSSI features per scan (`WAP001 … WAP520`)
    * 3 buildings × up to 5 floors → 13 actual `(building, floor)` classes
    * **`100` is not a real signal** — it is the sentinel for *"WAP not
      detected"*. Real RSSI is roughly in `[-104, 0]` dBm.

    ## Main target

    ```
    building_floor = "B" + BUILDINGID + "_F" + FLOOR    # e.g. "B1_F2"
    ```

    Helper targets for sanity checks only: `BUILDINGID`, `FLOOR`.

    ## Selected scenarios (per assignment)

    * **Scenario 2** — parametric vs. non-parametric classifiers.
    * **Scenario 5** — dimensionality reduction (PCA / LDA) before
      classification.

    ## Evaluation philosophy

    Classes are imbalanced and the validation split is a real *time / device /
    user* shift. Headline metrics:

    * **balanced accuracy**
    * **macro F1**
    * **confusion matrix** (for diagnosis)

    Plain accuracy is reported but is *not* the primary criterion.

    ## Out of scope (this pass)

    * Coordinate regression (`LONGITUDE`, `LATITUDE`).
    * Shiny app (will be built separately later).
    * Heavy feature selection / model tuning — code is modular so it can be
      added without rework.

    ## Reproduce

    ```bash
    uv sync
    uv run python -m ipykernel install --user --name ujiindoorloc-capstone
    uv run jupyter lab
    ```

    Then run notebooks `00 → 01 → 02 → 03 → 04 → 05` top-to-bottom.
    """),
]

write_nb("00_project_overview.ipynb", nb00)


# ===========================================================================
# 01 — Data Audit and EDA
# ===========================================================================

nb01 = [
    md("""
    # 01 — Data Audit & EDA

    One place for all exploratory analysis: numeric audit, target structure,
    train↔validation shift, spatial geometry, WAP correlation matrices,
    per-class fingerprints, and predicted-confusion structure.

    Every plot/table here supports a downstream modelling decision.

    > **`LONGITUDE` / `LATITUDE` are used in §8–9 for visualisation only.**
    > They stay on `LEAKAGE_COLUMNS` and never enter any model — see the
    > assertion in the imports cell below.
    """),
    code(BOOTSTRAP),
    code("""
    import numpy as np
    import pandas as pd

    from ujiindoorloc.constants import (
        COMBINED_TARGET, FIGURES_DIR, TABLES_DIR, LEAKAGE_COLUMNS,
    )
    from ujiindoorloc.data_loading import (
        create_building_floor_target,
        get_wap_columns,
        load_raw_data,
        split_features_targets,
    )
    from ujiindoorloc.preprocessing import (
        get_non_constant_columns, replace_missing_signal_values,
    )
    from ujiindoorloc import eda
    from ujiindoorloc import plots as p
    from ujiindoorloc.reporting import ensure_report_dirs, save_table, save_figure_path

    # Leakage guard — LON/LAT used below for plotting must remain banned as predictors.
    assert "LONGITUDE" in LEAKAGE_COLUMNS and "LATITUDE" in LEAKAGE_COLUMNS

    ensure_report_dirs()
    raw = load_raw_data()
    train, valid = raw.train, raw.valid

    # Per-row "# WAPs detected" — handy as a colour for the GPS scatter in §9.
    train = train.assign(detected_waps=eda.detected_per_row(train).values)
    valid = valid.assign(detected_waps=eda.detected_per_row(valid).values)

    print("Train:", train.shape, "  Validation:", valid.shape)
    """),
    md("## 1. Basic structure"),
    code("""
    shape_df = eda.summarize_dataset_shape(train, valid)
    save_table(shape_df, "dataset_shape.csv")
    shape_df
    """),
    code("""
    print("First 5 WAP columns :", get_wap_columns(train)[:5])
    print("Last 5 WAP columns  :", get_wap_columns(train)[-5:])
    print("Non-WAP columns     :", [c for c in train.columns if not c.startswith("WAP")])
    """),
    md("""
    ## 2. Encoded missingness — the `100` sentinel

    Officially the dataset has no NaNs. But the integer `100` in any WAP cell
    means *"WAP was not detected during this scan"*. Real RSSI values are
    negative. Treating `100` as a strong positive signal would catastrophically
    break any distance/scale-based model.
    """),
    code("""
    miss_df = eda.summarize_missingness(train, valid)
    save_table(miss_df, "missingness_summary.csv")
    miss_df
    """),
    md("## 3. WAP signal — per-row detection"),
    code("""
    det_train = eda.summarize_wap_detection(train).assign(split="train")
    det_valid = eda.summarize_wap_detection(valid).assign(split="validation")
    det_df = pd.concat([det_train, det_valid], ignore_index=True)
    save_table(det_df, "wap_detection_summary.csv")
    det_df
    """),
    code("""
    p.plot_detected_per_row_hist(
        eda.detected_per_row(train),
        save_figure_path("waps_detected_per_row_train.png"),
        title="WAPs detected per scan — training set",
    )
    """),
    md("## 4. WAP coverage train ↔ validation"),
    code("""
    cov_df = eda.summarize_wap_coverage(train, valid)
    save_table(cov_df, "wap_coverage_summary.csv")
    cov_df
    """),
    md("""
    > Some WAPs that appear in validation are **never seen in training**.
    > Models cannot learn anything about those, so a validation drop is
    > expected — this is a real domain-shift artifact, not a bug.
    """),
    md("## 5. Target distributions"),
    code("""
    dist_df = eda.summarize_target_distribution(train, valid)
    save_table(dist_df, "class_distribution.csv")
    dist_df.head(20)
    """),
    code("""
    p.plot_class_distribution(
        train["BUILDINGID"].astype(int),
        save_figure_path("building_distribution.png"),
        "BUILDINGID — training",
        xlabel="BUILDINGID",
    )
    p.plot_class_distribution(
        train["FLOOR"].astype(int),
        save_figure_path("floor_distribution.png"),
        "FLOOR — training",
        xlabel="FLOOR",
    )
    p.plot_class_distribution(
        create_building_floor_target(train),
        save_figure_path("building_floor_distribution.png"),
        "building_floor — training (main target)",
        xlabel="building_floor",
    )
    """),
    md("""
    > Building 2 dominates and floor 4 is small. This is why we will report
    > **balanced accuracy** and **macro F1** alongside plain accuracy.
    """),
    md("""
    ## 6. Metadata shift

    The validation split is **not** a random split. It comes from a different
    time window, fewer users (1 vs 18), different phones. The numbers below
    quantify that shift.
    """),
    code("""
    shift_df = eda.summarize_metadata_shift(train, valid)
    save_table(shift_df, "metadata_shift_summary.csv")
    shift_df
    """),
    md("""
    ## 7. WAP correlation analysis

    WAPs in the same physical area share signal patterns → high pairwise
    correlation → multicollinearity. This motivates regularised models
    (logistic regression with L2) and dimensionality reduction (Scenario 5).
    """),
    code("""
    split = split_features_targets(train, valid, target_name=COMBINED_TARGET)
    corr_summary, top_pairs = eda.compute_wap_correlations(split.X_train, top_k_pairs=20)
    save_table(corr_summary, "wap_correlation_summary.csv")
    save_table(top_pairs, "top_correlated_wap_pairs.csv")
    corr_summary
    """),
    code("""
    top_pairs.head(20)
    """),
    code("""
    abs_corr_values = eda.correlation_distribution(split.X_train)
    p.plot_correlation_distribution(
        abs_corr_values,
        save_figure_path("wap_correlation_distribution.png"),
    )
    """),
    md("""
    ## 8. Campus geometry — buildings & floors

    LON/LAT visualised. The first plot proves the buildings are spatially
    separated. The 1×3 facet then proves that *within* a building, floors
    are **not** geographically separated (a person on floor 0 stands above
    the same LON/LAT as one on floor 3) — which is precisely why we need
    WiFi to tell them apart.
    """),
    code("""
    bounds = eda.gps_bounds(train, valid)
    save_table(bounds, "gps_bounds.csv")
    bounds
    """),
    md("> Coordinates are projected metres (UTM-like), not lat/lon degrees. Campus footprint ≈ 390 m × 270 m."),
    code("""
    p.plot_gps_scatter(
        train, color_col="BUILDINGID",
        path=save_figure_path("gps_buildings.png"),
        title="Campus footprint — coloured by BUILDINGID",
    )
    p.plot_gps_facet_by_building(
        train, color_col="FLOOR",
        path=save_figure_path("gps_facet_by_floor.png"),
        title="Per-building floor layout (LON/LAT does NOT separate floors)",
    )
    """),
    md("""
    ## 9. Train ↔ validation spatial overlap & scan density
    """),
    code("""
    p.plot_gps_train_vs_valid(train, valid, save_figure_path("gps_train_vs_valid.png"))
    """),
    md("""
    > Validation samples sit *inside* the training footprint — confirming
    > that the train↔valid shift documented in §7 is a **device/time/user**
    > shift, not a spatial one. That justifies keeping the official split.
    """),
    code("""
    p.plot_gps_density_hexbin(
        train, save_figure_path("gps_density_hexbin.png"),
        title="Training scan density (hexbin)",
    )
    p.plot_gps_scatter(
        train, color_col="detected_waps",
        path=save_figure_path("gps_detected_waps.png"),
        title="LON/LAT coloured by # WAPs detected per scan (scan quality)",
        sample=8000,
    )
    """),
    md("""
    ## 10. WAP correlation analysis

    WAPs in the same physical area share signal patterns → high pairwise
    correlation → multicollinearity. This motivates regularised models
    (logistic regression with L2) and dimensionality reduction (Scenario 5).
    """),
    code("""
    split = split_features_targets(train, valid, target_name=COMBINED_TARGET)
    corr_summary, top_pairs = eda.compute_wap_correlations(split.X_train, top_k_pairs=20)
    save_table(corr_summary, "wap_correlation_summary.csv")
    save_table(top_pairs, "top_correlated_wap_pairs.csv")
    corr_summary
    """),
    code("""
    top_pairs.head(20)
    """),
    code("""
    abs_corr_values = eda.correlation_distribution(split.X_train)
    p.plot_correlation_distribution(
        abs_corr_values,
        save_figure_path("wap_correlation_distribution.png"),
    )
    """),
    md("""
    ## 11. Full WAP × WAP correlation heatmap (clustered)

    The histogram above told us *how many* pairs are highly correlated.
    This is the **matrix** version, reordered by hierarchical clustering so
    the AP groups become visually obvious as block structure.
    """),
    code("""
    kept = get_non_constant_columns(split.X_train)
    X_clean = replace_missing_signal_values(split.X_train[kept])
    corr_full = X_clean.corr()
    print("corr matrix:", corr_full.shape)

    cluster_order = eda.clustered_corr_order(corr_full)
    save_table(
        pd.DataFrame({"order_index": range(len(cluster_order)), "wap": cluster_order}),
        "wap_correlation_clustered_order.csv",
    )
    p.plot_correlation_heatmap(
        corr_full,
        path=save_figure_path("wap_correlation_heatmap_clustered.png"),
        title=f"WAP × WAP Pearson r — clustered (n={len(corr_full)})",
        cluster_order=cluster_order,
    )
    """),
    md("""
    ## 12. Top-K WAP correlation heatmap (readable subset)

    Same matrix, restricted to the 50 most class-discriminative WAPs (top
    ANOVA F-scores). Small enough that individual labels are readable.
    """),
    md("## 13. Feature relevance to target — ANOVA F-score"),
    code("""
    anova_df = eda.compute_anova_feature_scores(split.X_train, split.y_train, top_k=30)
    save_table(anova_df, "top_wap_features_anova.csv")
    p.plot_top_features(
        anova_df,
        save_figure_path("top_wap_features_anova.png"),
        title="Top 30 WAPs by ANOVA F-score (target=building_floor)",
    )
    anova_df
    """),
    code("""
    # Top 50 WAPs (used for the next three heatmaps)
    anova_top = eda.compute_anova_feature_scores(split.X_train, split.y_train, top_k=50)
    top_wap = anova_top["feature"].tolist()
    corr_top = corr_full.loc[top_wap, top_wap]
    p.plot_correlation_heatmap(
        corr_top,
        path=save_figure_path("wap_correlation_heatmap_top50.png"),
        title="Top 50 WAPs (by ANOVA F) — pairwise Pearson r",
        cluster_order=eda.clustered_corr_order(corr_top),
    )
    """),
    md("""
    ## 14. Class-mean RSSI fingerprint heatmap

    Rows = the 13 `building_floor` classes, columns = the top-50 WAPs,
    cells = mean RSSI for that class.

    > Mean is taken over **all** scans (including not-detected, coded as
    > `-110`), so a near-`-110` cell means *"this WAP is rarely heard by
    > this class"* — which is what the model actually sees.
    """),
    code("""
    centroids = eda.compute_class_centroids(split.X_train, split.y_train)
    centroids_top = centroids[top_wap]
    save_table(centroids_top.reset_index(), "class_centroids_top50.csv")
    p.plot_class_fingerprint_heatmap(
        centroids_top,
        path=save_figure_path("class_fingerprint_heatmap_top50.png"),
        title="Class-mean RSSI fingerprint — top 50 WAPs (rows = building_floor)",
    )
    """),
    md("""
    ## 15. Per-class WAP detection-rate heatmap

    Same axes, but cells are detection rate ∈ [0, 1] — fraction of scans
    where the WAP was detected at all. Highlights APs that are
    discriminative in *availability*, not just signal strength.
    """),
    code("""
    rates_full = eda.compute_wap_detection_rate_per_class(train)
    rates_top = rates_full[top_wap]
    save_table(rates_top.reset_index(), "per_class_wap_detection_rate_top50.csv")
    p.plot_detection_rate_heatmap(
        rates_top,
        path=save_figure_path("detection_rate_heatmap_top50.png"),
        title="Per-class WAP detection rate — top 50 WAPs",
    )
    """),
    md("""
    ## 16. Class centroid distance heatmap (predicted confusion)

    Pairwise Euclidean distance between class centroids in WAP space.
    **Small distance ≈ confusable classes** — this matrix predicts the
    structure of the confusion matrices we'll see in `nb03`.
    Same-building / adjacent-floor pairs typically sit closest together.
    """),
    code("""
    distances = eda.compute_class_centroid_distances(centroids)
    save_table(distances.reset_index(), "class_centroid_distances.csv")
    p.plot_class_centroid_distance_heatmap(
        distances,
        path=save_figure_path("class_centroid_distance_heatmap.png"),
    )
    """),
    md("""
    ## 17. Per-WAP train ↔ validation drift

    One point per WAP. Off-diagonal points are exactly the WAPs whose
    behaviour differs between training and validation — they are the most
    likely culprits behind any validation accuracy drop.
    """),
    code("""
    shift = eda.compute_per_wap_train_valid_shift(train, valid)
    save_table(shift, "per_wap_train_valid_shift.csv")
    print("WAPs with no detection in validation:",
          int(shift["valid_mean_rssi"].isna().sum()))
    print("WAPs with no detection in training  :",
          int(shift["train_mean_rssi"].isna().sum()))
    p.plot_wap_shift_scatter(
        shift, save_figure_path("wap_train_valid_shift_scatter.png"),
    )
    """),
    md("## 18. Sparsity at a glance"),
    code("""
    p.plot_sparsity_heatmap(
        train, save_figure_path("sparsity_heatmap_subsample.png"),
        n_rows=200,
        title="Detection mask — 200-row subsample × all 520 WAPs (black = detected)",
    )
    """),
    md("""
    ## 19. EDA conclusion

    * Dataset is **high-dimensional** (520 features) and **sparse** — ~96–97 %
      of WAP cells are the *not-detected* sentinel (§2, §3, §18).
    * `100` must be treated as *not detected*, never as a real signal.
    * Classes are **imbalanced** — building 2 and floor 0/1 dominate (§5).
    * Within a building, **floors are not spatially separated** (§8) — WiFi is
      the only signal that distinguishes them.
    * Validation is a **time / device / user shift, not a spatial one** (§9),
      so the official split is kept as the held-out test set.
    * Many WAPs are **strongly correlated** with visible block structure (§10–12)
      → use regularisation / PCA / LDA.
    * Per-class WAP **fingerprints are visibly distinct** (§14–15), so the
      classification task is genuinely learnable from RSSI alone.
    * Centroid-distance heatmap (§16) **predicts** the confusion structure we
      will see in `nb03` — same-building / adjacent-floor pairs sit closest.
    * Train↔valid drift (§17) names the specific WAPs likely to hurt
      generalisation — input for any future feature-selection scenario.

    These properties make the dataset a clean fit for **Scenario 2**
    (parametric vs non-parametric) and **Scenario 5** (DR before
    classification).
    """),
]

write_nb("01_data_audit_and_eda.ipynb", nb01)



# ===========================================================================
# 02 — Preprocessing and Targets
# ===========================================================================

nb02 = [
    md("""
    # 02 — Preprocessing and Targets

    All preprocessing decisions in one place, behind reusable functions in
    `ujiindoorloc.preprocessing`. The same code paths are used by every
    later notebook.
    """),
    code(BOOTSTRAP),
    code("""
    import numpy as np
    import pandas as pd
    import joblib

    from ujiindoorloc.constants import (
        COMBINED_TARGET, LEAKAGE_COLUMNS, MISSING_SIGNAL_VALUE,
        REPLACEMENT_SIGNAL_VALUE, MODELS_DIR,
    )
    from ujiindoorloc.data_loading import load_raw_data, split_features_targets, get_wap_columns
    from ujiindoorloc.preprocessing import prepare_classification_data
    from ujiindoorloc.reporting import ensure_report_dirs, save_table

    ensure_report_dirs()
    raw = load_raw_data()
    print("MISSING_SIGNAL_VALUE :", MISSING_SIGNAL_VALUE)
    print("REPLACEMENT_SIGNAL   :", REPLACEMENT_SIGNAL_VALUE)
    """),
    md("""
    ## 1. Target construction

    `building_floor` is built from `BUILDINGID` and `FLOOR`. We align the
    categorical class set across train+valid so confusion-matrix axes are
    stable.
    """),
    code("""
    split = split_features_targets(raw.train, raw.valid, target_name=COMBINED_TARGET)
    print("X_train:", split.X_train.shape)
    print("X_valid:", split.X_valid.shape)
    print("classes:", list(split.y_train.cat.categories))
    """),
    md("""
    ## 2. Preprocessing pipelines

    Two variants, both fit **on training only**:

    * **scaled** — replace 100→-110, drop train-constant cols, StandardScaler.
      Used by LR / LDA / QDA / kNN / PCA.
    * **tree** — replace 100→-110, drop train-constant cols, *no scaling*.
      Used by Decision Tree / Random Forest.
    """),
    code("""
    prepared = prepare_classification_data(split.X_train, split.X_valid)
    print("X_train_scaled :", prepared.X_train_scaled.shape)
    print("X_valid_scaled :", prepared.X_valid_scaled.shape)
    print("X_train_tree   :", prepared.X_train_tree.shape)
    print("X_valid_tree   :", prepared.X_valid_tree.shape)
    print("kept WAPs (after train-constant filter):", len(prepared.kept_wap_columns))
    """),
    md("## 3. Save the manifest of kept WAP columns"),
    code("""
    kept_df = pd.DataFrame({"wap": prepared.kept_wap_columns})
    save_table(kept_df, "selected_wap_columns_after_constant_filter.csv")
    kept_df.head()
    """),
    md("## 4. Quality checks"),
    code("""
    assert prepared.X_train_scaled.shape[0] == len(split.y_train)
    assert prepared.X_valid_scaled.shape[0] == len(split.y_valid)
    assert prepared.X_train_scaled.shape[1] == prepared.X_valid_scaled.shape[1]
    assert prepared.X_train_tree.shape[1]   == prepared.X_valid_tree.shape[1]
    assert not np.isnan(prepared.X_train_scaled).any()
    assert not np.isnan(prepared.X_valid_scaled).any()

    # Leakage guard — no metadata/target column may be among the predictors.
    bad = set(split.X_train.columns) & set(LEAKAGE_COLUMNS)
    assert not bad, f"leakage columns in features: {bad}"

    # Same WAP order in train and validation (we already filter to wap_cols upstream).
    assert list(split.X_train.columns) == list(split.X_valid.columns)

    print("All preprocessing quality checks passed.")
    """),
    md("""
    ## 5. Persist fitted pipelines

    Saved so any later notebook can re-use the *exact same* transformation
    without re-fitting.
    """),
    code("""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared.scaled_pipeline, MODELS_DIR / "preprocessor_scaled.joblib")
    joblib.dump(prepared.tree_pipeline,   MODELS_DIR / "preprocessor_tree.joblib")
    print("Saved preprocessors to:", MODELS_DIR)
    """),
    md("""
    ## 6. Why fit on training only?

    Fitting any preprocessing step on validation data leaks information from
    the future *test set* back into model selection. Concretely:

    * **Constant-column filter** — a column that is constant in training but
      varies in validation gives the model no signal during training; we keep
      it filtered out and let the model live with that.
    * **StandardScaler** — mean/std must come from training only, otherwise
      the validation distribution co-determines the model input.
    * **PCA / LDA** (Scenario 5) — same principle: `fit_transform(train)`,
      then `transform(valid)`.
    """),
]

write_nb("02_preprocessing_and_targets.ipynb", nb02)


# ===========================================================================
# 03 — Scenario 2: Model Comparison
# ===========================================================================

nb03 = [
    md("""
    # 03 — Scenario 2: Parametric vs Non-Parametric Models

    Goal: compare parametric and non-parametric classifiers on the same
    `building_floor` target, using the official validation split as the held-out
    test set. The comparison uses identical preprocessing and writes all model
    diagnostics to fixed report paths for final interpretation.
    """),
    code(BOOTSTRAP),
    code("""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ujiindoorloc.constants import COMBINED_TARGET, FIGURES_DIR, MODELS_DIR, TABLES_DIR
    from ujiindoorloc.data_loading import load_raw_data, split_features_targets
    from ujiindoorloc.preprocessing import prepare_classification_data
    from ujiindoorloc import modeling as M
    from ujiindoorloc import evaluation as E
    from ujiindoorloc import plots as P
    from ujiindoorloc.reporting import ensure_report_dirs, save_table

    ensure_report_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    """),
    code("""
    raw = load_raw_data()
    split = split_features_targets(raw.train, raw.valid, target_name=COMBINED_TARGET)
    prepared = prepare_classification_data(split.X_train, split.X_valid)

    Xs_tr, Xs_va = prepared.X_train_scaled, prepared.X_valid_scaled
    Xt_tr, Xt_va = prepared.X_train_tree,   prepared.X_valid_tree
    y_tr, y_va = np.asarray(split.y_train), np.asarray(split.y_valid)
    print("scaled :", Xs_tr.shape, Xs_va.shape)
    print("tree   :", Xt_tr.shape, Xt_va.shape)
    """),
    md("""
    ## Model groups definition

    Parametric models assume a fixed functional or distributional form
    (Logistic Regression, LDA, QDA). Non-parametric models adapt more directly
    to the training samples (kNN, Decision Tree, Random Forest). Each spec keeps
    its family label so the final selection can compare best-in-family models.
    """),
    code("""
    specs = [
        {
            "name": "logistic_regression",
            "family": "parametric",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", M.build_logistic_regression()),
            ]),
            "X_train": Xs_tr,
            "X_valid": Xs_va,
        },
        {
            "name": "lda",
            "family": "parametric",
            "model": M.build_lda(),
            "X_train": Xs_tr,
            "X_valid": Xs_va,
        },
        {
            "name": "qda_pca50",
            "family": "parametric",
            "model": M.build_pca_qda_pipeline(50),
            "X_train": Xs_tr,
            "X_valid": Xs_va,
        },
        {
            "name": "knn",
            "family": "non_parametric",
            "model": M.build_knn(),
            "X_train": Xs_tr,
            "X_valid": Xs_va,
        },
        {
            "name": "decision_tree",
            "family": "non_parametric",
            "model": Pipeline([("classifier", M.build_decision_tree())]),
            "X_train": Xt_tr,
            "X_valid": Xt_va,
        },
        {
            "name": "random_forest",
            "family": "non_parametric",
            "model": Pipeline([("classifier", M.build_random_forest())]),
            "X_train": Xt_tr,
            "X_valid": Xt_va,
        },
    ]
    family_by_model = {spec["name"]: spec["family"] for spec in specs}
    classification_report_paths = {
        "logistic_regression": "scenario_2_classification_report_logistic_regression.csv",
        "lda": "scenario_2_classification_report_lda.csv",
        "qda_pca50": "scenario_2_classification_report_qda_pca50.csv",
        "knn": "scenario_2_classification_report_knn.csv",
        "decision_tree": "scenario_2_classification_report_decision_tree.csv",
        "random_forest": "scenario_2_classification_report_random_forest.csv",
    }
    confusion_matrix_paths = {
        "logistic_regression": "scenario_2_confusion_matrix_logistic_regression.png",
        "lda": "scenario_2_confusion_matrix_lda.png",
        "qda_pca50": "scenario_2_confusion_matrix_qda_pca50.png",
        "knn": "scenario_2_confusion_matrix_knn.png",
        "decision_tree": "scenario_2_confusion_matrix_decision_tree.png",
        "random_forest": "scenario_2_confusion_matrix_random_forest.png",
    }
    pd.DataFrame([{k: v for k, v in spec.items() if k != "model"} for spec in specs])
    """),
    md("""
    ## Train and evaluate

    Every successful model writes a classification report and confusion matrix
    immediately. Failures are logged and skipped so one unstable classifier does
    not block the full scenario.
    """),
    code("""
    results = []
    fitted_models = {}

    for spec in specs:
        name = spec["name"]
        model = spec["model"]
        try:
            result = E.evaluate_classifier(
                model,
                spec["X_train"],
                y_tr,
                spec["X_valid"],
                y_va,
                name=name,
            )
            report_df = E.classification_report_to_dataframe(
                y_va, result.y_pred, labels=result.classes,
            )
            save_table(report_df, classification_report_paths[name])
            P.plot_confusion_matrix(
                result,
                FIGURES_DIR / confusion_matrix_paths[name],
                title=f"Scenario 2 — confusion matrix: {name}",
            )
            results.append(result)
            fitted_models[name] = model
            print(
                f"{name:25s} family={spec['family']:14s} "
                f"bal_acc={result.metrics['balanced_accuracy']:.3f} "
                f"macro_f1={result.metrics['macro_f1']:.3f} "
                f"fit={result.fit_seconds:.1f}s"
            )
        except Exception as exc:
            print(f"{name:25s}  FAILED: {type(exc).__name__}: {exc}")
    """),
    md("""
    ## Raw QDA attempt

    Raw QDA is theoretically parametric but numerically fragile in this
    high-dimensional correlated WAP space. We attempt it once, save diagnostics
    only if it succeeds, and otherwise keep the failure as an expected result.
    """),
    code("""
    try:
        qda_raw = M.build_qda(reg_param=1e-3)
        result = E.evaluate_classifier(
            qda_raw, Xs_tr, y_tr, Xs_va, y_va, name="qda_raw",
        )
        report_df = E.classification_report_to_dataframe(
            y_va, result.y_pred, labels=result.classes,
        )
        save_table(report_df, "scenario_2_classification_report_qda_raw.csv")
        P.plot_confusion_matrix(
            result,
            FIGURES_DIR / "scenario_2_confusion_matrix_qda_raw.png",
            title="Scenario 2 — confusion matrix: qda_raw",
        )
        results.append(result)
        fitted_models["qda_raw"] = qda_raw
        family_by_model["qda_raw"] = "parametric"
        print(
            f"qda_raw                  family=parametric     "
            f"bal_acc={result.metrics['balanced_accuracy']:.3f} "
            f"macro_f1={result.metrics['macro_f1']:.3f}"
        )
    except Exception as exc:
        print(f"qda_raw FAILED: {type(exc).__name__}: {exc}")
    """),
    md("## Metrics table"),
    code("""
    metrics_df = E.results_to_metrics_df(results)
    metrics_df["family"] = metrics_df["model"].map(family_by_model)
    ordered_cols = ["model", "family"] + [c for c in metrics_df.columns if c not in {"model", "family"}]
    metrics_df = metrics_df[ordered_cols]
    save_table(metrics_df, "scenario_2_model_metrics.csv")
    display(metrics_df)
    """),
    md("## Metric comparison plot"),
    code("""
    P.plot_metric_comparison(
        metrics_df,
        FIGURES_DIR / "scenario_2_metric_comparison.png",
        title="Scenario 2 — model comparison",
    )
    """),
    md("## Select best parametric and non-parametric models"),
    code("""
    parametric_metrics = metrics_df[metrics_df["family"] == "parametric"]
    non_parametric_metrics = metrics_df[metrics_df["family"] == "non_parametric"]

    best_parametric = E.select_best_model(parametric_metrics)
    best_non_parametric = E.select_best_model(non_parametric_metrics)

    parametric_path = MODELS_DIR / "scenario_2_best_parametric.joblib"
    non_parametric_path = MODELS_DIR / "scenario_2_best_non_parametric.joblib"
    joblib.dump(fitted_models[best_parametric["model"]], parametric_path)
    joblib.dump(fitted_models[best_non_parametric["model"]], non_parametric_path)

    print(
        "Best parametric:", best_parametric["model"],
        f"(balanced_accuracy={best_parametric['balanced_accuracy']:.3f}, "
        f"macro_f1={best_parametric['macro_f1']:.3f}) -> {parametric_path}"
    )
    print(
        "Best non-parametric:", best_non_parametric["model"],
        f"(balanced_accuracy={best_non_parametric['balanced_accuracy']:.3f}, "
        f"macro_f1={best_non_parametric['macro_f1']:.3f}) -> {non_parametric_path}"
    )
    """),
    md("## Logistic Regression interpretability"),
    code("""
    if "logistic_regression" in fitted_models:
        fitted_model = fitted_models["logistic_regression"]
        coef = fitted_model.named_steps["classifier"].coef_
        lr_coef_df = (
            pd.DataFrame({
                "wap": prepared.kept_wap_columns,
                "mean_abs_coef": np.mean(np.abs(coef), axis=0),
            })
            .sort_values("mean_abs_coef", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        save_table(lr_coef_df, "scenario_2_logistic_regression_coefficients.csv")
        P.plot_top_features(
            lr_coef_df.rename(columns={"wap": "feature", "mean_abs_coef": "score"}),
            FIGURES_DIR / "scenario_2_logistic_regression_top_coefficients.png",
            title="Scenario 2 — Logistic Regression top coefficients",
            feature_col="feature",
            score_col="score",
        )
        display(lr_coef_df)
    else:
        print("Logistic Regression did not succeed; skipping coefficient export.")
    """),
    md("## Random Forest interpretability"),
    code("""
    if "random_forest" in fitted_models:
        fitted_model = fitted_models["random_forest"]
        importances = fitted_model.named_steps["classifier"].feature_importances_
        rf_importance_df = (
            pd.DataFrame({
                "wap": prepared.kept_wap_columns,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        save_table(rf_importance_df, "scenario_2_random_forest_feature_importances.csv")
        P.plot_top_features(
            rf_importance_df.rename(columns={"wap": "feature", "importance": "score"}),
            FIGURES_DIR / "scenario_2_random_forest_feature_importances.png",
            title="Scenario 2 — Random Forest feature importances",
            feature_col="feature",
            score_col="score",
        )
        display(rf_importance_df)
    else:
        print("Random Forest did not succeed; skipping feature-importance export.")
    """),
    md("""
    ## Interpretation and conclusion

    This notebook produces one metric table, one classification report per
    successful model, one confusion matrix per successful model, family-specific
    best-model artifacts, and two interpretability exports. The final notebook
    uses these fixed outputs to decide whether the extra flexibility of
    non-parametric models is worth the explainability/runtime trade-off.
    """),
]

write_nb("03_scenario_2_model_comparison.ipynb", nb03)


# ===========================================================================
# 04 — Scenario 5: Dimensionality Reduction
# ===========================================================================

nb04 = [
    md("""
    # 04 — Scenario 5: Dimensionality Reduction Before Classification

    Goal: test whether PCA or LDA projections can compress the WAP feature
    space while preserving building-floor classification quality. PCA is
    unsupervised; LDA projection is supervised and limited to `n_classes - 1`
    discriminant axes.
    """),
    code(BOOTSTRAP),
    code("""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ujiindoorloc.constants import COMBINED_TARGET, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TABLES_DIR
    from ujiindoorloc.data_loading import load_raw_data, split_features_targets
    from ujiindoorloc.preprocessing import prepare_classification_data
    from ujiindoorloc import modeling as M
    from ujiindoorloc import evaluation as E
    from ujiindoorloc import plots as P
    from ujiindoorloc.reporting import ensure_report_dirs, save_table

    ensure_report_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    """),
    code("""
    raw = load_raw_data()
    split = split_features_targets(raw.train, raw.valid, target_name=COMBINED_TARGET)
    prepared = prepare_classification_data(split.X_train, split.X_valid)

    Xs_tr, Xs_va = prepared.X_train_scaled, prepared.X_valid_scaled
    Xt_tr, Xt_va = prepared.X_train_tree, prepared.X_valid_tree
    y_tr, y_va = np.asarray(split.y_train), np.asarray(split.y_valid)
    print("scaled X:", Xs_tr.shape)
    print("tree X  :", Xt_tr.shape)
    """),
    md("## PCA explained variance"),
    code("""
    pca_full = PCA(random_state=RANDOM_STATE).fit(Xs_tr)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    var_df = pd.DataFrame({
        "component": np.arange(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum,
    })
    save_table(var_df, "scenario_5_pca_explained_variance.csv")
    P.plot_pca_explained_variance(
        evr,
        FIGURES_DIR / "scenario_5_pca_explained_variance.png",
    )

    for k in [2, 5, 10, 20, 30, 50, 75, 100]:
        if k <= len(cum):
            print(f"  k={k:>3d}: cumulative variance = {cum[k-1]*100:5.1f}%")
    display(var_df.head(10))
    """),
    md("## PCA 2D scatter"),
    code("""
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs_tr)
    pcs2 = pca2.transform(Xs_tr)
    P.plot_pca_2d_scatter(
        pcs2,
        y_tr,
        FIGURES_DIR / "scenario_5_pca_2d_scatter.png",
        title="PCA — first 2 components, coloured by building_floor",
    )
    """),
    md("## PCA + classifier sweep"),
    code("""
    component_grid = [2, 5, 10, 20, 30, 50, 75, 100]
    pca_rows = []
    for k in component_grid:
        for classifier_name, build in [
            ("logistic_regression", lambda k=k: M.build_pca_logistic_pipeline(k)),
            ("knn",                 lambda k=k: M.build_pca_knn_pipeline(k)),
        ]:
            result = E.evaluate_classifier(
                build(),
                Xs_tr,
                y_tr,
                Xs_va,
                y_va,
                name=f"pca_{classifier_name}_{k}",
            )
            pca_rows.append(
                {
                    "reduction_method": "PCA",
                    "classifier_name": classifier_name,
                    "n_components": k,
                    "accuracy": result.metrics["accuracy"],
                    "balanced_accuracy": result.metrics["balanced_accuracy"],
                    "macro_f1": result.metrics["macro_f1"],
                    "weighted_f1": result.metrics["weighted_f1"],
                }
            )
            print(
                f"PCA({k:>3d}) + {classifier_name:20s} "
                f"bal_acc={result.metrics['balanced_accuracy']:.3f} "
                f"macro_f1={result.metrics['macro_f1']:.3f}"
            )

    pca_metrics_df = pd.DataFrame(pca_rows)
    save_table(pca_metrics_df, "scenario_5_pca_metrics.csv")
    P.plot_metric_vs_components(
        pca_metrics_df,
        FIGURES_DIR / "scenario_5_pca_metric_vs_components.png",
        title="Scenario 5 — PCA + classifier balanced accuracy vs components",
        metric="balanced_accuracy",
        model_col="classifier_name",
    )
    display(pca_metrics_df)
    """),
    md("## LDA projection before classification"),
    code("""
    n_classes = len(np.unique(y_tr))
    max_lda = min(n_classes - 1, Xs_tr.shape[1])
    lda_components = [c for c in [2, 5, 10, 12] if c <= max_lda]
    print("n_classes:", n_classes, "  max LDA components:", max_lda)

    lda_rows = []
    for n in lda_components:
        for classifier_name, build in [
            ("logistic_regression", lambda n=n: M.build_lda_projection_logistic_pipeline(n_components=n)),
            ("knn",                 lambda n=n: M.build_lda_projection_knn_pipeline(n_components=n)),
        ]:
            result = E.evaluate_classifier(
                build(),
                Xs_tr,
                y_tr,
                Xs_va,
                y_va,
                name=f"lda_projection_{classifier_name}_{n}",
            )
            lda_rows.append(
                {
                    "reduction_method": "LDA_projection",
                    "classifier_name": classifier_name,
                    "n_components": n,
                    "accuracy": result.metrics["accuracy"],
                    "balanced_accuracy": result.metrics["balanced_accuracy"],
                    "macro_f1": result.metrics["macro_f1"],
                    "weighted_f1": result.metrics["weighted_f1"],
                }
            )
            print(
                f"LDA({n:>2d}) + {classifier_name:20s} "
                f"bal_acc={result.metrics['balanced_accuracy']:.3f} "
                f"macro_f1={result.metrics['macro_f1']:.3f}"
            )

    lda_metrics_df = pd.DataFrame(lda_rows)
    save_table(lda_metrics_df, "scenario_5_lda_projection_metrics.csv")
    P.plot_metric_vs_components(
        lda_metrics_df,
        FIGURES_DIR / "scenario_5_lda_projection_metric_vs_components.png",
        title="Scenario 5 — LDA projection + classifier balanced accuracy vs components",
        metric="balanced_accuracy",
        model_col="classifier_name",
    )
    display(lda_metrics_df)
    """),
    md("## LDA 2D visualization"),
    code("""
    if max_lda >= 2:
        lda2 = LinearDiscriminantAnalysis(n_components=2, solver="svd")
        lda2_projection = lda2.fit_transform(Xs_tr, y_tr)
        P.plot_2d_projection_scatter(
            lda2_projection,
            y_tr,
            FIGURES_DIR / "scenario_5_lda_2d_scatter.png",
            title="LDA projection — first 2 discriminants",
            x_label="LD1",
            y_label="LD2",
        )
    else:
        print("LDA 2D visualization skipped: fewer than 2 discriminants available.")
    """),
    md("## Select best dimensionality reduction model"),
    code("""
    pca_metrics = pd.read_csv(TABLES_DIR / "scenario_5_pca_metrics.csv")
    lda_metrics = pd.read_csv(TABLES_DIR / "scenario_5_lda_projection_metrics.csv")
    all_dr_metrics = pd.concat([pca_metrics, lda_metrics], ignore_index=True)

    best_balanced_accuracy = all_dr_metrics["balanced_accuracy"].max()
    near_best = all_dr_metrics[
        all_dr_metrics["balanced_accuracy"] >= best_balanced_accuracy - 0.01
    ]
    best_row = near_best.sort_values(
        ["n_components", "balanced_accuracy", "macro_f1"],
        ascending=[True, False, False],
    ).iloc[0]

    if best_row["reduction_method"] == "PCA":
        best_pipeline = M.build_pca_logistic_pipeline(n_components=int(best_row["n_components"])) \
            if best_row["classifier_name"] == "logistic_regression" \
            else M.build_pca_knn_pipeline(n_components=int(best_row["n_components"]))
    elif best_row["reduction_method"] == "LDA_projection":
        best_pipeline = M.build_lda_projection_logistic_pipeline(n_components=int(best_row["n_components"])) \
            if best_row["classifier_name"] == "logistic_regression" \
            else M.build_lda_projection_knn_pipeline(n_components=int(best_row["n_components"]))
    else:
        raise ValueError(f"Unknown reduction method: {best_row['reduction_method']}")

    best_pipeline.fit(Xs_tr, y_tr)
    best_path = MODELS_DIR / "scenario_5_best_dimensionality_reduction_model.joblib"
    joblib.dump(best_pipeline, best_path)

    print(
        "Selected",
        f"{best_row['reduction_method']} + {best_row['classifier_name']}",
        f"with {int(best_row['n_components'])} components."
    )
    print(
        f"balanced_accuracy={best_row['balanced_accuracy']:.3f}, "
        f"macro_f1={best_row['macro_f1']:.3f}. "
        "Selection uses candidates within 0.01 balanced_accuracy of the best, "
        "then prefers fewer components."
    )
    print("Saved best dimensionality-reduction pipeline to:", best_path)
    """),
    md("""
    ## Scenario 5 conclusion

    PCA answers how much unsupervised compression is possible; LDA answers how
    compact a supervised class-separating representation can be. The selected
    pipeline is saved for reuse, while the final notebook compares this reduced
    model against the best original-feature models from Scenario 2.
    """),
]

write_nb("04_scenario_5_dimensionality_reduction.ipynb", nb04)


# ===========================================================================
# 05 — Final Comparison and Interpretation
# ===========================================================================

nb05 = [
    md("""
    # 05 — Final Comparison and Recommendation

    Pull together Scenario 2 and Scenario 5 outputs, compare the best model in
    each family, and print the final deployment recommendation. This notebook
    reads saved CSVs only; it does not retrain models.
    """),
    code(BOOTSTRAP),
    code("""
    import pandas as pd
    from ujiindoorloc.constants import FIGURES_DIR, MODELS_DIR, TABLES_DIR
    """),
    md("## Load Scenario 2 results"),
    code("""
    s2 = pd.read_csv(TABLES_DIR / "scenario_2_model_metrics.csv")
    s2_sorted = s2.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    display(s2_sorted)
    """),
    md("## Load Scenario 5 results"),
    code("""
    s5_pca = pd.read_csv(TABLES_DIR / "scenario_5_pca_metrics.csv")
    s5_lda = pd.read_csv(TABLES_DIR / "scenario_5_lda_projection_metrics.csv")

    display(s5_pca.sort_values(["classifier_name", "n_components"]).reset_index(drop=True))
    display(s5_lda.sort_values(["classifier_name", "n_components"]).reset_index(drop=True))
    """),
    md("## Best model per family"),
    code("""
    def best_by_balanced_accuracy(df: pd.DataFrame) -> pd.Series:
        return df.sort_values(
            ["balanced_accuracy", "macro_f1"], ascending=[False, False]
        ).iloc[0]


    best_parametric_original = best_by_balanced_accuracy(s2[s2["family"] == "parametric"])
    best_non_parametric_original = best_by_balanced_accuracy(s2[s2["family"] == "non_parametric"])
    best_pca_reduced = best_by_balanced_accuracy(s5_pca)
    best_lda_reduced = best_by_balanced_accuracy(s5_lda)
    best_s2 = best_by_balanced_accuracy(s2)
    best_s5 = best_by_balanced_accuracy(pd.concat([s5_pca, s5_lda], ignore_index=True))

    lr_original = s2[s2["model"] == "logistic_regression"].iloc[0]
    rf_rows = s2[s2["model"] == "random_forest"]
    rf_original = rf_rows.iloc[0] if not rf_rows.empty else None

    if lr_original["balanced_accuracy"] >= best_s2["balanced_accuracy"] - 0.03:
        final_name = "logistic_regression"
        final_source = "Scenario 2 original WAP features"
        final_reason = "within 0.03 balanced_accuracy of the best Scenario 2 model and most explainable"
        final_metrics = lr_original
    elif rf_original is not None and rf_original["balanced_accuracy"] > lr_original["balanced_accuracy"] + 0.03:
        final_name = "random_forest"
        final_source = "Scenario 2 original WAP features"
        final_reason = "Random Forest is more than 0.03 balanced_accuracy better than Logistic Regression"
        final_metrics = rf_original
    else:
        final_name = best_s2["model"]
        final_source = "Scenario 2 original WAP features"
        final_reason = "best overall Scenario 2 balanced_accuracy after tie-breaking"
        final_metrics = best_s2

    family_rows = [
        {
            "family": "best_parametric_original",
            "source": "Scenario 2",
            "model": best_parametric_original["model"],
            "reduction_method": "none",
            "n_components": pd.NA,
            "balanced_accuracy": best_parametric_original["balanced_accuracy"],
            "macro_f1": best_parametric_original["macro_f1"],
        },
        {
            "family": "best_non_parametric_original",
            "source": "Scenario 2",
            "model": best_non_parametric_original["model"],
            "reduction_method": "none",
            "n_components": pd.NA,
            "balanced_accuracy": best_non_parametric_original["balanced_accuracy"],
            "macro_f1": best_non_parametric_original["macro_f1"],
        },
        {
            "family": "best_pca_reduced",
            "source": "Scenario 5",
            "model": best_pca_reduced["classifier_name"],
            "reduction_method": "PCA",
            "n_components": int(best_pca_reduced["n_components"]),
            "balanced_accuracy": best_pca_reduced["balanced_accuracy"],
            "macro_f1": best_pca_reduced["macro_f1"],
        },
        {
            "family": "best_lda_reduced",
            "source": "Scenario 5",
            "model": best_lda_reduced["classifier_name"],
            "reduction_method": "LDA_projection",
            "n_components": int(best_lda_reduced["n_components"]),
            "balanced_accuracy": best_lda_reduced["balanced_accuracy"],
            "macro_f1": best_lda_reduced["macro_f1"],
        },
        {
            "family": "final_recommended_model",
            "source": final_source,
            "model": final_name,
            "reduction_method": "none",
            "n_components": pd.NA,
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "reason": final_reason,
        },
    ]

    best_family_df = pd.DataFrame(family_rows)
    best_family_df.to_csv(TABLES_DIR / "final_best_model_per_family.csv", index=False)
    display(best_family_df)
    """),
    md("## Final compact comparison"),
    code("""
    n_waps = "~465 WAPs"
    compact_rows = [
        {
            "approach": "Original WAP features",
            "input_features": n_waps,
            "model": best_parametric_original["model"],
            "n_features_or_components": n_waps,
            "balanced_accuracy": best_parametric_original["balanced_accuracy"],
            "macro_f1": best_parametric_original["macro_f1"],
            "explainability": "high",
            "recommendation": "safe baseline",
        },
        {
            "approach": "Original WAP features",
            "input_features": n_waps,
            "model": best_non_parametric_original["model"],
            "n_features_or_components": n_waps,
            "balanced_accuracy": best_non_parametric_original["balanced_accuracy"],
            "macro_f1": best_non_parametric_original["macro_f1"],
            "explainability": "medium",
            "recommendation": "best performance candidate",
        },
        {
            "approach": "PCA components",
            "input_features": "compressed PCs",
            "model": best_pca_reduced["classifier_name"],
            "n_features_or_components": f"{int(best_pca_reduced['n_components'])} PCs",
            "balanced_accuracy": best_pca_reduced["balanced_accuracy"],
            "macro_f1": best_pca_reduced["macro_f1"],
            "explainability": "lower",
            "recommendation": "good compression",
        },
        {
            "approach": "LDA projection",
            "input_features": "supervised LDs",
            "model": best_lda_reduced["classifier_name"],
            "n_features_or_components": f"{int(best_lda_reduced['n_components'])} LDs",
            "balanced_accuracy": best_lda_reduced["balanced_accuracy"],
            "macro_f1": best_lda_reduced["macro_f1"],
            "explainability": "medium",
            "recommendation": "supervised DR",
        },
    ]

    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(TABLES_DIR / "final_compact_model_comparison.csv", index=False)
    display(compact_df)
    """),
    md("## Final recommendation logic"),
    code("""
    notes = []
    if lr_original["balanced_accuracy"] >= best_s2["balanced_accuracy"] - 0.03:
        notes.append(
            "Logistic Regression is within 0.03 balanced_accuracy of the best "
            "original-feature model, so it is the safest explainable choice."
        )

    if rf_original is not None and rf_original["balanced_accuracy"] > lr_original["balanced_accuracy"] + 0.03:
        notes.append(
            "Random Forest is more than 0.03 balanced_accuracy better than "
            "Logistic Regression, so use it as the practical best model and "
            "keep Logistic Regression as the explainable baseline."
        )

    if best_pca_reduced["balanced_accuracy"] >= best_s2["balanced_accuracy"] - 0.02:
        notes.append(
            f"PCA with {int(best_pca_reduced['n_components'])} components is "
            "within 0.02 balanced_accuracy of the original-feature best, so it "
            "is a valid compression strategy."
        )

    if best_lda_reduced["balanced_accuracy"] >= best_s2["balanced_accuracy"] - 0.03:
        notes.append(
            f"LDA projection with {int(best_lda_reduced['n_components'])} "
            "components performs close to the original-feature best, making it "
            "a strong supervised dimensionality-reduction option."
        )

    print("Final recommendation:")
    print(
        f"Recommend {final_name} from {final_source}. It reaches "
        f"balanced_accuracy={final_metrics['balanced_accuracy']:.3f} and "
        f"macro_f1={final_metrics['macro_f1']:.3f}; rationale: {final_reason}."
    )
    if notes:
        print(" ".join(notes))
    print(
        "Use balanced accuracy as the headline metric, macro F1 as the class-balance "
        "check, and the exported confusion matrices plus feature-importance/coefficient "
        "tables for the defence narrative."
    )
    """),
    md("""
    ## Limitations and future work

    Results are tied to the official validation split, which is useful because
    it captures a real time/device/user shift but still only represents one
    campus. Future work should add explicit feature selection, device-aware
    validation, coordinate regression, and a small Python Shiny app for model
    inspection once the classification story is accepted.
    """),
]

write_nb("05_final_comparison_and_interpretation.ipynb", nb05)

print("All 6 notebooks written to:", NB_DIR)
