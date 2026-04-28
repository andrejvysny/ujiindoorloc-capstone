# src/ujiindoorloc KNOWLEDGE BASE

**Generated:** 2025-04-28

## OVERVIEW

Reusable ML pipeline for UJIIndoorLoc WiFi fingerprint classification. 9 modules, no sub-packages.

## WHERE TO LOOK

| Task              | File               | Key Symbol                                                     |
| ----------------- | ------------------ | -------------------------------------------------------------- |
| Paths & constants | `constants.py`     | `REPO_ROOT`, `LEAKAGE_COLUMNS`, `MISSING_SIGNAL_VALUE`         |
| Load CSVs         | `data_loading.py`  | `load_raw_data()`, `split_features_targets()`                  |
| Preprocessing     | `preprocessing.py` | `prepare_classification_data()`, `MissingSignalReplacer`       |
| Model factories   | `modeling.py`      | `build_logistic_regression()`, `build_random_forest()`         |
| Evaluation        | `evaluation.py`    | `evaluate_classifier()`, `results_to_metrics_df()`             |
| Plotting          | `plots.py`         | `plot_confusion_matrix()`, `plot_metric_comparison()`          |
| EDA               | `eda.py`           | `compute_wap_correlations()`, `compute_anova_feature_scores()` |
| IO helpers        | `reporting.py`     | `save_table()`, `save_figure_path()`                           |

## CONVENTIONS

- All sklearn transformers inherit `BaseEstimator` + `TransformerMixin`
- All factory functions return **unfitted** estimators
- All plotters save PNG and return `Path`
- All EDA helpers return tidy `pd.DataFrame`
- `RANDOM_STATE = 42` hard-coded in all stochastic estimators

## ANTI-PATTERNS

- Do not fit transformers on validation data — `fit_transform(train)`, then `transform(valid)`
- Do not pass `n_jobs` to `lbfgs` LogisticRegression
- Do not use raw `QDA` without PCA — unstable on 400+ correlated features
- Do not edit notebooks directly — change `scripts/build_notebooks.py` and regenerate
