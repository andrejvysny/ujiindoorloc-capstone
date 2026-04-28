# PROJECT KNOWLEDGE BASE

**Generated:** 2025-04-28
**Project:** UJIIndoorLoc Indoor Localization Capstone (FIIT OZNAL)

## OVERVIEW

Python 3.11+ capstone comparing parametric vs non-parametric classifiers and dimensionality reduction (PCA/LDA) for indoor building/floor classification from WiFi RSSI fingerprints. Editable-install package under `src/ujiindoorloc/`, executed via 6 ordered Jupyter notebooks.

## STRUCTURE

```
.
├── data/                       # Raw CSVs (read-only). processed/ is empty.
├── models/                     # Saved preprocessor joblibs (not final models)
├── notebooks/                  # 00–05, executed in order (generated from script)
├── reports/
│   ├── figures/                # 31+ PNG outputs
│   └── tables/                 # 20+ CSV outputs + classification reports
├── scripts/
│   └── build_notebooks.py      # Notebook generator — edit here, not .ipynb files
├── src/ujiindoorloc/           # Core ML package (see subdirectory AGENTS.md)
└── Elements of AI - Full Course/  # Read-only reference materials — do not edit
```

## WHERE TO LOOK

| Task              | Location                            | Notes                                             |
| ----------------- | ----------------------------------- | ------------------------------------------------- |
| Load data         | `src/ujiindoorloc/data_loading.py`  | `load_raw_data()`, `split_features_targets()`     |
| Preprocess        | `src/ujiindoorloc/preprocessing.py` | Two pipelines: scaled vs tree                     |
| Train models      | `src/ujiindoorloc/modeling.py`      | Factory functions return unfitted estimators      |
| Evaluate          | `src/ujiindoorloc/evaluation.py`    | `evaluate_classifier()`, `evaluate_many_models()` |
| Plot              | `src/ujiindoorloc/plots.py`         | All fns save PNG + display inline in Jupyter      |
| EDA               | `src/ujiindoorloc/eda.py`           | Tidy DataFrame helpers                            |
| Notebooks         | `scripts/build_notebooks.py`        | **Edit script, regenerate notebooks**             |
| Dataset semantics | `SUMMARY.md`                        | Pre-vetted modelling decisions                    |

## CODE MAP

| Symbol                        | Type      | Location           | Role                        |
| ----------------------------- | --------- | ------------------ | --------------------------- |
| `load_raw_data`               | function  | `data_loading.py`  | Load train+valid CSVs       |
| `split_features_targets`      | function  | `data_loading.py`  | WAP features + target split |
| `prepare_classification_data` | function  | `preprocessing.py` | Fit both pipelines          |
| `build_random_forest`         | function  | `modeling.py`      | RF factory                  |
| `build_logistic_regression`   | function  | `modeling.py`      | LR factory                  |
| `evaluate_classifier`         | function  | `evaluation.py`    | Fit, predict, metrics       |
| `EvalResult`                  | dataclass | `evaluation.py`    | Metrics container           |
| `MissingSignalReplacer`       | class     | `preprocessing.py` | sklearn transformer         |
| `ConstantColumnDropper`       | class     | `preprocessing.py` | sklearn transformer         |

## CONVENTIONS

- `from __future__ import annotations` in every module
- `pathlib.Path` for all paths (no `os.path`)
- Dataclasses for data containers: `RawData`, `SplitData`, `PreparedData`, `EvalResult`
- sklearn `Pipeline` for preprocessing; fit on train only, `.transform` on valid
- Factory functions return **unfitted** estimators
- Plotting functions: take data + path, save PNG, return Path
- EDA functions: return tidy DataFrames
- `RANDOM_STATE = 42` everywhere

## ANTI-PATTERNS (THIS PROJECT)

- **Never edit `.ipynb` files directly** — modify `scripts/build_notebooks.py` and run `uv run python scripts/build_notebooks.py`
- **Never use `100` as a real RSSI value** — it is the "WAP not detected" sentinel; replace with `-110`
- **Never use leakage columns as predictors**: `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`
- **Never fit preprocessing on validation data** — constant filter, scaler, PCA, LDA all fit on train only
- **Do not re-shuffle train+validation** — validation is a time/device/user shift, not a random split
- **Do not copy public notebooks/solutions** — UJIIndoorLoc is a known benchmark
- **No `n_jobs` on `lbfgs` LogisticRegression** — removed due to sklearn>=1.8 FutureWarning

## UNIQUE STYLES

- **Notebook bootstrap snippet**: Every notebook injects `sys.path` to make `src/` importable regardless of cwd
- **Two preprocessing variants**: `scaled` (StandardScaler) for LR/LDA/kNN/PCA, `tree` (no scaling) for DT/RF
- **Notebook generation from Python**: All 6 notebooks are generated from `scripts/build_notebooks.py` string constants
- **Primary metrics**: balanced accuracy > macro F1 > confusion matrix > simplicity > explainability

## COMMANDS

```bash
uv sync                                    # Install deps + editable package
uv run python scripts/build_notebooks.py   # Regenerate all notebooks
uv run python -m ipykernel install --user --name ujiindoorloc-capstone
uv run jupyter lab                         # Run notebooks 00→05 in order
```

## NOTES

- **No tests exist yet** — project has no pytest/unittest infrastructure
- **No CI/build automation** — no GitHub Actions, Makefile, etc.
- **Shiny app deferred** — README says "will be Python shiny later, not R"
- **Coordinate regression deferred** — out of current scope
- `Elements of AI - Full Course/` is read-only course context — do not modify
