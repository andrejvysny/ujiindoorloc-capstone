# Indoor Building & Floor Classification — UJIIndoorLoc (FIIT OZNAL Capstone)

Classify the user's **indoor building + floor** from WiFi RSSI fingerprints
(UJIIndoorLoc, UCI). Compare **parametric vs non-parametric** classifiers
(Scenario 2) and study **PCA / LDA dimensionality reduction** before
classification (Scenario 5).

> Scope of this pass: Python + Jupyter only. **No regression. No Shiny app.**
> Both are intentionally deferred.

---

## Dataset

- `data/raw/trainingData.csv` — 19,937 × 529 (UJIIndoorLoc training/reference).
- `data/raw/validationData.csv` — 1,111 × 529 (different time window, fewer
  users/devices — used as the **held-out test set**).

Predictors used: `WAP001 … WAP520` (520 WiFi RSSI features).
Target used: `building_floor = "B<BUILDINGID>_F<FLOOR>"` (multiclass, 13 classes).

**Critical gotcha:** the value `100` in any WAP column means _"WAP not detected"_.
Real RSSI is roughly `-104..0` dBm. Preprocessing replaces `100 → -110`.

Columns explicitly **never** used as predictors:
`LONGITUDE, LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION, USERID,
PHONEID, TIMESTAMP`.

---

## Environment

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m ipykernel install --user --name ujiindoorloc-capstone
uv run jupyter lab
```

The package `ujiindoorloc` (in `src/`) is installed editable by `uv sync`, so
notebooks just `from ujiindoorloc import ...`.

---

## Notebook execution order

Run top-to-bottom in this order:

1. `notebooks/00_project_overview.ipynb` — scope, target, scenarios.
2. `notebooks/01_data_audit_and_eda.ipynb` — full EDA: sparsity, coverage,
   class distributions, train↔validation shift, campus GPS geometry, WAP×WAP
   correlation matrix (clustered), per-class fingerprint & detection-rate
   heatmaps, ANOVA feature relevance, class-centroid distance heatmap,
   per-WAP drift, sparsity heatmap.
3. `notebooks/02_preprocessing_and_targets.ipynb` — pipeline definitions,
   leakage checks, processed-feature manifest.
4. `notebooks/03_scenario_2_model_comparison.ipynb` — Logistic Regression, LDA,
   QDA, kNN, Decision Tree, Random Forest.
5. `notebooks/04_scenario_5_dimensionality_reduction.ipynb` — PCA + LR / kNN,
   LDA-projection + LR.
6. `notebooks/05_final_comparison_and_interpretation.ipynb` — final summary,
   model recommendation, limitations.

---

## Outputs

- `reports/figures/` — saved PNGs (EDA + model comparison + PCA/LDA).
- `reports/tables/` — saved CSVs (metrics, coverage, correlations, etc.).
- `models/` — optional saved model artifacts.

---

## Evaluation philosophy

Classes are imbalanced (floor 4 is small, building 2 dominates) and the
validation split is a real _time / device / user_ shift. The headline metrics
are therefore **balanced accuracy** and **macro F1**, with confusion matrices
for diagnosis. Plain accuracy is reported but is _not_ the primary criterion.

Final model selection prioritises (in order): balanced accuracy → macro F1 →
confusion-matrix quality → simplicity → explainability → runtime stability.

---

## Scope notes

- Shiny app — deferred (will be Python `shiny` later, not R).
- Coordinate regression — deferred.
- Advanced feature selection — out of scope; modules are kept modular so it can
  be added later without rework.
