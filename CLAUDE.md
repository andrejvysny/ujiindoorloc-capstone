# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo state

Capstone project for FIIT _Objavovanie znalostí_ (OZNAL). **No source code exists yet** — only inputs:

- `data/raw/trainingData.csv` (19,937 × 529), `data/raw/validationData.csv` (1,111 × 529) — UJIIndoorLoc.
- `Assignment.pdf` — capstone requirements.
- `SUMMARY.md` — dataset analysis + chosen project direction. **Read this first** before proposing pipeline changes; the modelling decisions there are already vetted.
- `Elements of AI - Full Course/` — reference materials only (lectures, R/Tidyverse labs, Shiny tutorials). Do **not** edit; treat as read-only course context.

Directory name says `python`, but the course solutions and assignment-required Shiny app are R. Confirm language with the user before scaffolding — it has not been decided in code yet. If Python: use `uv` (per global rules), no venv/pyproject exists yet so you create them. If R: project will need an `app.R` Shiny app and Rmd report.

## Dataset semantics — critical gotchas

These are non-obvious from the CSV and will silently corrupt models:

- **`100` is not signal strength.** It is the sentinel for "WAP not detected." Real RSSI is roughly `-104..0` dBm. Always replace `100 → -110` (or generate binary detected-flag features) before any scaling/PCA/distance computation.
- **Validation is not a random split.** It is a different time period, 1 user (vs 18 in train), different phones, and 55 WAPs unseen in training. Treat `validationData.csv` as the realistic held-out test set; do **not** re-shuffle train+val.
- **Metadata columns are leakage-prone.** `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP` should not be used as predictors. Targets are `BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`. Predictors are `WAP001..WAP520`.
- **Building-only classification is ~100% accurate** with trivial models — don't pitch it as the main task. Primary target is the 13-class `BUILDINGID + FLOOR` combination; secondary is coordinate regression.
- **High multicollinearity** among WAPs (366 pairs with `|r|>0.8`). Prefer ridge / elastic net / PCA / RF over plain OLS or unregularized logistic regression.
- **Class imbalance**: floor 4 has few rows, building 2 dominates. Report macro F1 and balanced accuracy alongside accuracy.

## Assignment constraints

From `Assignment.pdf` / `SUMMARY.md`:

- Two ML scenarios required. Chosen: **Scenario 2** (parametric vs non-parametric) and **Scenario 5** (dimensionality reduction before modelling).
- Deliverables: reproducible preprocessing, model comparison, visualizations, one-page summary, and a **Shiny app** with dynamic model fitting / parameter controls (target, preprocessing, model, PCA components, regularization, etc.).
- UJIIndoorLoc is a known benchmark — **do not copy public notebooks**. Build preprocessing, modelling, and app from scratch and document choices.

## Layout conventions (when code is added)

When scaffolding, keep raw data untouched; write derived features and model artifacts under a sibling `outputs/` or `artifacts/` directory rather than mutating `data/`. The Shiny app and report should both consume the same preprocessing function so results stay consistent across the report and the live app.
