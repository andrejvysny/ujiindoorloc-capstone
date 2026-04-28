"""Model factory functions. Each returns an unfitted sklearn estimator/pipeline.

Defaults are tuned to be **safe and explainable**, not state-of-the-art.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .constants import RANDOM_STATE


# ---------------------------------------------------------------------------
# Parametric / statistical
# ---------------------------------------------------------------------------


def build_logistic_regression(C: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    """Multinomial logistic regression. Use scaled features."""
    # Note: `n_jobs` was removed because `lbfgs` is single-threaded in
    # sklearn>=1.8 and passing it triggers a FutureWarning.
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def build_lda(solver: str = "svd") -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver=solver)


def build_qda(reg_param: float = 0.0) -> QuadraticDiscriminantAnalysis:
    """QDA — unstable in high dim. Use AFTER PCA in practice."""
    return QuadraticDiscriminantAnalysis(reg_param=reg_param)


# ---------------------------------------------------------------------------
# Non-parametric
# ---------------------------------------------------------------------------


def build_knn(n_neighbors: int = 5, weights: str = "distance") -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="euclidean",
        n_jobs=-1,
    )


def build_decision_tree(
    max_depth: int | None = 20,
    min_samples_leaf: int = 5,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Scenario 5 — DR + classifier pipelines
# ---------------------------------------------------------------------------


def build_pca_logistic_pipeline(n_components: int, C: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", build_logistic_regression(C=C)),
        ]
    )


def build_pca_knn_pipeline(n_components: int, n_neighbors: int = 5) -> Pipeline:
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", build_knn(n_neighbors=n_neighbors)),
        ]
    )


def build_pca_qda_pipeline(n_components: int, reg_param: float = 1e-2) -> Pipeline:
    """QDA after PCA — much more stable than raw QDA on 500+ correlated features."""
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
            ("clf", build_qda(reg_param=reg_param)),
        ]
    )


def build_lda_projection_logistic_pipeline(
    n_components: int | None = None, C: float = 1.0
) -> Pipeline:
    """Use LDA as a supervised dimensionality reducer, then logistic regression.

    `n_components` defaults to (n_classes - 1) inside LDA when None is passed.
    """
    return Pipeline(
        steps=[
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="svd", n_components=n_components
                ),
            ),
            ("clf", build_logistic_regression(C=C)),
        ]
    )


def build_lda_projection_knn_pipeline(
    n_components: int | None = None, n_neighbors: int = 5
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="svd", n_components=n_components
                ),
            ),
            ("clf", build_knn(n_neighbors=n_neighbors)),
        ]
    )
