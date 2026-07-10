"""Arm A — frozen-embedding regression with subject-grouped nested CV.

Given the (N, d_model) Chronos embeddings and per-session cognitive scores, fit
classical regressors (Ridge / SVR / Linear) and report RMSE / MAE / R² under the
SAME evaluation protocol as the advisor's hand-crafted-feature pipeline
(diabetes-fitbit): nested cross-validation with an outer 5-fold GroupKFold on
`subid` for unbiased estimation and an inner 3-fold GroupKFold GridSearchCV for
tuning, StandardScaler fit on train folds only.

Because the two arms share this protocol, their numbers are directly comparable:
the only thing that changes is the feature set (512-ish-dim Chronos embedding vs
43 hand-crafted glycemic features).

`cv_scheme="session"` swaps the grouping for a plain KFold that ignores subject
boundaries. Comparing "group" vs "session" R² is the diagnostic that told the advisor's
team the signal is mostly subject-baseline rather than generalizable — we run the
same diagnostic here so the TSFM arm's conclusion is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from .data import CGMDataset, within_subject_normalize


def get_regression_models() -> dict[str, tuple]:
    """Model + hyperparameter grid, mirroring diabetes-fitbit config.py:22-49
    (minus XGBoost, which is imported lazily in run_arm_a if available)."""
    return {
        "Ridge": (Ridge(random_state=C.RANDOM_STATE),
                  {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
        "SVR": (SVR(kernel="rbf"),
                {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]}),
        "Linear": (LinearRegression(), {}),
    }


@dataclass
class FoldResult:
    rmse: float
    mae: float
    r2: float


@dataclass
class ModelResult:
    name: str
    folds: list[FoldResult] = field(default_factory=list)

    def _agg(self, attr: str) -> tuple[float, float]:
        vals = np.array([getattr(f, attr) for f in self.folds])
        return float(vals.mean()), float(vals.std())

    def summary_row(self) -> dict:
        rmse_m, rmse_s = self._agg("rmse")
        mae_m, mae_s = self._agg("mae")
        r2_m, r2_s = self._agg("r2")
        return {
            "model": self.name,
            "rmse": f"{rmse_m:.3f}±{rmse_s:.3f}",
            "mae": f"{mae_m:.3f}±{mae_s:.3f}",
            "r2": f"{r2_m:.3f}±{r2_s:.3f}",
            "r2_mean": r2_m,
        }


def _make_splitter(n_splits: int, scheme: str):
    if scheme == "group":
        return GroupKFold(n_splits=n_splits)
    if scheme == "session":
        return KFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    raise ValueError(f"unknown cv_scheme: {scheme}")


def nested_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator,
    param_grid: dict,
    outer_splits: int = C.OUTER_CV_SPLITS,
    inner_splits: int = C.INNER_CV_SPLITS,
    cv_scheme: str = "group",
    pca_components: int | None = None,
) -> ModelResult:
    """Nested CV. Outer loop estimates generalization; inner loop tunes.

    A leakage guard asserts train/test subjects are disjoint on every fold
    (mirrors diabetes-fitbit regression.py:33-35).

    `pca_components`: if set, insert PCA(n_components) between the scaler and the
    model. PCA is fit *inside* the pipeline, so it sees only each fold's train
    split — no leakage. Reduces the high-dim (e.g. 512-d) embeddings that make
    linear models ill-conditioned."""
    outer = _make_splitter(outer_splits, cv_scheme)
    result = ModelResult(name=getattr(estimator, "_cgm_name", estimator.__class__.__name__))

    split_args = (X, y, groups) if cv_scheme == "group" else (X, y)
    for train_idx, test_idx in outer.split(*split_args):
        if cv_scheme == "group":
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), \
                "subject leakage between train and test!"

        steps = [("scaler", StandardScaler())]
        if pca_components:
            steps.append(("pca", PCA(n_components=pca_components, random_state=C.RANDOM_STATE)))
        steps.append(("model", estimator))
        pipe = Pipeline(steps)
        if param_grid:
            inner = _make_splitter(inner_splits, cv_scheme)
            gs = GridSearchCV(
                pipe, param_grid, scoring="neg_mean_squared_error",
                cv=inner, n_jobs=-1,
            )
            fit_kwargs = {"groups": groups[train_idx]} if cv_scheme == "group" else {}
            gs.fit(X[train_idx], y[train_idx], **fit_kwargs)
            best = gs.best_estimator_
        else:
            best = pipe.fit(X[train_idx], y[train_idx])

        pred = best.predict(X[test_idx])
        yt = y[test_idx]
        result.folds.append(FoldResult(
            rmse=float(np.sqrt(mean_squared_error(yt, pred))),
            mae=float(mean_absolute_error(yt, pred)),
            r2=float(r2_score(yt, pred)),
        ))
    return result


def run_arm_a(
    dataset: CGMDataset,
    embeddings: np.ndarray,
    targets: list[str] | None = None,
    cv_scheme: str = "group",
    include_baseline: bool = True,
    pca_components: int | None = None,
    target_norm: str = "none",
) -> dict[str, dict]:
    """Run Arm A for every target. Returns {target: {"n","n_subjects","rows"}}.

    One model/pipeline per target (single-target regression, per the advisor).
    A DummyRegressor(mean) baseline is always included: R² is measured relative
    to it, so a model only "works" if its R² is clearly > 0.

    `pca_components`: optional PCA reduction (fit per-fold) applied to the feature
    matrix before each model; clamped to the number of features. None = no PCA.
    `target_norm`: "none" | "center" | "zscore" — within-subject target
    normalization (see data.within_subject_normalize). With centering, metrics are
    on the deviation-from-personal-baseline scale and R² measures within-subject
    predictability.
    """
    targets = targets or list(dataset.targets.keys())
    models = get_regression_models()

    # XGBoost if the install works (libomp is often missing on macOS).
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (
            XGBRegressor(random_state=C.RANDOM_STATE, verbosity=0),
            {"model__n_estimators": [100, 300], "model__max_depth": [3, 5],
             "model__learning_rate": [0.03, 0.1]},
        )
    except Exception:
        pass

    all_results: dict[str, dict] = {}
    for target in targets:
        mask = dataset.target_mask(target)
        X, y, g = embeddings[mask], dataset.targets[target][mask], dataset.subjects[mask]
        y = within_subject_normalize(y, g, target_norm)   # no-op when target_norm="none"
        # clamp PCA to available feature count (e.g. 43 for hand-crafted, 256 for bolt-tiny)
        eff_pca = min(pca_components, X.shape[1]) if pca_components else None

        rows = []
        if include_baseline:
            base = nested_group_cv(
                X, y, g, DummyRegressor(strategy="mean"), {},
                cv_scheme=cv_scheme,  # PCA irrelevant for a mean predictor
            )
            base.name = "Baseline(mean)"
            rows.append(base.summary_row())

        for name, (est, grid) in models.items():
            est._cgm_name = name
            res = nested_group_cv(X, y, g, est, grid, cv_scheme=cv_scheme, pca_components=eff_pca)
            rows.append(res.summary_row())

        all_results[target] = {"n": int(mask.sum()), "n_subjects": int(len(np.unique(g))), "rows": rows}
    return all_results


def format_results(results: dict[str, dict], cv_scheme: str, title: str = "Arm A") -> str:
    lines = [f"\n=== {title} results  (cv_scheme={cv_scheme}) ===",
             "R² is vs a mean-predictor; R²>0 means the embeddings help.\n"]
    for target, info in results.items():
        lines.append(f"[{target}]  (n={info['n']} sessions, {info['n_subjects']} subjects)")
        lines.append(f"  {'model':16s} {'RMSE':>14s} {'MAE':>14s} {'R2':>14s}")
        for r in info["rows"]:
            lines.append(f"  {r['model']:16s} {r['rmse']:>14s} {r['mae']:>14s} {r['r2']:>14s}")
        lines.append("")
    return "\n".join(lines)
