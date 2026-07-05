"""Mack's 43 hand-crafted glycemic features — faithful port for the head-to-head.

This is a line-for-line port of
`diabetes-fitbit/analysis/glucose_cognitive_ml/features.py` (the feature set whose
classical-ML arm got no predictive power), so the TSFM arm can be compared
against the *exact same* features under the *exact same* grouped CV — the only
thing that differs is the feature representation.

Every feature name and formula matches the original; source line numbers are
cited inline. Differences from the original are deliberate and minimal:
  * interface returns an (N, 43) matrix + names (not a per-row pandas Series),
    so it drops straight into `regression.run_arm_a` alongside Chronos embeddings;
  * `np.trapezoid` (NumPy>=2.0) is shimmed to `np.trapz` for NumPy 1.26 envs.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

# NumPy 2.0 renamed trapz -> trapezoid. diabetes-fitbit features.py:84 uses the
# new name; shim so this runs on NumPy 1.26 too (e.g. mod-actigraphy .venv).
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Exact names & order from diabetes-fitbit features.py:7-23
GLUCOSE_FEATURE_NAMES = [
    "gluc_mean", "gluc_std", "gluc_min", "gluc_max", "gluc_median",
    "gluc_range", "gluc_iqr", "gluc_skew", "gluc_kurtosis",
    "gluc_cv", "gluc_mage", "gluc_roc_std",
    "gluc_first", "gluc_last", "gluc_delta", "gluc_slope",
    "gluc_mean_roc", "gluc_max_abs_roc",
    "gluc_time_in_range", "gluc_time_below", "gluc_time_above",
    "gluc_auc_norm",
    "gluc_last5_0", "gluc_last5_1", "gluc_last5_2", "gluc_last5_3", "gluc_last5_4",
    "gluc_n_readings",
    "gluc_mag", "gluc_gvp", "gluc_j_index",
    "gluc_lbgi", "gluc_hbgi",
    "gluc_conga_4", "gluc_conga_12",
    "gluc_roc_iqr", "gluc_roc_skew", "gluc_roc_kurtosis",
    "gluc_pct_cv", "gluc_sd_roc_roc",
    "gluc_q10", "gluc_q90", "gluc_interdecile_range",
]


def extract_glucose_features(glucose_array) -> dict[str, float] | None:
    """Port of diabetes-fitbit features.py:26-159. Returns None for <3 readings."""
    if glucose_array is None or len(glucose_array) < 3:
        return None
    arr = np.asarray(glucose_array, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return None

    f: dict[str, float] = {}
    # Basic statistics (features.py:40-48)
    f["gluc_mean"] = np.mean(arr)
    f["gluc_std"] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    f["gluc_min"] = np.min(arr)
    f["gluc_max"] = np.max(arr)
    f["gluc_median"] = np.median(arr)
    f["gluc_range"] = f["gluc_max"] - f["gluc_min"]
    f["gluc_iqr"] = np.percentile(arr, 75) - np.percentile(arr, 25)
    f["gluc_skew"] = skew(arr)
    f["gluc_kurtosis"] = kurtosis(arr)
    # Coefficient of variation (features.py:51-55)
    f["gluc_cv"] = f["gluc_std"] / f["gluc_mean"] if f["gluc_mean"] != 0 else 0.0

    # Rate of change (features.py:58-63)
    diffs = np.diff(arr)
    f["gluc_mean_roc"] = np.mean(diffs)
    f["gluc_roc_std"] = np.std(diffs) if len(diffs) > 0 else 0.0
    f["gluc_max_abs_roc"] = np.max(np.abs(diffs)) if len(diffs) > 0 else 0.0

    # MAGE approximation (excursions > 1 SD) (features.py:66-68)
    threshold = f["gluc_std"]
    excursions = np.abs(diffs)[np.abs(diffs) > threshold]
    f["gluc_mage"] = np.mean(excursions) if len(excursions) > 0 else 0.0

    # Trend (features.py:71-75)
    f["gluc_first"] = arr[0]
    f["gluc_last"] = arr[-1]
    f["gluc_delta"] = arr[-1] - arr[0]
    f["gluc_slope"] = np.polyfit(np.arange(len(arr)), arr, 1)[0]

    # Clinical time-in-range 70-180 mg/dL (features.py:78-80)
    f["gluc_time_in_range"] = np.mean((arr >= 70) & (arr <= 180))
    f["gluc_time_below"] = np.mean(arr < 70)
    f["gluc_time_above"] = np.mean(arr > 180)

    # Normalized AUC (features.py:83-85)
    f["gluc_auc_norm"] = _trapz(arr) / len(arr) if len(arr) > 1 else arr[0]

    # Last 5 readings, edge-padded (features.py:88-93)
    last5 = arr[-5:] if len(arr) >= 5 else np.pad(arr, (5 - len(arr), 0), mode="edge")
    for i in range(5):
        f[f"gluc_last5_{i}"] = last5[i]

    # Metadata (features.py:96)
    f["gluc_n_readings"] = float(len(arr))

    # MAG (features.py:101)
    f["gluc_mag"] = np.mean(np.abs(diffs))
    # GVP -- Glycemic Variability Percentage (features.py:104-108)
    line_length = np.sum(np.sqrt(1 + diffs ** 2))
    straight = np.sqrt((len(arr) - 1) ** 2 + (arr[-1] - arr[0]) ** 2)
    f["gluc_gvp"] = (line_length / straight - 1) * 100 if straight > 0 else 0.0
    # J-index (features.py:111-113)
    f["gluc_j_index"] = 0.001 * (f["gluc_mean"] + f["gluc_std"]) ** 2

    # LBGI / HBGI -- Kovatchev (features.py:116-120)
    f_bg = 1.509 * (np.log(np.clip(arr, 1, None)) ** 1.084 - 5.381)
    f["gluc_lbgi"] = np.mean(10 * np.minimum(f_bg, 0) ** 2)
    f["gluc_hbgi"] = np.mean(10 * np.maximum(f_bg, 0) ** 2)

    # CONGA at lag 4 / 12 (features.py:123-128)
    for lag, label in [(4, "gluc_conga_4"), (12, "gluc_conga_12")]:
        if len(arr) > lag:
            f[label] = np.std(arr[lag:] - arr[:-lag], ddof=1)
        else:
            f[label] = 0.0

    # ROC distribution (features.py:131-139)
    f["gluc_roc_iqr"] = (np.percentile(diffs, 75) - np.percentile(diffs, 25)) if len(diffs) >= 4 else 0.0
    f["gluc_roc_skew"] = float(skew(diffs)) if len(diffs) >= 3 else 0.0
    f["gluc_roc_kurtosis"] = float(kurtosis(diffs)) if len(diffs) >= 3 else 0.0

    # Percent CV (features.py:142-146)
    f["gluc_pct_cv"] = (f["gluc_std"] / f["gluc_mean"]) * 100 if f["gluc_mean"] > 0 else 0.0
    # Acceleration SD (features.py:149-152)
    diffs2 = np.diff(diffs)
    f["gluc_sd_roc_roc"] = np.std(diffs2, ddof=1) if len(diffs2) > 1 else 0.0

    # Interdecile range (features.py:155-157)
    f["gluc_q10"] = np.percentile(arr, 10)
    f["gluc_q90"] = np.percentile(arr, 90)
    f["gluc_interdecile_range"] = f["gluc_q90"] - f["gluc_q10"]

    return f


def extract_handcrafted_features(windows: list[np.ndarray]) -> np.ndarray:
    """Return an (N, 43) feature matrix aligned to `windows`.

    NaN features (e.g. skew/kurtosis on a constant array) are filled with 0,
    matching diabetes-fitbit features.py:175. Rows too short to featurize (which
    CGMDataset windowing already excludes) become all-zero rows so the matrix
    stays aligned with the Chronos-embedding matrix.
    """
    rows = []
    for w in windows:
        feats = extract_glucose_features(w)
        if feats is None:
            rows.append([0.0] * len(GLUCOSE_FEATURE_NAMES))
        else:
            rows.append([feats[name] for name in GLUCOSE_FEATURE_NAMES])
    X = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0)
