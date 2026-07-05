"""Data loading + windowing, and a synthetic generator with the real schema.

The real dataset (Liuyi's `data/Merged_glucose_data/`) is one row per cognitive-
test session, with a pre-clipped, variable-length array of CGM readings
(`Glucose_Before_Test`) and three cognitive scores. The files are gitignored and
live only on Liuyi's machine, so `load_real_data` mirrors diabetes-fitbit's own
loader (it will run unchanged once the CSVs are present), and
`generate_synthetic_data` produces the *identical schema* so the whole pipeline
is runnable today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Parsing the raw glucose string  (mirrors diabetes-fitbit data.py:11-24)
# ---------------------------------------------------------------------------
def parse_glucose(raw) -> np.ndarray | None:
    """Parse a `Glucose_Before_Test` cell into a 1-D float array (NaNs dropped).

    Cohort2 stores literal ``nan`` tokens inside the list string, which break
    ``ast.literal_eval`` — so we use ``eval`` with a restricted namespace that
    maps ``nan`` to ``np.nan`` (exactly as the existing pipeline does).
    Empty / ``[]`` / missing → ``None``.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.asarray(raw, dtype=float)
    else:
        s = str(raw).strip()
        if not s or s in ("[]", "nan", "None"):
            return None
        try:
            parsed = eval(s, {"__builtins__": {}}, {"nan": np.nan, "NaN": np.nan})
        except Exception:
            return None
        arr = np.asarray(parsed, dtype=float)
    arr = arr[~np.isnan(arr)]           # strip missing readings
    return arr if arr.size > 0 else None


def apply_window(glucose: np.ndarray, window: C.WindowConfig) -> np.ndarray | None:
    """Cap a glucose array to the most-recent `max_readings`; drop if too short."""
    if glucose is None or glucose.size < window.min_readings:
        return None
    if window.max_readings is not None and glucose.size > window.max_readings:
        glucose = glucose[-window.max_readings:]   # most-recent readings
    return glucose


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class CGMDataset:
    """A windowed CGM-cognition dataset ready for the TSFM pipeline.

    Attributes
    ----------
    windows : list[np.ndarray]
        One variable-length 1-D glucose array per session (already windowed).
    subjects : np.ndarray
        Subject id per session — the CV grouping key.
    sessions : np.ndarray
        Session id per session.
    targets : dict[str, np.ndarray]
        Maps target name -> per-session score array (float, NaN where missing).
        `prices_cognitive_score` is already sign-corrected if config says so.
    """
    windows: list[np.ndarray]
    subjects: np.ndarray
    sessions: np.ndarray
    targets: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.windows)

    def target_mask(self, target: str) -> np.ndarray:
        """Boolean mask of sessions with a non-missing value for `target`."""
        return ~np.isnan(self.targets[target])

    def summary(self) -> str:
        n_subj = len(np.unique(self.subjects))
        lens = np.array([w.size for w in self.windows])
        lines = [
            f"CGMDataset: {len(self)} sessions, {n_subj} subjects",
            f"  window length (readings): min={lens.min()} median={int(np.median(lens))} "
            f"max={lens.max()}  (~{np.median(lens) * C.CGM_SAMPLING_MINUTES / 60:.1f} h median)",
        ]
        for t, y in self.targets.items():
            m = ~np.isnan(y)
            lines.append(f"  {t}: n={m.sum()} mean={np.nanmean(y):.3f} std={np.nanstd(y):.3f}")
        return "\n".join(lines)


def within_subject_normalize(
    y: np.ndarray, subjects: np.ndarray, mode: str = "center"
) -> np.ndarray:
    """Normalize each target value against that subject's OWN sessions.

    mode:
      "none"    → unchanged.
      "center"  → subtract the subject's mean score (predict deviation from personal baseline).
      "zscore"  → also divide by the subject's std (guarded; 0-std subject → 0).

    Why: Mack's arm + our synthetic diagnostic both point to the glucose→cognition
    signal being dominated by BETWEEN-subject baseline differences. Centering per
    subject strips that baseline out, so the model must predict moment-to-moment
    fluctuation — the effect the K01 actually hypothesizes.

    ⚠️ This uses each subject's own sessions to compute their mean/std (ORACLE
    centering). Under leave-subjects-out / GroupKFold CV a test subject is entirely
    held out, so their personal mean would be unknown in a real prospective setting.
    Treat within-subject results as answering "is there a within-subject signal at
    all?" — not as a deployable new-subject predictor. (Same framing as
    diabetes-fitbit's next_steps within-subject analysis.)
    """
    if mode == "none":
        return y
    y = np.asarray(y, dtype=float).copy()
    out = np.empty_like(y)
    for s in np.unique(subjects):
        idx = subjects == s
        vals = y[idx]
        m = vals.mean()
        if mode == "zscore":
            sd = vals.std()
            out[idx] = (vals - m) / sd if sd > 1e-8 else 0.0
        elif mode == "center":
            out[idx] = vals - m
        else:
            raise ValueError(f"unknown target-norm mode: {mode}")
    return out


def _frame_to_dataset(df: pd.DataFrame, window: C.WindowConfig) -> CGMDataset:
    """Turn a parsed dataframe (glucose already arrays) into a CGMDataset."""
    windows, subjects, sessions = [], [], []
    targets: dict[str, list] = {t: [] for t in C.COGNITIVE_TARGETS}

    for _, row in df.iterrows():
        w = apply_window(row["glucose"], window)
        if w is None:
            continue
        windows.append(w.astype(np.float32))
        subjects.append(row[C.SUBJECT_COL])
        sessions.append(row[C.SESSION_COL])
        for t in C.COGNITIVE_TARGETS:
            val = float(row[t]) if pd.notna(row.get(t)) else np.nan
            if t == "prices_cognitive_score" and C.PRICES_IS_INVERTED and not np.isnan(val):
                val = -val                # higher = better after negation
            targets[t].append(val)

    return CGMDataset(
        windows=windows,
        subjects=np.asarray(subjects),
        sessions=np.asarray(sessions),
        targets={t: np.asarray(v, dtype=float) for t, v in targets.items()},
    )


# ---------------------------------------------------------------------------
# Real-data loader  (mirrors diabetes-fitbit data.py:26-68; runs once CSVs exist)
# ---------------------------------------------------------------------------
def load_real_data(
    data_dir: Path | str = C.DATA_DIR,
    window: C.WindowConfig | None = None,
) -> CGMDataset:
    """Load the two merged cohort CSVs and window them.

    Harmonizes the differing cohort column names, keeps the common columns,
    parses `Glucose_Before_Test`, and returns a windowed CGMDataset.
    """
    window = window or C.WindowConfig()
    data_dir = Path(data_dir)
    frames = []
    for fname in C.COHORT_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing {fpath}. The merged cohort CSVs are gitignored and live "
                "on Liuyi's machine — request them, or use generate_synthetic_data()."
            )
        d = pd.read_csv(fpath)
        # Harmonize cohort-specific column names (diabetes-fitbit data.py:42-43)
        d = d.rename(columns={
            "session_id": C.SESSION_COL,
            "combined time": "combined_time",
            "Unix Time": "combined_time",
        })
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["glucose"] = df[C.GLUCOSE_COL].apply(parse_glucose)
    return _frame_to_dataset(df, window)


# ---------------------------------------------------------------------------
# Synthetic generator — SAME schema, realistic structure, runnable today
# ---------------------------------------------------------------------------
def generate_synthetic_data(
    n_subjects: int = 20,
    sessions_per_subject: tuple[int, int] = (32, 67),
    window: C.WindowConfig | None = None,
    signal_strength: float = 0.25,
    within_subject_signal: float = 0.0,
    seed: int = C.RANDOM_STATE,
) -> CGMDataset:
    """Generate a synthetic CGM-cognition dataset matching the real schema.

    Design goals (so the pipeline behaves like the real problem):
      * one row per session; ~32-67 sessions per subject; ~20 subjects;
      * variable-length 5-min CGM windows with realistic mg/dL dynamics;
      * scores dominated by a *subject random effect* (baseline), plus a weak
        glucose effect, plus noise. This reproduces the real finding that most
        variance is between-subjects, so subject-grouped CV is much harder than
        session-level CV.

    Two independent signal knobs (so within-subject normalization can be tested):
      * `signal_strength` scales a BETWEEN-subject effect (driven by the subject's
        typical glucose level `mean_g`). It raises raw grouped-CV R², but is
        *removed* by within-subject centering.
      * `within_subject_signal` scales a WITHIN-subject effect (driven by the
        session's own recent trend, ~zero-mean within a subject). It *survives*
        within-subject centering — set it >0 to verify the pipeline detects a
        within-subject glucose→cognition signal when one exists (the pipeline's
        whole scientific purpose). Default 0 = no within-subject signal.
    Set both to 0 for a pure null dataset.
    """
    window = window or C.WindowConfig()
    rng = np.random.default_rng(seed)

    rows = []
    for s in range(n_subjects):
        subj_id = f"S{s:03d}"
        # Subject-specific glucose baseline (mg/dL) and cognitive baselines.
        subj_gluc_mean = rng.uniform(120, 190)
        subj_gluc_vol = rng.uniform(10, 35)
        base = {t: rng.normal(0, 1.0) for t in C.COGNITIVE_TARGETS}
        n_sess = rng.integers(sessions_per_subject[0], sessions_per_subject[1] + 1)

        for j in range(n_sess):
            L = int(rng.integers(20, 71))          # variable window length
            # Realistic CGM: mean-reverting random walk around subject mean.
            g = np.empty(L, dtype=float)
            g[0] = rng.normal(subj_gluc_mean, subj_gluc_vol)
            for t in range(1, L):
                g[t] = g[t - 1] + 0.6 * (subj_gluc_mean - g[t - 1]) / 4 \
                    + rng.normal(0, subj_gluc_vol / 2)
            g = np.clip(g, 40, 400)
            # occasional dropout (missing readings), like real CGM
            if rng.random() < 0.15:
                drop = rng.random(L) < 0.05
                g = g[~drop]
            if g.size < window.min_readings:
                continue

            # BETWEEN-subject effect: dominated by the subject's typical glucose
            # level (mean_g), so it's largely constant within a subject and is
            # removed by within-subject centering.
            mean_g = (g.mean() - 150) / 40.0
            frac_hypo = (g < 70).mean()
            frac_hyper = (g > 250).mean()
            true_effect = -0.4 * frac_hypo - 0.2 * mean_g + 0.15 * frac_hyper
            # WITHIN-subject effect: the session's own recent trend (last-first),
            # ~zero-mean within a subject → survives within-subject centering.
            recent_trend = (g[-1] - g[0]) / 30.0

            scores = {}
            for t in C.COGNITIVE_TARGETS:
                z = (base[t] + signal_strength * true_effect
                     + within_subject_signal * recent_trend + rng.normal(0, 0.7))
                if t == "grids_cognitive_score":
                    scores[t] = 0.33 + 0.3 * z
                elif t == "symbols_cognitive_score":
                    scores[t] = 1.80 + 0.5 * z
                else:  # prices — larger scale; store RAW (inverted) like real data
                    raw = 40.0 - 15.0 * z          # note: higher raw = worse
                    scores[t] = raw
            # random missingness in the scores (like real: 0.6-3.8%)
            for t in C.COGNITIVE_TARGETS:
                if rng.random() < 0.02:
                    scores[t] = np.nan

            rows.append({
                C.SUBJECT_COL: subj_id,
                C.SESSION_COL: f"{subj_id}_{j:03d}",
                "glucose": g.astype(np.float32),
                **scores,
            })

    df = pd.DataFrame(rows)
    return _frame_to_dataset(df, window)
