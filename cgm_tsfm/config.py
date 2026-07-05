"""Configuration: paths, targets, windowing, encoder, and CV settings.

The target names and CV protocol deliberately MIRROR Liuyi's existing
`diabetes-fitbit/analysis/glucose_cognitive_ml/config.py` so that the TSFM arm's
results are directly comparable to Mack's hand-crafted-feature arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Targets — the three ARC/PARC cognitive subtests (Grids, Symbols, Prices).
# Names match diabetes-fitbit config.py:13-17 exactly.
# NOTE: `prices_cognitive_score` is INVERTED in the raw data (higher = worse
# performance). See PRICES_IS_INVERTED below and docs/01_PIPELINE_DESIGN.md.
# ---------------------------------------------------------------------------
COGNITIVE_TARGETS: list[str] = [
    "grids_cognitive_score",
    "symbols_cognitive_score",
    "prices_cognitive_score",
]

PRICES_IS_INVERTED = True  # negate prices before modeling so higher = better

# Column names in the merged CSVs (see diabetes-fitbit/analysis/.../data.py).
SUBJECT_COL = "subid"      # participant id — the CV grouping key
SESSION_COL = "session"    # one row = one cognitive-test session
GLUCOSE_COL = "Glucose_Before_Test"   # string-encoded list of pre-test readings
COHORT_COL = "cohort"

CGM_SAMPLING_MINUTES = 5   # Dexcom G6 cadence (per diabetes-fitbit + K01)

# Reproducibility / CV — mirrors diabetes-fitbit config.py:9-11
RANDOM_STATE = 42
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3

# Where the real data lives once available (mirrors diabetes-fitbit layout).
# Not required for the synthetic demo.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Merged_glucose_data"
COHORT_FILES = [
    "Cohort1_scores_merged_with_glucose.csv",
    "Cohort2_scores_with_glucose.csv",
]

# Cache directory for extracted embeddings (Chronos forward pass is the
# expensive step — extract once, reuse across regressors/targets).
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embeddings"


@dataclass
class WindowConfig:
    """How to turn a variable-length pre-test glucose array into model input.

    Chronos handles variable length natively (it left-pads with NaN internally),
    so windowing is optional. But capping to the most-recent `max_readings`
    keeps embeddings comparable across sessions and lets us focus on the
    physiologically relevant lookback (the K01's hypothesis targets the ~2h
    before a test).
    """
    min_readings: int = 3        # drop sessions with fewer valid readings
    max_readings: int | None = None  # keep only the most-recent N (None = keep all)
    # e.g. 30 readings * 5 min = 2.5 h (the window diabetes-fitbit's augmenter used)


@dataclass
class EncoderConfig:
    """Frozen TSFM encoder settings."""
    # "chronos" uses the installed `chronos` package (Bolt or T5, auto-detected);
    # "mock" returns deterministic pseudo-embeddings so the pipeline runs with
    # no model download (useful for smoke tests / CI / offline).
    encoder_type: str = "chronos"
    pretrained_model: str = "amazon/chronos-bolt-small"
    pooling: str = "mean"        # "mean" | "last"  (mean over patch/token axis)
    device: str = "cpu"          # "cpu" | "cuda" | "mps"
    torch_dtype: str = "float32"
    batch_size: int = 32


@dataclass
class PipelineConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    targets: list[str] = field(default_factory=lambda: list(COGNITIVE_TARGETS))
    outer_splits: int = OUTER_CV_SPLITS
    inner_splits: int = INNER_CV_SPLITS
    random_state: int = RANDOM_STATE
