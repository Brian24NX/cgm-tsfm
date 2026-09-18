"""Configuration: paths, targets, windowing, encoder, and CV settings.

The target names and CV protocol deliberately MIRROR the advisor's existing
`diabetes-fitbit/analysis/glucose_cognitive_ml/config.py` so that the TSFM arm's
results are directly comparable to Mack's hand-crafted-feature arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Targets — the three ARC/PARC cognitive subtests (Grids, Symbols, Prices).
# Names match diabetes-fitbit config.py:13-17 exactly.
# NOTE (confirmed by the advisor, 2026-07): for ALL THREE scores, LOWER = better
# — they are error / response-time-type measures (per the ARC app), not accuracy.
# So we keep every target in its RAW orientation (no negation) and read all three
# consistently as "lower = better". Sign choice does NOT change R²/RMSE/MAE (those
# are invariant to negating the target); it only affects interpretation.
# ---------------------------------------------------------------------------
COGNITIVE_TARGETS: list[str] = [
    "grids_cognitive_score",
    "symbols_cognitive_score",
    "prices_cognitive_score",
]

PRICES_IS_INVERTED = False  # advisor: all three are lower=better; keep raw orientation (no negation)

# Column names in the merged CSVs (see diabetes-fitbit/analysis/.../data.py).
SUBJECT_COL = "subid"      # participant id — the CV grouping key
SESSION_COL = "session"    # one row = one cognitive-test session
GLUCOSE_COL = "Glucose_Before_Test"   # string-encoded list of pre-test readings
COHORT_COL = "cohort"

CGM_SAMPLING_MINUTES = 5   # Dexcom G6 cadence (per diabetes-fitbit + K01)

# ---------------------------------------------------------------------------
# DEFAULT WINDOW (changed 2026-08, after the window sweep).
# Every session now uses the SAME span of glucose: the 24 readings (2.0 h)
# immediately before the test. Sessions with fewer than 24 readings are dropped.
# Cost: 916 of 956 sessions kept (95.8%), all 20 participants.
#
# Why 2 h, and why it was chosen this way: the K01 hypothesis concerns roughly
# the 2 h before a test, so this is the PRE-COMMITTED choice on physiological
# grounds. It was NOT picked because it scored best -- it did not. Windows from
# 15 min to 3 h were all measured (results/uniform_window_real.md) and the
# differences between them are ~80x smaller than the fold-to-fold spread, so the
# data cannot distinguish them. Picking the best-scoring of 48 measured cells
# would have been reading noise.
#
# Before this change the input length varied from 3 to 288 readings per session,
# and length itself carried time-of-day information (longer window <=> morning
# test, r = -0.506). The uniform window removes that.
# Pass --variable-window to any runner to reproduce the old behaviour.
# ---------------------------------------------------------------------------
DEFAULT_MAX_READINGS = 24      # 24 * 5 min = 2.0 h
DEFAULT_REQUIRE_FULL = True

# Reproducibility / CV — mirrors diabetes-fitbit config.py:9-11
RANDOM_STATE = 42
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3

# Where the real data lives once available (mirrors diabetes-fitbit layout).
# Not required for the synthetic demo.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Merged_glucose_data"
COHORT_FILES = [
    "Cohort1_scores_merged_with_glucose.csv",
    # Phil's corrected Cohort2 (2026-07). The original Cohort2_scores_with_glucose.csv
    # had 44/240 empty-glucose rows, which dropped one Cohort2 subject entirely →
    # only 19 usable subjects. This "Updated_" file backfills the missing glucose
    # (1/240 empty) so all 6 Cohort2 subjects survive → the full 20 patients.
    # Original file kept on disk (data/Merged_glucose_data/) for provenance.
    "Updated_Cohort2_scores_with_glucose.csv",
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
    max_readings: int | None = DEFAULT_MAX_READINGS  # keep only the most-recent N (None = keep all)
    # e.g. 30 readings * 5 min = 2.5 h (the window diabetes-fitbit's augmenter used)

    # UNIFORM WINDOWS (added 2026-08). With require_full=False (the default),
    # `max_readings` only CAPS long sessions -- short ones are kept at whatever
    # length they happen to be, so the input length still varies session to
    # session. That is what every earlier "window sweep" actually measured.
    # With require_full=True, sessions with fewer than `max_readings` readings
    # are DROPPED, so every surviving session has EXACTLY the same span. That
    # removes session length as a source of variation between sessions.
    # Cost on the real data: 24 readings (2.0 h) keeps 916/956 sessions and all
    # 20 participants; 36 readings (3.0 h) keeps only 490.
    require_full: bool = DEFAULT_REQUIRE_FULL


@dataclass
class EncoderConfig:
    """Frozen TSFM encoder settings."""
    # "chronos" uses the installed `chronos` package (Bolt or T5, auto-detected);
    # "mock" returns deterministic pseudo-embeddings so the pipeline runs with
    # no model download (useful for smoke tests / CI / offline).
    encoder_type: str = "chronos"
    pretrained_model: str = "amazon/chronos-bolt-small"
    # Pooling over the token axis of embed()'s (B, num_patches+1, d_model) output:
    #   "mean"          -> average of all tokens (patches + the summary token)
    #   "mean_patches"  -> average of the patch tokens only, excluding the summary token
    #   "reg" / "last"  -> the trailing summary token itself ([REG] for Bolt, EOS for T5).
    #                      NOTE "last" means the SUMMARY token, not the most-recent patch.
    # All three are batch-invariant as of the 2026-07-30 fix (see encoders.py).
    pooling: str = "mean"
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
