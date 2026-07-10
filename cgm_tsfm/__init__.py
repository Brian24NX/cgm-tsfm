"""cgm_tsfm — Time-Series Foundation Model (TSFM) pipeline for CGM → cognition.

This package implements the *raw-data / TSFM arm* of the T1D glucose–cognition
project (Brian Zhou, supervised by the advisor). It takes a variable-length window of
continuous-glucose-monitor (CGM) readings recorded *before* a cognitive test and
predicts a single continuous cognitive score, using a frozen pretrained
time-series foundation model (Amazon Chronos) as the feature encoder.

Two arms are provided (see docs/01_PIPELINE_DESIGN.md):

  Arm A — frozen Chronos embeddings + a classical regressor (Ridge / SVR).
          This mirrors the ICML'25 "representations-in-tsfms" SVM experiment,
          swapped to regression, and is directly comparable to the existing
          hand-crafted-feature pipeline in `diabetes-fitbit` (same grouped CV).

  Arm B — frozen Chronos encoder + a small trainable neural head, adapted from
          Ben's `FoundationModelClassifier` (see cgm_tsfm/ben_adapter/).

Nothing here requires the real CGM data to run: `data.py` ships a synthetic
generator that produces the *exact* schema of the real dataset, so the whole
pipeline is runnable today and drop-in ready for the advisor's real CSVs.
"""

__version__ = "0.1.0"
