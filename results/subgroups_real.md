# Subgroup signal search — glucose regimes

- **Data**: REAL  ·  **Encoder**: `amazon/chronos-bolt-small`  ·  **target-norm**: `none`
- **Generated**: 2026-07-21T00:45:59
- Per subgroup we re-run Arm A (Chronos embeddings + Ridge/SVR) under subject-grouped CV.
- **R² vs a mean-predictor; higher = better; 0 = no better than guessing the subgroup's average.**

> ⚠️ **Exploratory.** Subgroups shrink the subject count (shown), so grouped-CV R² here is noisy; read directions/trends, not precise values. Subgroups with fewer subjects than the fold count are skipped.

| subgroup | sessions | subjects | grids | symbols | prices | note |
|---|--:|--:|--:|--:|--:|---|
| all sessions (reference) | 956 | 20 | -0.089 | -0.056 | -0.012 | grouped CV, 5 folds |
| hypo  (any reading < 70) | 201 | 19 | -0.187 | -0.094 | -0.090 | grouped CV, 5 folds |
| hyper (any reading > 250) | 449 | 20 | -0.077 | -0.121 | -0.087 | grouped CV, 5 folds |
| any excursion (low OR high) | 577 | 20 | -0.085 | -0.242 | -0.018 | grouped CV, 5 folds |
| in-range only (70–180) | 143 | 19 | -0.175 | -0.296 | -0.111 | grouped CV, 5 folds |
