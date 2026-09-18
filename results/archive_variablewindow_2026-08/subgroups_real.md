# Subgroup signal search — glucose regimes

- **Data**: REAL  ·  **Encoder**: `amazon/chronos-bolt-small`  ·  **target-norm**: `none`
- **Generated**: 2026-08-01T17:46:46
- Per subgroup we re-run Arm A (Chronos embeddings + Ridge/SVR) under subject-grouped CV.
- **R² vs a mean-predictor; higher = better; 0 = no better than guessing the subgroup's average.**

> ⚠️ **Exploratory.** Subgroups shrink the subject count (shown), so grouped-CV R² here is noisy; read directions/trends, not precise values. Subgroups with fewer subjects than the fold count are skipped.

| subgroup | sessions | subjects | grids | symbols | prices | note |
|---|--:|--:|--:|--:|--:|---|
| all sessions (reference) | 956 | 20 | -0.114 | -0.052 | -0.012 | grouped CV, 5 folds |
| hypo  (any reading < 70) | 201 | 19 | -0.224 | -0.109 | -0.086 | grouped CV, 5 folds |
| hyper (any reading > 250) | 449 | 20 | -0.094 | -0.134 | -0.087 | grouped CV, 5 folds |
| any excursion (low OR high) | 577 | 20 | -0.118 | -0.246 | -0.018 | grouped CV, 5 folds |
| in-range only (70–180) | 143 | 19 | -0.153 | -0.289 | -0.110 | grouped CV, 5 folds |
