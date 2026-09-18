# Rigor checks — the known-answer check + the shuffle test

- **Data**: REAL  ·  **Encoder**: `amazon/chronos-bolt-small`  ·  **Shuffle rounds**: 200
- **Generated**: 2026-08-01T18:54:44
- Model: `StandardScaler → PCA(32) → Ridge(alpha=10)`, subject-grouped 5-fold CV (same for every row).

## 1. Known-answer check — can the SAME embeddings predict a property of the *glucose*?
*(These should be high. If they are, but the cognitive scores stay near zero, then the low accuracy reflects the data rather than a broken pipeline.)*

| glucose target | Ridge R² | SVR R² | best |
|---|--:|--:|--:|
| mean glucose (mg/dL) | 0.042 | 0.042 | **0.042** |
| glucose SD | 0.443 | 0.462 | **0.462** |
| % time > 180 | 0.068 | -0.279 | **0.068** |

## 2. The shuffle test — do we do any better than scrambled scores?
*Randomly reassign the scores to the wrong sessions 200× (destroying any real relationship) and recompute grouped-CV R² each time. `p` = the fraction of scrambled runs that did at least as well as the real one. **p > 0.05 means scrambled scores do about as well as the real ones — i.e. no measurable relationship.*** Note the scrambled runs average about −0.05 rather than 0, because R² is measured against each test fold's own average.

| score | real R² | scrambled mean ± sd | scrambled 95th %ile | p-value |
|---|--:|--:|--:|--:|
| grids | -0.107 | -0.048 ± 0.015 | -0.025 | **1.000** |
| symbols | -0.096 | -0.052 ± 0.015 | -0.028 | **0.995** |
| prices | -0.069 | -0.050 ± 0.014 | -0.027 | **0.910** |

### Verdict

- ⚠️ Known-answer check is PARTIAL: the best glucose property reaches R²=0.462 (so the pipeline does extract real information), but not all of them clear 0.5. Absolute level is recovered poorly because Chronos-Bolt subtracts each series' own mean and divides by its own standard deviation before the encoder sees it. Do NOT report this as a clean pass — say which properties are recovered and which are not.
- ✅ Shuffle test: scrambled scores do about as well as the real ones for every score (p > 0.05), so there is no measurable relationship between the glucose and the score.
