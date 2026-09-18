# Rigor checks — positive control + permutation test

- **Data**: REAL  ·  **Encoder**: `amazon/chronos-bolt-small`  ·  **Permutations**: 200
- **Generated**: 2026-07-21T03:41:36
- Model: `StandardScaler → PCA(32) → Ridge(alpha=10)`, subject-grouped 5-fold CV (same for every row).

## 1. Positive control — can the SAME embeddings predict a *glucose* property?
*(Sanity: these should be HIGH. If they are but cognition ≈ 0, the null is real — not a broken pipeline.)*

| glucose target | Ridge R² | SVR R² | best |
|---|--:|--:|--:|
| mean glucose (mg/dL) | 0.060 | 0.070 | **0.070** |
| glucose SD | 0.431 | 0.462 | **0.462** |
| % time > 180 | 0.075 | -0.245 | **0.075** |

## 2. Permutation test — is the cognition R² better than chance?
*Shuffle each score 200× and recompute grouped-CV R². `p` = fraction of shuffles ≥ the real R². **p > 0.05 ⇒ the real R² is within the random-chance range ⇒ no signal beyond chance.***

| score | real R² | chance mean ± sd | chance 95th %ile | p-value |
|---|--:|--:|--:|--:|
| grids | -0.106 | -0.049 ± 0.015 | -0.027 | **1.000** |
| symbols | -0.127 | -0.052 ± 0.016 | -0.028 | **1.000** |
| prices | -0.085 | -0.051 ± 0.014 | -0.025 | **0.995** |

### Verdict

- ✅ Permutation test: cognition R² is NOT above chance (p>0.05 for every score) → no signal beyond chance.
