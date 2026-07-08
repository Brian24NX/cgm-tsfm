# Head-to-head: representations for CGM → cognition

- **Data**: REAL
- **CV**: `group`  ·  **target-norm**: `center`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-07-07T22:05:48
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

> within-subject `center`: R² = fraction of WITHIN-subject variance explained (oracle centering — a characterization, not a new-subject predictor).

## grids_cognitive_score  (n=951 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | Ridge | -0.018±0.023 | 0.460±0.057 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.004±0.011 | 0.457±0.061 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=920 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | Ridge | -0.022±0.050 | 0.444±0.042 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.004±0.007 | 0.441±0.048 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=936 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.002±0.003 | 16.501±0.619 |
| Hand-crafted (43) + Ridge/SVR | SVR | -0.002±0.002 | 16.497±0.625 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
