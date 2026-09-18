# Head-to-head: representations for CGM → cognition

- **Data**: REAL
- **CV**: `group`  ·  **target-norm**: `center`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-08-01T17:45:49
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

> within-subject `center`: R² = fraction of WITHIN-subject variance explained (oracle centering — a characterization, not a new-subject predictor).

## grids_cognitive_score  (n=951 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.027±0.022 | 0.462±0.058 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.004±0.011 | 0.457±0.061 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=920 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.028±0.017 | 0.446±0.050 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.004±0.007 | 0.441±0.048 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=936 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.001±0.002 | 16.495±0.616 |
| Hand-crafted (43) + Ridge/SVR | SVR | -0.002±0.002 | 16.497±0.625 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
