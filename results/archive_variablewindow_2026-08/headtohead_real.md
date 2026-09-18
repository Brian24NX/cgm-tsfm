# Head-to-head: representations for CGM → cognition

- **Data**: REAL
- **CV**: `group`  ·  **target-norm**: `none`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-08-01T17:44:58
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

## grids_cognitive_score  (n=951 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | Ridge | -0.114±0.119 | 0.510±0.055 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.065±0.084 | 0.500±0.061 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.313±0.114 | 0.554±0.055 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=920 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.052±0.049 | 0.559±0.071 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.038±0.050 | 0.555±0.075 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.412±0.216 | 0.640±0.046 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=936 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.012±0.010 | 17.236±0.570 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.009±0.009 | 17.210±0.558 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.360±0.069 | 19.995±1.140 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
