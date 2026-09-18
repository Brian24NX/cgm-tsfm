# Head-to-head: representations for CGM → cognition

- **Data**: REAL
- **CV**: `group`  ·  **target-norm**: `none`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-09-05T19:14:21
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

## grids_cognitive_score  (n=911 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | Ridge | -0.135±0.033 | 0.518±0.048 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.088±0.039 | 0.508±0.050 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.329±0.128 | 0.559±0.048 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=882 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.290±0.266 | 0.581±0.132 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.257±0.316 | 0.572±0.133 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.581±0.406 | 0.631±0.115 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=897 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.015±0.019 | 17.326±0.518 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.012±0.013 | 17.301±0.552 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.170±0.090 | 18.590±0.952 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
