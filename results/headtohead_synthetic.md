# Head-to-head: representations for CGM → cognition

- **Data**: SYNTHETIC
- **CV**: `group`  ·  **target-norm**: `none`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-07-03T14:35:33
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

> ⚠️ **SYNTHETIC data** — harness check, not scientific results. Re-run with `--real`.

## grids_cognitive_score  (n=917 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.728±0.662 | 0.374±0.075 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.779±0.779 | 0.378±0.081 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -1.090±0.979 | 0.407±0.084 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=920 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.225±0.090 | 0.593±0.135 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.363±0.212 | 0.623±0.137 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.589±0.308 | 0.661±0.108 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=925 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.185±0.134 | 16.549±1.994 |
| Hand-crafted (43) + Ridge/SVR | SVR | -0.200±0.114 | 16.651±1.854 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.410±0.281 | 17.882±1.500 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
