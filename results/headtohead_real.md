# Head-to-head: representations for CGM → cognition

- **Data**: REAL
- **CV**: `group`  ·  **target-norm**: `none`  ·  **PCA**: `none`
- **Encoder**: `amazon/chronos-bolt-small`
- **Generated**: 2026-07-07T21:54:40
- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).

## grids_cognitive_score  (n=951 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | Ridge | -0.089±0.123 | 0.505±0.059 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.065±0.084 | 0.500±0.061 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.218±0.102 | 0.534±0.055 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## symbols_cognitive_score  (n=920 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.056±0.060 | 0.559±0.070 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.038±0.050 | 0.555±0.075 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.281±0.141 | 0.612±0.054 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation

## prices_cognitive_score  (n=936 sessions, 20 subjects)

| representation | best model | R² | RMSE |
|---|---|---|---|
| Chronos (512d) + Ridge/SVR | SVR | -0.012±0.010 | 17.236±0.572 |
| Hand-crafted (43) + Ridge/SVR | Ridge | -0.009±0.009 | 17.210±0.558 |
| Chronos (512d) + MLP head | Chronos+MLPhead | -0.168±0.051 | 18.532±1.010 |

**Verdict:** all ≈ baseline — no generalizable signal from any representation
