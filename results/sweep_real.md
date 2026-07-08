# Chronos sweep results

- **Data**: REAL
- **Global target-norm** (checkpoint/window/pooling/pca axes): `none`
- **Generated**: 2026-07-07T22:10:48
- **Protocol**: subject-grouped nested CV; cell = best-model R² per target vs a mean-predictor (higher = better).

```
REAL data (target_norm=none for non-targetnorm axes):
CGMDataset: 956 sessions, 20 subjects
  window length (readings): min=3 median=36 max=288  (~3.0 h median)
  grids_cognitive_score: n=951 mean=0.478 std=0.500
  symbols_cognitive_score: n=920 mean=1.846 std=0.560
  prices_cognitive_score: n=936 mean=-41.592 std=17.170
```
### PCA sweep (model=chronos-bolt-small, window=full, pooling=mean, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| no PCA (full-dim) | 512 | -0.089 | -0.056 | -0.012 | -0.052 |
| PCA=8 | 512 | -0.074 | -0.052 | -0.009 | -0.045 |
| PCA=16 | 512 | -0.080 | -0.049 | -0.009 | -0.046 |
| PCA=32 | 512 | -0.075 | -0.051 | -0.010 | -0.045 |
| PCA=64 | 512 | -0.074 | -0.076 | -0.010 | -0.053 |
| PCA=128 | 512 | -0.089 | -0.053 | -0.011 | -0.051 |

