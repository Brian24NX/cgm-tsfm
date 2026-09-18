# Chronos sweep results

- **Data**: REAL
- **Global target-norm** (checkpoint/window/pooling/pca axes): `none`
- **Generated**: 2026-07-21T00:52:02
- **Protocol**: subject-grouped nested CV; cell = best-model R² per target vs a mean-predictor (higher = better).

```
REAL data (target_norm=none for non-targetnorm axes):
CGMDataset: 956 sessions, 20 subjects
  window length (readings): min=3 median=36 max=288  (~3.0 h median)
  grids_cognitive_score: n=951 mean=0.478 std=0.500
  symbols_cognitive_score: n=920 mean=1.846 std=0.560
  prices_cognitive_score: n=936 mean=41.592 std=17.170
```
### Checkpoint sweep (window=full, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| chronos-bolt-tiny | 256 | -0.070 | -0.063 | -0.012 | -0.049 |
| chronos-bolt-mini | 384 | -0.091 | -0.051 | -0.013 | -0.052 |
| chronos-bolt-small | 512 | -0.089 | -0.056 | -0.012 | -0.052 |
| chronos-bolt-base | 768 | -0.117 | -0.062 | -0.012 | -0.063 |
| chronos-t5-small | 512 | -0.093 | -0.067 | -0.012 | -0.057 |

### Window sweep (model=chronos-bolt-small, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| 24rd (~2.0h) | 512 | -0.126 | -0.089 | -0.012 | -0.076 |
| 30rd (~2.5h) | 512 | -0.135 | -0.076 | -0.013 | -0.075 |
| 36rd (~3.0h) | 512 | -0.120 | -0.072 | -0.013 | -0.068 |
| full | 512 | -0.089 | -0.056 | -0.012 | -0.052 |

### Pooling sweep (model=chronos-bolt-small, window=full, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| mean | 512 | -0.089 | -0.056 | -0.012 | -0.052 |
| last | 512 | -0.114 | -0.053 | -0.013 | -0.060 |

### PCA sweep (model=chronos-bolt-small, window=full, pooling=mean, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| no PCA (full-dim) | 512 | -0.089 | -0.056 | -0.012 | -0.052 |
| PCA=8 | 512 | -0.074 | -0.052 | -0.009 | -0.045 |
| PCA=16 | 512 | -0.080 | -0.049 | -0.009 | -0.046 |
| PCA=32 | 512 | -0.075 | -0.051 | -0.010 | -0.045 |
| PCA=64 | 512 | -0.074 | -0.076 | -0.010 | -0.053 |
| PCA=128 | 512 | -0.089 | -0.053 | -0.011 | -0.051 |

### Target-norm sweep (model=chronos-bolt-small, window=full, pooling=mean); center/zscore R² = within-subject variance explained (oracle centering)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| none (raw score) | 512 | -0.089 | -0.056 | -0.012 | -0.052 |
| within-subj center | 512 | -0.018 | -0.022 | -0.002 | -0.014 |
| within-subj zscore | 512 | -0.011 | -0.011 | -0.026 | -0.016 |

