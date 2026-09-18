# Chronos sweep results

- **Data**: REAL
- **Global target-norm** (checkpoint/window/pooling/pca axes): `none`
- **Generated**: 2026-08-01T18:20:17
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
| chronos-bolt-tiny | 256 | -0.086 | -0.053 | -0.014 | -0.051 |
| chronos-bolt-mini | 384 | -0.081 | -0.052 | -0.013 | -0.049 |
| chronos-bolt-small | 512 | -0.114 | -0.052 | -0.012 | -0.059 |
| chronos-bolt-base | 768 | -0.131 | -0.057 | -0.014 | -0.067 |
| chronos-t5-small | 512 | -0.120 | -0.066 | -0.013 | -0.067 |

### Window sweep (model=chronos-bolt-small, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| 24rd (~2.0h) | 512 | -0.130 | -0.089 | -0.012 | -0.077 |
| 30rd (~2.5h) | 512 | -0.138 | -0.077 | -0.013 | -0.076 |
| 36rd (~3.0h) | 512 | -0.132 | -0.068 | -0.013 | -0.071 |
| full | 512 | -0.114 | -0.052 | -0.012 | -0.059 |

### Pooling sweep (model=chronos-bolt-small, window=full, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| mean | 512 | -0.114 | -0.052 | -0.012 | -0.059 |
| last | 512 | -0.114 | -0.053 | -0.013 | -0.060 |

### PCA sweep (model=chronos-bolt-small, window=full, pooling=mean, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| no PCA (full-dim) | 512 | -0.114 | -0.052 | -0.012 | -0.059 |
| PCA=8 | 512 | -0.066 | -0.069 | -0.010 | -0.048 |
| PCA=16 | 512 | -0.072 | -0.052 | -0.010 | -0.045 |
| PCA=32 | 512 | -0.074 | -0.051 | -0.010 | -0.045 |
| PCA=64 | 512 | -0.076 | -0.063 | -0.012 | -0.050 |
| PCA=128 | 512 | -0.102 | -0.069 | -0.011 | -0.061 |

### Target-norm sweep (model=chronos-bolt-small, window=full, pooling=mean); center/zscore R² = within-subject variance explained (oracle centering)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| none (raw score) | 512 | -0.114 | -0.052 | -0.012 | -0.059 |
| within-subj center | 512 | -0.027 | -0.028 | -0.001 | -0.019 |
| within-subj zscore | 512 | -0.035 | -0.027 | -0.024 | -0.029 |

