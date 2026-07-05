# Chronos sweep results

- **Data**: SYNTHETIC
- **Global target-norm** (checkpoint/window/pooling/pca axes): `none`
- **Generated**: 2026-07-03T14:30:43
- **Protocol**: subject-grouped nested CV; cell = best-model R² per target vs a mean-predictor (higher = better).

```
SYNTHETIC data (target_norm=none for non-targetnorm axes):
CGMDataset: 938 sessions, 20 subjects
  window length (readings): min=20 median=45 max=70  (~3.8 h median)
  grids_cognitive_score: n=917 mean=0.304 std=0.349
  symbols_cognitive_score: n=920 mean=1.800 std=0.586
  prices_cognitive_score: n=925 mean=-40.973 std=16.220
```

> ⚠️ **SYNTHETIC data** — these numbers exercise the sweep harness only; they are not scientific results. Re-run with `--real` for the real comparison.
### Checkpoint sweep (window=full, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| chronos-bolt-tiny | 256 | -0.736 | -0.220 | -0.185 | -0.380 |
| chronos-bolt-mini | 384 | -0.754 | -0.225 | -0.196 | -0.392 |
| chronos-bolt-small | 512 | -0.728 | -0.225 | -0.185 | -0.379 |
| chronos-bolt-base | 768 | -0.731 | -0.229 | -0.186 | -0.382 |
| chronos-t5-small | 512 | -0.824 | -0.347 | -0.201 | -0.458 |

### Window sweep (model=chronos-bolt-small, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| 24rd (~2.0h) | 512 | -0.734 | -0.250 | -0.188 | -0.391 |
| 30rd (~2.5h) | 512 | -0.776 | -0.244 | -0.189 | -0.403 |
| 36rd (~3.0h) | 512 | -0.776 | -0.240 | -0.187 | -0.401 |
| full | 512 | -0.728 | -0.225 | -0.185 | -0.379 |

### Pooling sweep (model=chronos-bolt-small, window=full, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| mean | 512 | -0.728 | -0.225 | -0.185 | -0.379 |
| last | 512 | -0.741 | -0.240 | -0.186 | -0.389 |

### PCA sweep (model=chronos-bolt-small, window=full, pooling=mean, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| no PCA (full-dim) | 512 | -0.728 | -0.225 | -0.185 | -0.379 |
| PCA=8 | 512 | -0.674 | -0.186 | -0.188 | -0.349 |
| PCA=16 | 512 | -0.674 | -0.185 | -0.186 | -0.348 |
| PCA=32 | 512 | -0.672 | -0.184 | -0.186 | -0.347 |
| PCA=64 | 512 | -0.671 | -0.193 | -0.187 | -0.350 |
| PCA=128 | 512 | -0.698 | -0.212 | -0.187 | -0.365 |

### Target-norm sweep (model=chronos-bolt-small, window=full, pooling=mean); center/zscore R² = within-subject variance explained (oracle centering)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| none (raw score) | 512 | -0.728 | -0.225 | -0.185 | -0.379 |
| within-subj center | 512 | -0.036 | -0.046 | -0.003 | -0.029 |
| within-subj zscore | 512 | -0.012 | -0.021 | -0.009 | -0.014 |

