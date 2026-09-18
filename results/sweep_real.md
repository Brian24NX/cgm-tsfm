# Chronos sweep results

- **Data**: REAL
- **Global target-norm** (checkpoint/window/pooling/pca axes): `none`
- **Generated**: 2026-09-05T19:48:22
- **Protocol**: subject-grouped nested CV; cell = best-model R² per target vs a mean-predictor (higher = better).

```
REAL data (target_norm=none for non-targetnorm axes):
CGMDataset: 916 sessions, 20 subjects
  window length (readings): min=24 median=24 max=24  (~2.0 h median)
  grids_cognitive_score: n=911 mean=0.471 std=0.499
  symbols_cognitive_score: n=882 mean=1.837 std=0.562
  prices_cognitive_score: n=897 mean=41.706 std=17.269
```
### Checkpoint sweep (window=2h uniform, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| chronos-bolt-tiny | 256 | -0.100 | -0.287 | -0.015 | -0.134 |
| chronos-bolt-mini | 384 | -0.123 | -0.295 | -0.015 | -0.144 |
| chronos-bolt-small | 512 | -0.135 | -0.290 | -0.015 | -0.147 |
| chronos-bolt-base | 768 | -0.152 | -0.282 | -0.016 | -0.150 |
| chronos-t5-small | 512 | -0.132 | -0.268 | -0.015 | -0.138 |

### Window sweep (model=chronos-bolt-small, pooling=mean, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| 15 min (3 readings) | 512 | -0.093 | -0.057 | -0.009 | -0.053 |
| 30 min (6 readings) | 512 | -0.076 | -0.065 | -0.018 | -0.053 |
| 60 min (12 readings) | 512 | -0.082 | -0.107 | -0.012 | -0.067 |
| 90 min (18 readings) | 512 | -0.076 | -0.196 | -0.013 | -0.095 |
| 120 min (24 readings) | 512 | -0.135 | -0.290 | -0.015 | -0.147 |
| 180 min (36 readings) | 512 | -0.066 | -0.378 | -0.066 | -0.170 |
| variable (old behaviour) | 512 | -0.114 | -0.052 | -0.012 | -0.059 |

### Pooling sweep (model=chronos-bolt-small, window=2h uniform, no PCA, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| mean | 512 | -0.135 | -0.290 | -0.015 | -0.147 |
| last | 512 | -0.111 | -0.290 | -0.017 | -0.139 |

### PCA sweep (model=chronos-bolt-small, window=2h uniform, pooling=mean, target_norm=none)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| no PCA (full-dim) | 512 | -0.135 | -0.290 | -0.015 | -0.147 |
| PCA=8 | 512 | -0.095 | -0.266 | -0.010 | -0.124 |
| PCA=16 | 512 | -0.100 | -0.263 | -0.016 | -0.126 |
| PCA=32 | 512 | -0.119 | -0.263 | -0.016 | -0.132 |
| PCA=64 | 512 | -0.129 | -0.264 | -0.016 | -0.136 |
| PCA=128 | 512 | -0.134 | -0.271 | -0.016 | -0.140 |

### Target-norm sweep (model=chronos-bolt-small, window=2h uniform, pooling=mean); center/zscore R² = within-subject variance explained (oracle centering)

| config | d | grids | symbols | prices | mean R² |
|---|---|---|---|---|---|
| none (raw score) | 512 | -0.135 | -0.290 | -0.015 | -0.147 |
| within-subj center | 512 | -0.042 | -0.045 | -0.001 | -0.029 |
| within-subj zscore | 512 | -0.034 | -0.049 | -0.018 | -0.034 |

