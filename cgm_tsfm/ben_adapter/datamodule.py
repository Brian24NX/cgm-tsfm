"""Lightning DataModule for Arm B grouped-CV training.

Serves one CV fold's train/val/test as batches of `{"glucose", "label"}` — the
exact keys `CGMRegressionModule._shared_step` reads. "glucose" here is the
PRECOMPUTED Chronos embedding (the frozen encoder is run once, upstream, and
cached), so training touches only the trainable head.

The fold indices are produced by `train.py` with a `GroupKFold` on `subid` — the
*same* deterministic outer folds Arm A uses — so Arm B's test estimates are
directly comparable to Arm A's. Within each outer-train set, a grouped
`GroupShuffleSplit` carves out a validation set for early stopping (no subject
appears in more than one of train/val/test).
"""

from __future__ import annotations

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class _EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        # No "glucose_mask" key → CGMRegressionModule sees mask=None (fine: the
        # PassthroughEmbedder ignores it). Default collate stacks the dict values.
        return {"glucose": self.X[i], "label": self.y[i]}


class CGMEmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self._tr = _EmbeddingDataset(X_train, y_train)
        self._va = _EmbeddingDataset(X_val, y_val)
        self._te = _EmbeddingDataset(X_test, y_test)
        self.batch_size = batch_size
        self.num_workers = num_workers

    # train-set target stats — CGMRegressionModule z-scores with these (train-only,
    # so no leakage of val/test scale into training).
    @property
    def target_mean(self) -> float:
        return float(self._tr.y.mean())

    @property
    def target_std(self) -> float:
        return float(self._tr.y.std(unbiased=False))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._tr, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._va, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._te, batch_size=self.batch_size, num_workers=self.num_workers)
