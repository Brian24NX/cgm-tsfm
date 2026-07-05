"""CGMRegressionModule — regression LightningModule for the single-channel arm.

Adapted from Ben's `FinetuneLightningModule`
(mod-actigraphy-advanced/src/models/modules.py:330-795). Key changes:

  * Regression is explicit, not inferred from a task-name string. Ben's
    `if task == "ptb_classification": ... else: <regression>` (modules.py:354-359)
    made *any* non-PTB task silently regress. Here `is_classification` is gone —
    this module only regresses.
  * General target normalization replaces the gestational-age-specific
    `labels/365.0` ... `preds*365.0` (modules.py:588-591). We z-score with the
    TRAIN-set mean/std of the target (computed once, passed in), so the loss is
    on an O(1) scale for any cognitive score, and predictions are de-normalized
    back to the score's native units for reporting.
  * Single channel: `_shared_step` reads `glucose` + `glucose_mask` (not
    activity/light/day_mask). Loss = MSE; metrics = MSE/RMSE/R²/MAE (Ben already
    had the regression metric block at modules.py:432-448 — we keep it and add MAE).
"""

from __future__ import annotations

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score


class CGMRegressionModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        target_mean: float,
        target_std: float,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        # normalization constants (train-set stats) — replaces the /365.0 magic
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(max(float(target_std), 1e-6)))
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss()

        base = {"mse": MeanSquaredError(), "rmse": MeanSquaredError(squared=False),
                "mae": MeanAbsoluteError(), "r2": R2Score()}
        self.val_metrics = MetricCollection({k: v.clone() for k, v in base.items()}, prefix="val/")
        self.test_metrics = MetricCollection({k: v.clone() for k, v in base.items()}, prefix="test/")

    def forward(self, glucose, glucose_mask=None):
        return self.model(glucose, glucose_mask)

    def _shared_step(self, batch, stage: str):
        glucose = batch["glucose"]
        glucose_mask = batch.get("glucose_mask")
        labels = batch["label"].float()

        valid = ~torch.isnan(labels)
        if not valid.any():
            return None
        glucose, labels = glucose[valid], labels[valid]
        if glucose_mask is not None:
            glucose_mask = glucose_mask[valid]

        logits = self(glucose, glucose_mask).squeeze(-1)          # (B,)
        labels_norm = (labels - self.target_mean) / self.target_std
        loss = self.loss_fn(logits, labels_norm)

        preds = logits * self.target_std + self.target_mean       # de-normalize
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        if stage in ("val", "test"):
            metrics = getattr(self, f"{stage}_metrics")
            metrics.update(preds, labels)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Only the head is trainable (encoder is frozen).
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
