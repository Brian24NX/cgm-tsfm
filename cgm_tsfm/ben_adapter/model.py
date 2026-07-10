"""FoundationModelRegressor — single-channel CGM regression head on a frozen TSFM.

This is the direct adaptation of Ben's `FoundationModelClassifier`
(mod-actigraphy-advanced/src/models/pretrained_models.py:270-704) requested by
The advisor: "adapt/generalize Ben's code to accept one channel of CGM."

What changed vs Ben's model (full mapping in README.md):
  * num_channels 2 (activity+light) -> 1 (glucose). The hardcoded
    `num_channels = 2` and the two named tensor args are gone.
  * forward(activity, light, activity_mask, light_mask, day_mask)
       -> forward(glucose, glucose_mask).
  * The across-days causal transformer is dropped: each CGM sample is ONE
    pre-test window -> ONE score (D=1), so there is no day sequence to attend
    over. We keep Ben's frozen-encoder + MLP-head structure (his `classifier`
    Sequential, pretrained_models.py:384-390) and set the output width to the
    number of regression targets (1, per the advisor's single-target decision).
  * Ben's `ChronosEncoder` (Bolt `.encode()`) is replaced by the higher-level,
    version-robust `BaseChronosPipeline.embed()` + mean-pool recipe (see
    cgm_tsfm/encoders.py). The encoder stays frozen (no gradients).

The frozen encoder is injected, so this module is testable offline with a mock
encoder (no download, no training) — see `MockTorchEmbedder`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _valid_series(glucose: torch.Tensor, mask: torch.Tensor | None) -> list[torch.Tensor]:
    """Split a padded (B, T) batch into a list of per-sample valid 1-D series.

    Chronos left-pads internally, so we hand it the raw valid readings and let
    it tokenize. Fully-empty rows fall back to a single zero so embed() is safe.
    """
    out = []
    for i in range(glucose.shape[0]):
        row = glucose[i]
        if mask is not None:
            row = row[mask[i] > 0]
        row = row[~torch.isnan(row)]
        out.append(row if row.numel() > 0 else torch.zeros(1, device=glucose.device))
    return out


class ChronosTorchEmbedder(nn.Module):
    """Frozen Chronos encoder returning mean-pooled (B, d_model) torch embeddings."""

    def __init__(self, pretrained_model: str = "amazon/chronos-bolt-small",
                 device: str = "cpu", pooling: str = "mean"):
        super().__init__()
        from chronos import BaseChronosPipeline
        self.pipeline = BaseChronosPipeline.from_pretrained(
            pretrained_model, device_map=device, torch_dtype=torch.float32)
        self.pooling = pooling
        self._d_model: int | None = None

    @property
    def embedding_dim(self) -> int:
        if self._d_model is None:
            # probe with a dummy series
            with torch.no_grad():
                emb, _ = self.pipeline.embed([torch.zeros(8)])
            self._d_model = emb.shape[-1]
        return self._d_model

    @torch.no_grad()
    def forward(self, glucose: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        series = _valid_series(glucose, mask)
        emb, _ = self.pipeline.embed(series)            # (B, P+1, d_model)
        pooled = emb[:, -1, :] if self.pooling == "last" else emb.mean(dim=1)
        self._d_model = pooled.shape[-1]
        return pooled.to(torch.float32)


class MockTorchEmbedder(nn.Module):
    """Deterministic frozen stand-in (no download) for smoke tests."""

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.d_model = d_model

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    @torch.no_grad()
    def forward(self, glucose: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        feats = []
        for row, m in zip(glucose, mask if mask is not None else [None] * len(glucose)):
            v = row[m > 0] if m is not None else row
            v = v[~torch.isnan(v)]
            if v.numel() == 0:
                v = torch.zeros(1)
            d = torch.diff(v) if v.numel() > 1 else torch.zeros(1)
            f = torch.tensor([
                v.mean(), v.std(unbiased=False), v.min(), v.max(), v.median(),
                (v < 70).float().mean(), (v > 180).float().mean(), (v > 250).float().mean(),
                d.mean(), d.std(unbiased=False), d.abs().mean(), float(v.numel()),
                v[-1] - v[0], v.quantile(0.25), v.quantile(0.75), v.float().var(unbiased=False),
            ])
            feats.append(f[: self.d_model])
        return torch.stack(feats).to(torch.float32)


class PassthroughEmbedder(nn.Module):
    """A no-op "encoder" for training the head on PRECOMPUTED embeddings.

    Since the Chronos encoder is frozen, we extract embeddings once (cached) and
    train only the head — much faster than re-encoding every epoch, and identical
    in result. `forward` just returns the already-embedded batch, so the same
    `FoundationModelRegressor` + `CGMRegressionModule` work whether fed raw glucose
    windows (real encoder) or precomputed embeddings (this)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return embeddings


class FoundationModelRegressor(nn.Module):
    """Frozen TSFM encoder + trainable MLP regression head (single channel)."""

    def __init__(
        self,
        encoder: nn.Module,
        num_targets: int = 1,           # single-target per the advisor (was num_classes)
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder          # frozen; not updated by the optimizer
        for p in self.encoder.parameters():
            p.requires_grad = False
        d = encoder.embedding_dim
        # Same head shape as Ben's `classifier` (pretrained_models.py:384-390),
        # output width = num_targets instead of num_classes.
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_targets),
        )

    def forward(self, glucose: torch.Tensor, glucose_mask: torch.Tensor | None = None) -> torch.Tensor:
        """glucose: (B, T) readings; glucose_mask: (B, T) 1=valid. Returns (B, num_targets)."""
        emb = self.encoder(glucose, glucose_mask)       # (B, d_model), no grad
        return self.head(emb)


def collate_windows(windows: list[np.ndarray], device: str = "cpu"):
    """Right-pad a list of variable-length windows into (B, T) + mask (B, T)."""
    T = max(w.size for w in windows)
    B = len(windows)
    g = torch.zeros(B, T, dtype=torch.float32, device=device)
    m = torch.zeros(B, T, dtype=torch.float32, device=device)
    for i, w in enumerate(windows):
        w = torch.tensor(np.asarray(w, dtype=np.float32), device=device)
        g[i, : w.numel()] = w
        m[i, : w.numel()] = 1.0
    return g, m
