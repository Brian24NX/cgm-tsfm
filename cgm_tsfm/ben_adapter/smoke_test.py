"""Offline smoke test for Arm B: verify the single-channel regressor + Lightning
module forward/backward correctly (mock embedder, tiny synthetic batch, CPU).

    python -m cgm_tsfm.ben_adapter.smoke_test
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import generate_synthetic_data
from .lightning_module import CGMRegressionModule
from .model import FoundationModelRegressor, MockTorchEmbedder, collate_windows


def main() -> None:
    torch.manual_seed(0)
    ds = generate_synthetic_data(n_subjects=4, sessions_per_subject=(10, 15))
    target = "symbols_cognitive_score"
    y = ds.targets[target]
    mask = ~np.isnan(y)
    windows = [w for w, keep in zip(ds.windows, mask) if keep]
    y = y[mask]

    g, m = collate_windows(windows)
    labels = torch.tensor(y, dtype=torch.float32)
    print(f"batch: glucose={tuple(g.shape)} mask={tuple(m.shape)} labels={tuple(labels.shape)}")

    model = FoundationModelRegressor(MockTorchEmbedder(d_model=16), num_targets=1)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"model: {n_trainable} trainable params, {n_frozen} frozen (encoder)")

    out = model(g, m)
    assert out.shape == (len(windows), 1), out.shape
    print(f"forward OK -> {tuple(out.shape)}")

    lm = CGMRegressionModule(model, target_mean=float(y.mean()), target_std=float(y.std()))
    opt = lm.configure_optimizers()
    batch = {"glucose": g, "glucose_mask": m, "label": labels}

    losses = []
    for step in range(30):
        opt.zero_grad()
        loss = lm._shared_step(batch, "train")
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"train loss: {losses[0]:.4f} -> {losses[-1]:.4f}  ({'DECREASED' if losses[-1] < losses[0] else 'no change'})")
    assert losses[-1] < losses[0], "head did not learn on a trivial batch"
    print("\nArm B smoke test PASSED — model + LightningModule are correctly wired.")


if __name__ == "__main__":
    main()
