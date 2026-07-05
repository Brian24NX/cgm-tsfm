"""Arm B — grouped-CV training of the trainable regression head on frozen embeddings.

`run_arm_b(dataset, embeddings, ...)` returns results in the SAME shape as
`regression.run_arm_a` ({target: {"n","n_subjects","rows"}}), so it drops into
`format_results` and the head-to-head unchanged.

CV design (documented so it's comparable to Arm A, not silently different):
  * OUTER: `GroupKFold(5)` on `subid` — the *same deterministic folds* Arm A uses,
    so test estimates are apples-to-apples.
  * Within each outer-train set: a grouped `GroupShuffleSplit` (20%) holds out a
    validation set for EARLY STOPPING on val loss. Unlike Arm A's inner
    `GridSearchCV`, Arm B uses a fixed head architecture (no hyperparameter grid) —
    early stopping is the regularizer. This keeps 15 net-trainings tractable on CPU.
  * Labels are z-scored with TRAIN-set stats only (in CGMRegressionModule).

Run standalone:
    python -m cgm_tsfm.ben_adapter.train --encoder chronos
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from ..data import within_subject_normalize
from ..regression import FoldResult, ModelResult
from .datamodule import CGMEmbeddingDataModule
from .lightning_module import CGMRegressionModule
from .model import FoundationModelRegressor, PassthroughEmbedder

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)


def _train_one_fold(
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    *, max_epochs, patience, batch_size, head_hidden_dim, head_dropout, lr, accelerator, seed,
) -> FoldResult:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping

    pl.seed_everything(seed, workers=True)
    dm = CGMEmbeddingDataModule(X_tr, y_tr, X_va, y_va, X_te, y_te, batch_size=batch_size)
    model = FoundationModelRegressor(
        PassthroughEmbedder(X_tr.shape[1]),
        num_targets=1, head_hidden_dim=head_hidden_dim, head_dropout=head_dropout,
    )
    lm = CGMRegressionModule(model, target_mean=dm.target_mean, target_std=dm.target_std, lr=lr)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices=1,
        callbacks=[EarlyStopping(monitor="val/loss", patience=patience, mode="min")],
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1,
    )
    trainer.fit(lm, dm)
    res = trainer.test(lm, dm, verbose=False)[0]
    return FoldResult(rmse=res["test/rmse"], mae=res["test/mae"], r2=res["test/r2"])


def run_arm_b(
    dataset,
    embeddings: np.ndarray,
    targets: list[str] | None = None,
    outer_splits: int = 5,
    val_frac: float = 0.2,
    max_epochs: int = 100,
    patience: int = 8,
    batch_size: int = 64,
    head_hidden_dim: int = 256,
    head_dropout: float = 0.1,
    lr: float = 1e-3,
    accelerator: str = "cpu",
    seed: int = 42,
    target_norm: str = "none",
) -> dict[str, dict]:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    targets = targets or list(dataset.targets.keys())
    results: dict[str, dict] = {}

    for target in targets:
        mask = dataset.target_mask(target)
        X, y, g = embeddings[mask], dataset.targets[target][mask], dataset.subjects[mask]
        y = within_subject_normalize(y, g, target_norm)   # no-op when "none"

        res = ModelResult(name="Chronos+MLPhead")
        outer = GroupKFold(n_splits=outer_splits)
        for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y, g)):
            # grouped val split out of the outer-train subjects (early stopping)
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr_local, va_local = next(gss.split(X[train_idx], y[train_idx], g[train_idx]))
            tr_abs, va_abs = train_idx[tr_local], train_idx[va_local]
            assert set(g[tr_abs]).isdisjoint(g[va_abs]) and set(g[tr_abs]).isdisjoint(g[test_idx]) \
                and set(g[va_abs]).isdisjoint(g[test_idx]), "subject leakage across train/val/test!"

            res.folds.append(_train_one_fold(
                X[tr_abs], y[tr_abs], X[va_abs], y[va_abs], X[test_idx], y[test_idx],
                max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                head_hidden_dim=head_hidden_dim, head_dropout=head_dropout, lr=lr,
                accelerator=accelerator, seed=seed + fold_idx,
            ))

        results[target] = {
            "n": int(mask.sum()), "n_subjects": int(len(np.unique(g))),
            "rows": [res.summary_row()],
        }
    return results


def main() -> None:
    from .. import config as C
    from ..data import generate_synthetic_data, load_real_data
    from ..encoders import extract_embeddings
    from ..regression import format_results

    ap = argparse.ArgumentParser(description="Arm B — trainable head, grouped-CV training")
    ap.add_argument("--encoder", choices=["mock", "chronos"], default="chronos")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-readings", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--target-norm", choices=["none", "center", "zscore"], default="none",
                    help="within-subject target normalization (see data.within_subject_normalize)")
    args = ap.parse_args()

    window = C.WindowConfig(max_readings=args.max_readings)
    ds = load_real_data(window=window) if args.real else generate_synthetic_data(window=window)
    print(ds.summary())

    enc_cfg = C.EncoderConfig(encoder_type=args.encoder, pretrained_model=args.model, device=args.device)
    print(f"\nExtracting {args.encoder} embeddings (frozen; run once, then train the head) ...")
    emb = extract_embeddings(ds.windows, enc_cfg)
    print(f"  embeddings: {emb.shape}")

    print(f"\nTraining Arm B (trainable MLP head) with grouped 5-fold CV  (target_norm={args.target_norm}) ...")
    results = run_arm_b(ds, emb, max_epochs=args.max_epochs, accelerator=args.device, target_norm=args.target_norm)
    print(format_results(results, cv_scheme="group", title="Arm B (Chronos + trainable head)"))


if __name__ == "__main__":
    main()
