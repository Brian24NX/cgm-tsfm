"""Frozen TSFM encoders: turn glucose windows into fixed-size embeddings.

The recipe (verified against the ICML'25 `representations-in-tsfms` repo and the
installed `chronos` 2.3.0 API):

    windows (list of 1-D float arrays, variable length)
      -> BaseChronosPipeline.embed(context)          # (B, num_patches+1, d_model)
      -> mean-pool over the patch/token axis          # (B, d_model)

Chronos mean-scales each series internally and is NaN-tolerant (missing readings
map to a mask token), so we feed raw mg/dL values with no external normalization
and no gap-filling. `BaseChronosPipeline` auto-selects Chronos-Bolt or Chronos-T5
from the checkpoint name, so the same code path serves both model families.

A `MockEncoder` is included so the full pipeline runs offline with no model
download (deterministic hand-crafted summary features standing in for a TSFM).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config as C


def _windows_hash(windows: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in windows:
        h.update(np.asarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


class MockEncoder:
    """Deterministic stand-in for a TSFM (no torch / no download required).

    Produces a small fixed-size vector of standardized summary statistics per
    window. NOT a foundation model — only for offline plumbing tests. The real
    scientific comparison uses ChronosEncoder.
    """

    def __init__(self, d_model: int = 16):
        self.d_model = d_model

    def get_embedding_dim(self) -> int:
        return self.d_model

    def encode(self, windows: list[np.ndarray]) -> np.ndarray:
        feats = []
        for w in windows:
            w = np.asarray(w, dtype=float)
            d = np.diff(w) if w.size > 1 else np.array([0.0])
            f = [
                w.mean(), w.std(), w.min(), w.max(), np.median(w),
                np.percentile(w, 25), np.percentile(w, 75),
                (w < 70).mean(), (w > 180).mean(), (w > 250).mean(),
                d.mean(), d.std(), np.abs(d).mean(), w.size,
                (w[-1] - w[0]), np.polyfit(np.arange(w.size), w, 1)[0] if w.size > 1 else 0.0,
            ]
            feats.append(f[: self.d_model] + [0.0] * max(0, self.d_model - len(f)))
        return np.asarray(feats, dtype=np.float32)


class ChronosEncoder:
    """Frozen Amazon Chronos encoder (Bolt or T5) -> mean-pooled embeddings.

    Wraps `chronos.BaseChronosPipeline` and its `.embed()` method. The encoder is
    used only for inference (no gradients), matching the "raw-data / frozen-TSFM"
    design. This is a cleaner reimplementation of Ben's `ChronosEncoder`
    (mod-actigraphy-advanced/src/models/pretrained_models.py) using the
    higher-level pipeline API + the ICML mean-over-tokens pooling.
    """

    def __init__(self, cfg: C.EncoderConfig | None = None):
        self.cfg = cfg or C.EncoderConfig()
        import torch
        from chronos import BaseChronosPipeline

        self._torch = torch
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(self.cfg.torch_dtype, torch.float32)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.cfg.pretrained_model,
            device_map=self.cfg.device,
            torch_dtype=dtype,
        )
        # d_model is discovered on the first encode() call (varies by checkpoint).
        self._d_model: int | None = None

    def get_embedding_dim(self) -> int:
        if self._d_model is None:
            raise RuntimeError("call encode() once before get_embedding_dim()")
        return self._d_model

    def encode(self, windows: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out = []
        bs = self.cfg.batch_size
        for i in range(0, len(windows), bs):
            batch = [torch.tensor(np.asarray(w, dtype=np.float32)) for w in windows[i:i + bs]]
            with torch.no_grad():
                emb, _ = self.pipeline.embed(batch)   # (B, num_patches+1, d_model)
                if self.cfg.pooling == "last":
                    pooled = emb[:, -1, :]
                else:
                    pooled = emb.mean(dim=1)          # mean over patch/token axis
            out.append(pooled.to(torch.float32).cpu().numpy())
        arr = np.concatenate(out, axis=0)
        self._d_model = arr.shape[1]
        return arr


def build_encoder(cfg: C.EncoderConfig | None = None):
    cfg = cfg or C.EncoderConfig()
    if cfg.encoder_type == "mock":
        return MockEncoder()
    if cfg.encoder_type == "chronos":
        return ChronosEncoder(cfg)
    raise ValueError(f"unknown encoder_type: {cfg.encoder_type}")


def extract_embeddings(
    windows: list[np.ndarray],
    cfg: C.EncoderConfig | None = None,
    cache_dir: Path | str | None = C.CACHE_DIR,
    use_cache: bool = True,
) -> np.ndarray:
    """Extract (N, d_model) embeddings, caching to .npy keyed by (model, data).

    The Chronos forward pass is the expensive step; cache once and reuse across
    all three targets and every downstream regressor.
    """
    cfg = cfg or C.EncoderConfig()
    key = f"{cfg.encoder_type}__{cfg.pretrained_model.replace('/', '_')}__{cfg.pooling}__{_windows_hash(windows)}"
    cache_path = Path(cache_dir) / f"{key}.npy" if cache_dir else None

    if use_cache and cache_path and cache_path.exists():
        return np.load(cache_path)

    encoder = build_encoder(cfg)
    emb = encoder.encode(windows)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb
