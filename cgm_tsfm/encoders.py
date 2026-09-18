"""Frozen TSFM encoders: turn glucose windows into fixed-size embeddings.

The recipe (verified against the ICML'25 `representations-in-tsfms` repo and the
installed `chronos` 2.3.1 API):

    windows (list of 1-D float arrays, variable length)
      -> BaseChronosPipeline.embed(context)          # (B, num_patches+1, d_model)
      -> pool over the patch/token axis               # (B, d_model)

Chronos normalizes each series internally and is NaN-tolerant, so we feed raw
mg/dL values with no external normalization and no gap-filling.
`BaseChronosPipeline` auto-selects Chronos-Bolt or Chronos-T5 from the checkpoint
name, so the same code path serves both model families.

⚠️ NOTE on Chronos-Bolt's internal normalization: it applies *instance
normalization* — `(x - mean) / population_std` per series (chronos_bolt.py
`InstanceNorm`, lines 95-134) — i.e. it removes BOTH the level and the amplitude.
(Chronos-T5 instead does mean-absolute scaling with no centering.) `embed()`
returns those two removed numbers as its second value; we currently discard them,
which is why the embedding is largely blind to absolute glucose level.

------------------------------------------------------------------------------
BATCH-INVARIANCE (bug fixed 2026-07-30 — read this before changing `encode`)
------------------------------------------------------------------------------
`pipeline.embed()` left-pads every series in a batch with NaN out to the longest
one, so the returned token axis is sized by the LONGEST member of the batch.
The previous implementation pooled with a plain `emb.mean(dim=1)` over *all*
token positions, which averaged in the positions produced by pure padding.
Consequence: a session's embedding depended on which other sessions happened to
share its batch. Measured on the real data (956 sessions, batch_size=32):
861/956 windows (90%) had cosine similarity < 0.9 against their correctly-pooled
vector, median 0.37; a 13-reading window batched beside a 288-reading window
came out at cosine 0.22 versus the same window embedded alone.

The fix here is to BUCKET windows by their token count so every batch is
padding-homogeneous. Because a window of length T yields
`ceil(min(T, context_length) / patch_size)` patches whether it is alone or
batched with same-token-count peers, bucketing makes the result *bit-identical*
to encoding each window on its own, while keeping the speed of batching. An
assertion checks the returned token count against the bucket key, so any
geometry surprise fails loudly instead of silently corrupting features.

The `[REG]`/EOS summary token (index -1) was already batch-invariant and is
available as `pooling="reg"` (alias `"last"` — note this selects the summary
token, NOT the most recent patch).

Re-running the full evaluation with corrected embeddings did NOT change the
scientific conclusion; see `results/README.md`.

A `MockEncoder` is included so the full pipeline runs offline with no model
download (deterministic hand-crafted summary features standing in for a TSFM).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C

# Bumped whenever the embedding recipe changes, so stale .npy caches computed by
# an older recipe can never be silently reused. v2 = batch-invariant pooling.
EMBEDDING_RECIPE_VERSION = "v2"


def _windows_hash(windows: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for w in windows:
        h.update(np.asarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Token geometry — shared by ChronosEncoder (Arm A) and ChronosTorchEmbedder
# (Arm B) so the batch-invariance fix lives in exactly one place.
# ---------------------------------------------------------------------------
def probe_token_geometry(pipeline, torch_mod) -> tuple[int, int, int]:
    """Return `(patch_size, extra_tokens, context_length)` for a Chronos pipeline.

    `patch_size` is *probed* rather than read from the config, so the same code
    serves Chronos-Bolt (16 readings per patch) and Chronos-T5 (one token per
    reading → patch_size 1). `extra_tokens` counts the trailing summary token
    ([REG] for Bolt, EOS for T5).
    """
    def n_tok(length: int) -> int:
        with torch_mod.no_grad():
            emb, _ = pipeline.embed([torch_mod.zeros(length)])
        return int(emb.shape[1])

    base = n_tok(1)
    patch = 1
    for L in (2, 4, 8, 16, 32, 64):
        if n_tok(L) == base:
            patch = L
        else:
            break
    extra = base - 1                      # tokens beyond the single patch

    ctx = None
    for get in (
        lambda: pipeline.model.config.chronos_config["context_length"],   # Bolt
        lambda: pipeline.tokenizer.config.context_length,                 # T5
        lambda: pipeline.model.config.context_length,
    ):
        try:
            ctx = int(get())
            break
        except Exception:
            continue
    # If no context length is discoverable, assume no truncation. Our longest
    # window is 288 readings, far below both 2048 (Bolt) and 512 (T5), so this
    # never binds here — and the assertion in the encoders catches it if it did.
    return patch, extra, ctx or 10**9


def tokens_for(length: int, geom: tuple[int, int, int]) -> int:
    """Token count `embed()` returns for a *single* series of this length."""
    patch, extra, ctx = geom
    return math.ceil(min(int(length), ctx) / patch) + extra


def bucket_by_tokens(lengths, geom: tuple[int, int, int]) -> dict[int, list[int]]:
    """Group index positions by token count, so batches are padding-homogeneous."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, L in enumerate(lengths):
        buckets[tokens_for(L, geom)].append(i)
    return buckets


def pool_tokens(emb, pooling: str):
    """Pool a (B, tokens, d_model) tensor down to (B, d_model).

    Only valid when every row of `emb` has the same *real* token count — i.e.
    when the batch was built with `bucket_by_tokens`. See the module docstring.
    """
    if pooling in ("reg", "last"):
        return emb[:, -1, :]                       # [REG]/EOS summary token
    if pooling == "mean_patches":
        return emb[:, :-1, :].mean(dim=1) if emb.shape[1] > 1 else emb[:, 0, :]
    if pooling == "mean":
        return emb.mean(dim=1)
    raise ValueError(f"unknown pooling: {pooling!r}")


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
        self._geom: tuple[int, int, int] | None = None

    def get_embedding_dim(self) -> int:
        if self._d_model is None:
            raise RuntimeError("call encode() once before get_embedding_dim()")
        return self._d_model

    def geometry(self) -> tuple[int, int, int]:
        """(patch_size, extra_tokens, context_length), probed once and cached."""
        if self._geom is None:
            self._geom = probe_token_geometry(self.pipeline, self._torch)
        return self._geom

    def encode(self, windows: list[np.ndarray]) -> np.ndarray:
        """Batch-invariant embedding extraction.

        Windows are grouped by token count so no window is ever padded to match a
        longer one. The result is identical to encoding each window alone.
        """
        torch = self._torch
        bs = self.cfg.batch_size
        geom = self.geometry()
        buckets = bucket_by_tokens([np.asarray(w).size for w in windows], geom)

        order: list[int] = []
        chunks: list[np.ndarray] = []
        for n_tok in sorted(buckets):
            idxs = buckets[n_tok]
            for s in range(0, len(idxs), bs):
                sel = idxs[s:s + bs]
                batch = [torch.tensor(np.asarray(windows[j], dtype=np.float32)) for j in sel]
                with torch.no_grad():
                    emb, _ = self.pipeline.embed(batch)   # (B, n_tok, d_model)
                    assert emb.shape[1] == n_tok, (
                        f"token-count mismatch: expected {n_tok}, got {emb.shape[1]} "
                        f"for lengths {[int(np.asarray(windows[j]).size) for j in sel]}. "
                        "Padding would no longer be homogeneous — refusing to pool."
                    )
                    pooled = pool_tokens(emb, self.cfg.pooling)
                chunks.append(pooled.to(torch.float32).cpu().numpy())
                order.extend(sel)

        arr = np.concatenate(chunks, axis=0)
        out = np.empty_like(arr)
        out[np.asarray(order)] = arr             # restore the caller's order
        self._d_model = out.shape[1]
        return out


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
    # The recipe version is part of the key: v1 caches were produced by the
    # pre-2026-07-30 pooling that averaged over padding, and must never be reused.
    key = (f"{cfg.encoder_type}__{cfg.pretrained_model.replace('/', '_')}"
           f"__{cfg.pooling}__{EMBEDDING_RECIPE_VERSION}__{_windows_hash(windows)}")
    cache_path = Path(cache_dir) / f"{key}.npy" if cache_dir else None

    if use_cache and cache_path and cache_path.exists():
        return np.load(cache_path)

    encoder = build_encoder(cfg)
    emb = encoder.encode(windows)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb
