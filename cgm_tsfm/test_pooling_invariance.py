"""Regression test for the 2026-07-30 batch-invariance bug in embedding pooling.

THE BUG. `pipeline.embed()` left-pads every series in a batch out to the longest
member, so the token axis is sized by the longest window. Pooling with a plain
`emb.mean(dim=1)` therefore averaged in positions produced by pure padding, and a
session's embedding depended on which other sessions happened to share its batch.
On the real data (956 sessions, batch_size=32) 861 windows (90%) came out with
cosine similarity < 0.9 against their correctly-pooled vector; a 13-reading
window batched beside a 288-reading window landed at cosine 0.22.

THE FIX (`encoders.py`). Windows are bucketed by token count before encoding, so
no window is ever padded to match a longer one. That makes the result identical
to encoding each window alone while keeping the speed of batching.

Run it:
    python -m cgm_tsfm.test_pooling_invariance                  # mock, offline, instant
    python -m cgm_tsfm.test_pooling_invariance --chronos        # real Chronos (downloads)
    python -m cgm_tsfm.test_pooling_invariance --chronos --device cuda

Exits non-zero on failure so it can be wired into CI.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import config as C
from .encoders import bucket_by_tokens, extract_embeddings, tokens_for

# Window lengths chosen to span the real data (3..288 readings) and to straddle
# patch boundaries (16, 17, 32, 33) where the padding arithmetic is easiest to get
# wrong. The extreme mix is the point: a 3-reading window next to a 288-reading
# one is exactly the case the bug corrupted worst.
LENGTHS = [3, 13, 16, 17, 20, 32, 33, 36, 45, 64, 71, 144, 288]

TOL = 1e-5          # float32 reduction-order noise on GPU is ~2e-7


def _windows(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(150, 30, L).astype(np.float32) for L in LENGTHS]


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check embedding pooling is batch-invariant")
    ap.add_argument("--chronos", action="store_true", help="use the real Chronos encoder")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    enc_type = "chronos" if args.chronos else "mock"
    windows = _windows()
    print(f"pooling invariance check — encoder={enc_type} lengths={LENGTHS}")

    def embed(ws, *, pooling="mean", batch_size=32):
        cfg = C.EncoderConfig(encoder_type=enc_type, pretrained_model=args.model,
                              pooling=pooling, device=args.device, batch_size=batch_size)
        # use_cache=False: we are testing computation, not the cache
        return extract_embeddings(ws, cfg, use_cache=False)

    # 1. Batch size must not change the answer. With the bug this failed badly,
    #    because batch composition determined how much padding each window saw.
    ref = embed(windows, batch_size=32)
    for bs in (1, 2, 5, 64, 256):
        got = embed(windows, batch_size=bs)
        d = np.abs(got - ref).max()
        print(f"  batch_size={bs:<4d} max|diff| vs bs=32 = {d:.2e}")
        if d > TOL:
            _fail(f"batch_size={bs} changed the embedding by {d:.3e} (> {TOL:.0e}). "
                  "Pooling is averaging over padding again.")

    # 2. Input order must not change the answer.
    perm = np.random.default_rng(1).permutation(len(windows))
    shuffled = embed([windows[i] for i in perm])
    restored = np.empty_like(shuffled)
    restored[perm] = shuffled
    d = np.abs(restored - ref).max()
    print(f"  shuffled input order        max|diff| = {d:.2e}")
    if d > TOL:
        _fail(f"input order changed the embedding by {d:.3e}")

    # 3. The strongest form: batching must equal encoding each window alone.
    alone = np.concatenate([embed([w], batch_size=1) for w in windows], axis=0)
    d = np.abs(ref - alone).max()
    print(f"  batched vs one-at-a-time    max|diff| = {d:.2e}")
    if d > TOL:
        _fail(f"batched pooling differs from per-window pooling by {d:.3e}")

    # 4. The worst-case pair that originally exposed the bug: shortest + longest.
    short, long_ = windows[0], windows[-1]
    pair = embed([short, long_], batch_size=32)
    solo = embed([short], batch_size=1)[0]
    cos = float(pair[0] @ solo / (np.linalg.norm(pair[0]) * np.linalg.norm(solo)))
    print(f"  {LENGTHS[0]}-reading window alone vs beside a {LENGTHS[-1]}-reading one: cosine = {cos:.6f}")
    if cos < 1 - 1e-4:
        _fail(f"cosine {cos:.4f} — the original bug is back (it measured 0.22)")

    # 5. Bucketing must predict the true token count. Only meaningful for a real
    #    checkpoint; the mock encoder has no token axis.
    if args.chronos:
        import torch

        from .encoders import ChronosEncoder
        e = ChronosEncoder(C.EncoderConfig(pretrained_model=args.model, device=args.device))
        geom = e.geometry()
        print(f"  geometry (patch, extra, context) = {geom}")
        for L in LENGTHS:
            with torch.no_grad():
                actual = e.pipeline.embed([torch.zeros(L, device=args.device)])[0].shape[1]
            predicted = tokens_for(L, geom)
            if predicted != actual:
                _fail(f"token count for L={L}: predicted {predicted}, actual {actual}")
        print(f"  token-count formula correct for all {len(LENGTHS)} lengths")

        buckets = bucket_by_tokens(LENGTHS, geom)
        print(f"  {len(LENGTHS)} windows fell into {len(buckets)} homogeneous buckets: "
              f"{ {k: len(v) for k, v in sorted(buckets.items())} }")

    print("\npooling invariance check PASSED — embeddings do not depend on batching.")


if __name__ == "__main__":
    main()
