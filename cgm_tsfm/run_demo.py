"""End-to-end demo of Arm A (frozen Chronos embeddings + grouped-CV regression).

Runs today with no real data and no model download:

    # fully offline smoke test (mock encoder):
    python -m cgm_tsfm.run_demo --encoder mock

    # real Chronos embeddings (downloads amazon/chronos-bolt-small on first run):
    python -m cgm_tsfm.run_demo --encoder chronos

    # once the real CSVs are in data/Merged_glucose_data/:
    python -m cgm_tsfm.run_demo --encoder chronos --real

It also prints the group-CV vs session-CV diagnostic, reproducing the check that
told the advisor's team the glucose->cognition signal is largely subject-baseline.
"""

from __future__ import annotations

import argparse

from . import config as C
from .data import generate_synthetic_data, load_real_data
from .encoders import extract_embeddings
from .regression import format_results, run_arm_a


def main() -> None:
    ap = argparse.ArgumentParser(description="CGM->cognition TSFM Arm-A demo")
    ap.add_argument("--encoder", choices=["mock", "chronos"], default="mock")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true", help="use real CSVs instead of synthetic")
    ap.add_argument("--max-readings", type=int, default=C.DEFAULT_MAX_READINGS,
                    help="give every session exactly N readings (default 24 = 2.0 h)")
    ap.add_argument("--variable-window", action="store_true",
                    help="revert to the pre-2026-08 behaviour: variable-length input")
    ap.add_argument("--signal", type=float, default=0.25,
                    help="synthetic true-effect strength (0 = null dataset)")
    args = ap.parse_args()

    window = C.WindowConfig(max_readings=None if args.variable_window else args.max_readings,
                              require_full=not args.variable_window)

    if args.real:
        print(f"Loading real data from {C.DATA_DIR} ...")
        ds = load_real_data(window=window)
    else:
        print(f"Generating synthetic data (signal_strength={args.signal}) ...")
        ds = generate_synthetic_data(window=window, signal_strength=args.signal)
    print(ds.summary())

    enc_cfg = C.EncoderConfig(
        encoder_type=args.encoder, pretrained_model=args.model, device=args.device,
    )
    print(f"\nExtracting embeddings with encoder='{args.encoder}' "
          f"({args.model if args.encoder == 'chronos' else 'summary-stats stand-in'}) ...")
    emb = extract_embeddings(ds.windows, enc_cfg)
    print(f"Embeddings: {emb.shape}  (N sessions x d_model)")

    for scheme in ("group", "session"):
        results = run_arm_a(ds, emb, cv_scheme=scheme)
        print(format_results(results, scheme))

    print("Interpretation: if 'group' R² ~ 0 but 'session' R² > 0, the signal is\n"
          "subject-baseline, not a generalizable glucose->cognition relationship\n"
          "(the same conclusion the hand-crafted-feature arm reached).")


if __name__ == "__main__":
    main()
