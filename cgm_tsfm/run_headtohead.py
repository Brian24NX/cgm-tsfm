"""ONE-COMMAND head-to-head: Chronos vs Mack's 43 hand-crafted features (+ optional Arm B).

All contenders go through subject-grouped CV so the only difference is the
representation / model. This is the K01's "raw-data learning vs feature
engineering" comparison, made fair.

Contenders:
  * Chronos (frozen embeddings) + Ridge/SVR    — Arm A
  * Hand-crafted (43 features)  + Ridge/SVR    — Mack's arm, ported verbatim
  * Chronos (frozen embeddings) + trainable MLP head  — Arm B  (only with --with-arm-b)

    python -m cgm_tsfm.run_headtohead --encoder chronos                 # A vs hand-crafted
    python -m cgm_tsfm.run_headtohead --encoder chronos --with-arm-b    # + trainable head
    python -m cgm_tsfm.run_headtohead --encoder chronos --real          # the real thing
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import config as C
from .data import generate_synthetic_data, load_real_data
from .encoders import extract_embeddings
from .handcrafted import GLUCOSE_FEATURE_NAMES, extract_handcrafted_features
from .regression import format_results, run_arm_a

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _best_row(rows: list[dict]) -> dict:
    """Best non-baseline model by mean R² (Arm B has a single non-baseline row)."""
    models = [r for r in rows if r["model"] != "Baseline(mean)"]
    return max(models, key=lambda r: r["r2_mean"])


def _verdict(entries: list[tuple[str, float]], tol: float = 0.02) -> str:
    """entries: (label, mean R²). Winner, tie, or all-baseline."""
    ranked = sorted(entries, key=lambda e: -e[1])
    if ranked[0][1] <= tol:
        return "all ≈ baseline — no generalizable signal from any representation"
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] <= tol:
        return f"~tie at top ({ranked[0][0]} ≈ {ranked[1][0]}, R²≈{ranked[0][1]:.3f})"
    return f"{ranked[0][0]} wins (R²={ranked[0][1]:.3f})"


def _write_markdown(path: Path, ds, contenders: list[tuple[str, dict]], args) -> None:
    """Dump the head-to-head as per-target contender tables + verdicts (writeup-ready)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Head-to-head: representations for CGM → cognition",
        "",
        f"- **Data**: {'REAL' if args.real else 'SYNTHETIC'}",
        f"- **CV**: `{args.cv}`  ·  **target-norm**: `{args.target_norm}`  ·  **PCA**: `{args.pca or 'none'}`",
        f"- **Encoder**: `{args.model if args.encoder == 'chronos' else 'mock (summary-stats stand-in)'}`",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "- **Protocol**: subject-grouped CV; cell = best model per representation; R² vs a mean-predictor (higher = better).",
        "",
    ]
    if args.target_norm != "none":
        lines += [f"> within-subject `{args.target_norm}`: R² = fraction of WITHIN-subject variance explained "
                  "(oracle centering — a characterization, not a new-subject predictor).", ""]
    if not args.real:
        lines += ["> ⚠️ **SYNTHETIC data** — harness check, not scientific results. Re-run with `--real`.", ""]

    for target in ds.targets:
        info0 = contenders[0][1][target]
        lines += [f"## {target}  (n={info0['n']} sessions, {info0['n_subjects']} subjects)", "",
                  "| representation | best model | R² | RMSE |", "|---|---|---|---|"]
        entries = []
        for label, res in contenders:
            b = _best_row(res[target]["rows"])
            lines.append(f"| {label} | {b['model']} | {b['r2']} | {b['rmse']} |")
            entries.append((label, b["r2_mean"]))
        lines += ["", f"**Verdict:** {_verdict(entries)}", ""]

    path.write_text("\n".join(lines))
    print(f"\nWrote head-to-head markdown to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronos vs hand-crafted (+ optional Arm B), head-to-head")
    ap.add_argument("--encoder", choices=["mock", "chronos"], default="chronos")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-readings", type=int, default=C.DEFAULT_MAX_READINGS)
    ap.add_argument("--variable-window", action="store_true",
                    help="revert to the pre-2026-08 behaviour: keep every session at whatever length it happens to be")
    ap.add_argument("--cv", choices=["group", "session"], default="group")
    ap.add_argument("--with-arm-b", action="store_true",
                    help="also train the Chronos + MLP-head arm (slower: trains nets over folds)")
    ap.add_argument("--max-epochs", type=int, default=100, help="Arm B max epochs (early-stopped)")
    ap.add_argument("--pca", type=int, default=None,
                    help="PCA n_components (fit per-fold) before Ridge/SVR — applies to the "
                         "Arm A contenders; use the best value from run_sweep --kind pca")
    ap.add_argument("--target-norm", choices=["none", "center", "zscore"], default="none",
                    help="within-subject target normalization — tests the subject-baseline "
                         "hypothesis (center = predict deviation from personal mean)")
    ap.add_argument("--verbose", action="store_true", help="also print full per-model tables")
    ap.add_argument("--out", default=None,
                    help="markdown output path (default results/headtohead_<synthetic|real>.md)")
    args = ap.parse_args()

    window = C.WindowConfig(max_readings=None if args.variable_window else args.max_readings,
                              require_full=not args.variable_window)
    if args.real:
        print(f"Loading real data from {C.DATA_DIR} ...")
        ds = load_real_data(window=window)
    else:
        print("Generating synthetic data ...")
        ds = generate_synthetic_data(window=window)
    print(ds.summary())

    # One frozen encoder pass (cached); two representations share the same windows.
    enc_cfg = C.EncoderConfig(encoder_type=args.encoder, pretrained_model=args.model, device=args.device)
    print(f"\nExtracting Chronos embeddings ({args.encoder}) ...")
    X_chronos = extract_embeddings(ds.windows, enc_cfg)
    print(f"  Chronos features: {X_chronos.shape}")
    print("Extracting 43 hand-crafted glycemic features ...")
    X_hand = extract_handcrafted_features(ds.windows)
    print(f"  Hand-crafted features: {X_hand.shape}  ({len(GLUCOSE_FEATURE_NAMES)} named)")

    pca_tag = f" +PCA{args.pca}" if args.pca else ""
    base_label = "Chronos" if args.encoder == "chronos" else "Mock"
    chronos_label = f"{base_label} ({X_chronos.shape[1]}d){pca_tag} + Ridge/SVR"
    contenders: list[tuple[str, dict]] = [
        (chronos_label, run_arm_a(ds, X_chronos, cv_scheme=args.cv, pca_components=args.pca, target_norm=args.target_norm)),
        (f"Hand-crafted (43){pca_tag} + Ridge/SVR",
         run_arm_a(ds, X_hand, cv_scheme=args.cv, pca_components=args.pca, target_norm=args.target_norm)),
    ]

    if args.with_arm_b:
        if args.cv != "group":
            print("\n[--with-arm-b uses grouped CV regardless of --cv]")
        from .ben_adapter.train import run_arm_b
        print("\nTraining Arm B (Chronos + trainable MLP head) over grouped folds ...")
        contenders.append((
            f"Chronos ({X_chronos.shape[1]}d) + MLP head",
            run_arm_b(ds, X_chronos, accelerator=args.device, max_epochs=args.max_epochs,
                      target_norm=args.target_norm),
        ))

    width = 96
    print("\n" + "=" * width)
    print(f" HEAD-TO-HEAD   [cv={args.cv}, target_norm={args.target_norm}]   {len(contenders)} contenders")
    print(" Subject-grouped CV. R² is vs a mean-predictor; higher = better.")
    if args.target_norm != "none":
        print(f" (within-subject {args.target_norm}: R² = fraction of WITHIN-subject variance explained)")
    print("=" * width)
    for target in ds.targets:
        info0 = contenders[0][1][target]
        print(f"\n[{target}]  n={info0['n']} sessions, {info0['n_subjects']} subjects")
        print(f" {'representation':34s} {'best model':14s} {'R2':>16s} {'RMSE':>16s}")
        entries = []
        for label, res in contenders:
            b = _best_row(res[target]["rows"])
            print(f" {label:34s} {b['model']:14s} {b['r2']:>16s} {b['rmse']:>16s}")
            entries.append((label, b["r2_mean"]))
        print(f"    → verdict: {_verdict(entries)}")

    if args.verbose:
        print("\n----- full per-model tables -----")
        for label, res in contenders:
            print(f"### {label}", format_results(res, args.cv, title=label))

    out = Path(args.out) if args.out else RESULTS_DIR / f"headtohead_{'real' if args.real else 'synthetic'}.md"
    _write_markdown(out, ds, contenders, args)

    print("\nNote: on SYNTHETIC data the true signal is defined via summary statistics,\n"
          "which favors hand-crafted features — treat synthetic numbers as a plumbing check,\n"
          "not evidence. Re-run with --real for the real comparison.")


if __name__ == "__main__":
    main()
