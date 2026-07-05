"""Sweep Chronos checkpoint, window, pooling, PCA, and within-subject target-norm.

One-factor-at-a-time (not a full grid): vary one axis while holding the others at
a baseline. Each config extracts embeddings (cached per model/pooling/window) and
runs the SAME grouped nested CV as the head-to-head, then reports best-model R²
per target + the 3-target mean. Every table is also written to a markdown file
for the writeup (see --out).

    python -m cgm_tsfm.run_sweep --kind all            # all axes -> results/sweep_synthetic.md
    python -m cgm_tsfm.run_sweep --kind all --real      # -> results/sweep_real.md
    python -m cgm_tsfm.run_sweep --kind targetnorm --out results/mytable.md
    python -m cgm_tsfm.run_sweep --kind pca --target-norm center

Chronos-Bolt sizes: tiny/mini/small/base (no "large"). Chronos-T5: adds large.
T5 uses the same BaseChronosPipeline.embed path (auto-detected), so it drops in.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path

from . import config as C
from .data import generate_synthetic_data, load_real_data
from .encoders import extract_embeddings
from .regression import run_arm_a

BASELINE_MODEL = "amazon/chronos-bolt-small"
CHECKPOINTS = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    "amazon/chronos-t5-small",
]
WINDOWS = [24, 30, 36, None]      # 2h, 2.5h, 3h, full  (5-min cadence)
POOLINGS = ["mean", "last"]
PCA_COMPONENTS = [None, 8, 16, 32, 64, 128]   # None = full-dim (no PCA)
TARGET_NORMS = ["none", "center", "zscore"]   # within-subject target normalization
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _make_dataset(real: bool, max_readings: int | None):
    window = C.WindowConfig(max_readings=max_readings)
    return load_real_data(window=window) if real else generate_synthetic_data(window=window)


def _best_r2_per_target(results: dict) -> dict[str, float]:
    out = {}
    for t, info in results.items():
        best = max((r for r in info["rows"] if r["model"] != "Baseline(mean)"),
                   key=lambda r: r["r2_mean"])
        out[t] = best["r2_mean"]
    return out


def _eval_config(ds, model, pooling, device, pca_components=None, target_norm="none"):
    enc = C.EncoderConfig(encoder_type="chronos", pretrained_model=model, pooling=pooling, device=device)
    emb = extract_embeddings(ds.windows, enc)          # cached per model/pooling/window
    res = run_arm_a(ds, emb, cv_scheme="group", pca_components=pca_components, target_norm=target_norm)
    return emb.shape[1], _best_r2_per_target(res)


def _run_axis(configs, device) -> list[dict]:
    """configs: list of (label, model, pooling, dataset, pca_components, target_norm)."""
    rows = []
    for label, model, pooling, ds, pca, tnorm in configs:
        print(f"  … {label}", flush=True)
        try:
            d, r2 = _eval_config(ds, model, pooling, device, pca_components=pca, target_norm=tnorm)
            rows.append({"label": label, "d": d, "r2": r2})
        except Exception as e:
            traceback.print_exc()
            rows.append({"label": label, "error": f"{type(e).__name__}: {e}"})
    return rows


def _print_table(title: str, targets: list[str], rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f" {title}")
    print(" best-model R² per target under grouped nested CV (higher = better; vs mean-predictor)")
    print("=" * 100)
    short = [t.replace("_cognitive_score", "") for t in targets]
    print(f" {'config':32s} {'d':>5s} " + " ".join(f"{s:>10s}" for s in short) + f" {'mean_R²':>10s}")
    for r in rows:
        if r.get("error"):
            print(f" {r['label']:32s}   ERR  {r['error']}")
            continue
        cells = " ".join(f"{r['r2'][t]:>10.3f}" for t in targets)
        mean_r2 = sum(r["r2"].values()) / len(targets)
        print(f" {r['label']:32s} {r['d']:>5d} {cells} {mean_r2:>10.3f}")


def _markdown_table(title: str, targets: list[str], rows: list[dict]) -> str:
    short = [t.replace("_cognitive_score", "") for t in targets]
    lines = [f"### {title}", ""]
    lines.append("| config | d | " + " | ".join(short) + " | mean R² |")
    lines.append("|" + "---|" * (len(short) + 3))
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['label']} | — | " + " | ".join(["ERR"] * len(short)) + f" | {r['error']} |")
            continue
        cells = " | ".join(f"{r['r2'][t]:.3f}" for t in targets)
        mean_r2 = sum(r["r2"].values()) / len(targets)
        lines.append(f"| {r['label']} | {r['d']} | {cells} | {mean_r2:.3f} |")
    lines.append("")
    return "\n".join(lines)


def _do_axis(title, configs, device, targets, sections) -> None:
    rows = _run_axis(configs, device)
    _print_table(title, targets, rows)
    sections.append((title, rows))


def _write_markdown(path: Path, targets, sections, real: bool, target_norm: str, summary: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = [
        "# Chronos sweep results",
        "",
        f"- **Data**: {'REAL' if real else 'SYNTHETIC'}",
        f"- **Global target-norm** (checkpoint/window/pooling/pca axes): `{target_norm}`",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "- **Protocol**: subject-grouped nested CV; cell = best-model R² per target vs a mean-predictor (higher = better).",
        "",
        "```",
        summary,
        "```",
        "",
    ]
    if not real:
        hdr += ["> ⚠️ **SYNTHETIC data** — these numbers exercise the sweep harness only; "
                "they are not scientific results. Re-run with `--real` for the real comparison.", ""]
    body = "\n".join(_markdown_table(t, targets, rows) for t, rows in sections)
    path.write_text("\n".join(hdr) + body + "\n")
    print(f"\nWrote {len(sections)} table(s) to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronos checkpoint / window / pooling / PCA / target-norm sweep")
    ap.add_argument("--kind", choices=["checkpoint", "window", "pooling", "pca", "targetnorm", "all"], default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--target-norm", choices=TARGET_NORMS, default="none",
                    help="within-subject target-norm applied to the checkpoint/window/pooling/pca axes")
    ap.add_argument("--out", default=None, help="markdown output path (default results/sweep_<synthetic|real>.md)")
    args = ap.parse_args()
    tn = args.target_norm

    ds_full = _make_dataset(args.real, None)
    targets = list(ds_full.targets.keys())
    summary = ("REAL" if args.real else "SYNTHETIC") + f" data (target_norm={tn} for non-targetnorm axes):\n" + ds_full.summary()
    print(summary)

    sections: list[tuple[str, list[dict]]] = []

    if args.kind in ("checkpoint", "all"):
        print("\n>>> CHECKPOINT SWEEP (window=full, pooling=mean, no PCA)")
        cfgs = [(m.split("/")[-1], m, "mean", ds_full, None, tn) for m in CHECKPOINTS]
        _do_axis(f"Checkpoint sweep (window=full, pooling=mean, no PCA, target_norm={tn})", cfgs, args.device, targets, sections)

    if args.kind in ("window", "all"):
        print("\n>>> WINDOW SWEEP (model=bolt-small, pooling=mean, no PCA)")
        cfgs = []
        for w in WINDOWS:
            lab = "full" if w is None else f"{w}rd (~{w*C.CGM_SAMPLING_MINUTES/60:.1f}h)"
            cfgs.append((lab, BASELINE_MODEL, "mean", _make_dataset(args.real, w), None, tn))
        _do_axis(f"Window sweep (model=chronos-bolt-small, pooling=mean, no PCA, target_norm={tn})", cfgs, args.device, targets, sections)

    if args.kind in ("pooling", "all"):
        print("\n>>> POOLING SWEEP (model=bolt-small, window=full, no PCA)")
        cfgs = [(p, BASELINE_MODEL, p, ds_full, None, tn) for p in POOLINGS]
        _do_axis(f"Pooling sweep (model=chronos-bolt-small, window=full, no PCA, target_norm={tn})", cfgs, args.device, targets, sections)

    if args.kind in ("pca", "all"):
        print("\n>>> PCA SWEEP (model=bolt-small, window=full, pooling=mean)")
        cfgs = [("no PCA (full-dim)" if k is None else f"PCA={k}", BASELINE_MODEL, "mean", ds_full, k, tn)
                for k in PCA_COMPONENTS]
        _do_axis(f"PCA sweep (model=chronos-bolt-small, window=full, pooling=mean, target_norm={tn})", cfgs, args.device, targets, sections)

    if args.kind in ("targetnorm", "all"):
        print("\n>>> TARGET-NORM SWEEP (model=bolt-small, window=full, pooling=mean, no PCA)")
        labels = {"none": "none (raw score)", "center": "within-subj center", "zscore": "within-subj zscore"}
        cfgs = [(labels[t], BASELINE_MODEL, "mean", ds_full, None, t) for t in TARGET_NORMS]
        _do_axis("Target-norm sweep (model=chronos-bolt-small, window=full, pooling=mean); center/zscore R² = within-subject variance explained (oracle centering)",
                 cfgs, args.device, targets, sections)

    out = Path(args.out) if args.out else RESULTS_DIR / f"sweep_{'real' if args.real else 'synthetic'}.md"
    _write_markdown(out, targets, sections, args.real, tn, summary)

    print("\nNote: on SYNTHETIC data these numbers only exercise the sweep harness — the true\n"
          "signal is summary-stat-based. Re-run with --real for a meaningful comparison.")


if __name__ == "__main__":
    main()
