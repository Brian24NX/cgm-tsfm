"""Subgroup signal-search: does the CGM→cognition relationship show up in
specific glucose regimes (hypoglycemia / hyperglycemia) even if it's absent
overall?  Per the advisor's steer: "always prioritize actively hunting for signal
(e.g. restrict to hypo/hyper windows, or specific subgroups)."

For each subgroup we RE-RUN Arm A (frozen Chronos embeddings + Ridge/SVR under
subject-grouped CV) on just the sessions in that subgroup, and report best-model
R² per target alongside the subgroup's session/subject counts.

    python -m cgm_tsfm.run_subgroups --encoder chronos --real --device cuda

⚠️ Subgroups shrink both the sessions AND the subjects, so these are EXPLORATORY:
grouped CV on a handful of subjects is noisy. We (a) print n_subjects for every
subgroup, (b) adapt the CV fold count to the subjects available, and (c) skip
subgroups with too few subjects for an honest grouped split.
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config as C
from .data import CGMDataset, generate_synthetic_data, load_real_data
from .encoders import extract_embeddings
from .regression import run_arm_a

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MIN_SUBJECTS = 4          # need at least this many subjects to attempt grouped CV


def subgroup_masks(windows: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean mask per subgroup, defined by each window's glucose excursions."""
    mins = np.array([float(np.min(w)) for w in windows])
    maxs = np.array([float(np.max(w)) for w in windows])
    return {
        "all sessions (reference)":     np.ones(len(windows), dtype=bool),
        "hypo  (any reading < 70)":     mins < 70,
        "hyper (any reading > 250)":    maxs > 250,
        "any excursion (low OR high)":  (mins < 70) | (maxs > 250),
        "in-range only (70–180)":       (mins >= 70) & (maxs <= 180),
    }


def _subset(ds: CGMDataset, emb: np.ndarray, m: np.ndarray) -> tuple[CGMDataset, np.ndarray]:
    sub = CGMDataset(
        windows=[w for w, keep in zip(ds.windows, m) if keep],
        subjects=ds.subjects[m],
        sessions=ds.sessions[m],
        targets={t: v[m] for t, v in ds.targets.items()},
    )
    return sub, emb[m]


def _best_r2(results: dict) -> dict[str, float | None]:
    """Best non-baseline mean-R² per target (None if a target had too few data)."""
    out: dict[str, float | None] = {}
    for t, info in results.items():
        # keep non-baseline models with a FINITE mean-R² (drop NaN from degenerate tiny folds)
        vals = [r["r2_mean"] for r in info["rows"]
                if r["model"] != "Baseline(mean)" and r["r2_mean"] == r["r2_mean"]]
        out[t] = max(vals) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Glucose-regime subgroup signal search (Arm A)")
    ap.add_argument("--encoder", choices=["mock", "chronos"], default="chronos")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-readings", type=int, default=C.DEFAULT_MAX_READINGS,
                    help="give every session exactly N readings (default 24 = 2.0 h)")
    ap.add_argument("--variable-window", action="store_true",
                    help="revert to the pre-2026-08 behaviour: variable-length input")
    ap.add_argument("--target-norm", choices=["none", "center", "zscore"], default="none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    window = C.WindowConfig(max_readings=None if args.variable_window else args.max_readings,
                              require_full=not args.variable_window)
    ds = load_real_data(window=window) if args.real else generate_synthetic_data(window=window)
    print(ds.summary())

    enc = C.EncoderConfig(encoder_type=args.encoder, pretrained_model=args.model, device=args.device)
    print(f"\nExtracting {args.encoder} embeddings ...")
    emb = extract_embeddings(ds.windows, enc)
    print(f"  embeddings: {emb.shape}")

    targets = list(ds.targets.keys())
    masks = subgroup_masks(ds.windows)

    rows = []          # (label, n_sessions, n_subjects, {target: r2 or None}, note)
    for label, m in masks.items():
        n_sess = int(m.sum())
        n_subj = int(len(np.unique(ds.subjects[m]))) if n_sess else 0
        print(f"\n>>> {label}: {n_sess} sessions, {n_subj} subjects", flush=True)
        if n_subj < MIN_SUBJECTS:
            rows.append((label, n_sess, n_subj, {t: None for t in targets},
                         f"skipped (<{MIN_SUBJECTS} subjects)"))
            continue
        k = min(C.OUTER_CV_SPLITS, n_subj)              # adapt outer folds to subjects
        inner = max(2, min(C.INNER_CV_SPLITS, k - 1))   # keep inner CV valid
        try:
            sub_ds, sub_emb = _subset(ds, emb, m)
            res = run_arm_a(sub_ds, sub_emb, cv_scheme="group",
                            target_norm=args.target_norm, outer_splits=k, inner_splits=inner)
            r2 = _best_r2(res)
            rows.append((label, n_sess, n_subj, r2, f"grouped CV, {k} folds"))
            print("   best R²:", {t.replace('_cognitive_score', ''): (f"{v:.3f}" if v is not None else "—")
                                  for t, v in r2.items()})
        except Exception as e:
            traceback.print_exc()
            rows.append((label, n_sess, n_subj, {t: None for t in targets}, f"ERROR: {type(e).__name__}"))

    # ---- console table ----
    short = [t.replace("_cognitive_score", "") for t in targets]
    print("\n" + "=" * 92)
    print(" SUBGROUP SIGNAL SEARCH — best-model R² per target (subject-grouped CV; higher = better; 0 = baseline)")
    print("=" * 92)
    print(f" {'subgroup':30s} {'sess':>5s} {'subj':>5s} " + " ".join(f"{s:>9s}" for s in short) + "  note")
    for label, n_sess, n_subj, r2, note in rows:
        cells = " ".join((f"{r2[t]:>9.3f}" if r2[t] is not None else f"{'—':>9s}") for t in targets)
        print(f" {label:30s} {n_sess:>5d} {n_subj:>5d} {cells}  {note}")

    # ---- markdown ----
    out = Path(args.out) if args.out else RESULTS_DIR / f"subgroups_{'real' if args.real else 'synthetic'}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Subgroup signal search — glucose regimes",
        "",
        f"- **Data**: {'REAL' if args.real else 'SYNTHETIC'}  ·  **Encoder**: "
        f"`{args.model if args.encoder == 'chronos' else 'mock'}`  ·  **target-norm**: `{args.target_norm}`",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "- Per subgroup we re-run Arm A (Chronos embeddings + Ridge/SVR) under subject-grouped CV.",
        "- **R² vs a mean-predictor; higher = better; 0 = no better than guessing the subgroup's average.**",
        "",
        "> ⚠️ **Exploratory.** Subgroups shrink the subject count (shown), so grouped-CV R² here is noisy; "
        "read directions/trends, not precise values. Subgroups with fewer subjects than the fold count are skipped.",
        "",
        "| subgroup | sessions | subjects | " + " | ".join(short) + " | note |",
        "|---|--:|--:|" + "--:|" * len(short) + "---|",
    ]
    for label, n_sess, n_subj, r2, note in rows:
        cells = " | ".join((f"{r2[t]:.3f}" if r2[t] is not None else "—") for t in targets)
        lines.append(f"| {label} | {n_sess} | {n_subj} | {cells} | {note} |")
    if not args.real:
        lines += ["", "> ⚠️ SYNTHETIC data — plumbing check only, not evidence."]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
