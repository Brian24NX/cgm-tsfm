"""Rigor checks that make the 'low prediction accuracy' conclusion convincing
(the advisor: "if it turns out no signal, we want very concrete evidence").

Wording note: these two checks are conventionally called a "positive control"
and a "permutation test". The advisor asked us not to use that vocabulary, so
the output of this script says "known-answer check" and "shuffle test" instead.
Same procedures, plainer names.

Two checks, both reusing the frozen Chronos embeddings + subject-grouped CV:

  1) THE KNOWN-ANSWER CHECK — predict a *glucose-derived* property (mean glucose,
     SD, % time high) from the SAME embeddings under the SAME grouped CV. These
     should be well predicted, since the embedding describes the curve. If they
     are but the cognitive scores sit near zero, the low accuracy reflects the
     data rather than a broken pipeline / dead embeddings.
     Caveat measured on the real data: glucose VARIABILITY comes back at
     R²~0.46, but absolute LEVEL only ~0.04, because Chronos-Bolt subtracts each
     series' own mean and divides by its own standard deviation before encoding.
     So this check passes for shape and not for level -- report it that way.

  2) THE SHUFFLE TEST — randomly reassign the cognitive scores to the wrong
     sessions many times and recompute grouped-CV R². If the real R² sits inside
     the scrambled range (large p-value), scrambled scores do as well as real
     ones, i.e. there is no measurable relationship.

    python -m cgm_tsfm.run_rigor --encoder chronos --real --device cuda --n-perm 200
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C
from .data import generate_synthetic_data, load_real_data
from .encoders import extract_embeddings

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def grouped_cv_r2(X, y, g, estimator, pca: int | None = 32, k: int = 5) -> float:
    """Mean R² over subject-grouped folds, using scaler(→PCA)→estimator, fit per fold."""
    n_groups = len(np.unique(g))
    gkf = GroupKFold(n_splits=min(k, n_groups))
    scores = []
    for tr, te in gkf.split(X, y, g):
        steps = [("sc", StandardScaler())]
        if pca:
            steps.append(("pca", PCA(n_components=min(pca, X.shape[1]), random_state=C.RANDOM_STATE)))
        steps.append(("m", clone(estimator)))
        pipe = Pipeline(steps).fit(X[tr], y[tr])
        scores.append(r2_score(y[te], pipe.predict(X[te])))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rigor checks: the known-answer check + the shuffle test")
    ap.add_argument("--encoder", choices=["mock", "chronos"], default="chronos")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_real_data() if args.real else generate_synthetic_data()
    print(ds.summary())
    enc = C.EncoderConfig(encoder_type=args.encoder, pretrained_model=args.model, device=args.device)
    print(f"\nExtracting {args.encoder} embeddings ...")
    X = extract_embeddings(ds.windows, enc)
    g_all = ds.subjects
    print(f"  embeddings: {X.shape}")

    ridge = Ridge(alpha=10.0, random_state=C.RANDOM_STATE)   # one fixed model (no tuning) for a clean test

    # ---- 1) THE KNOWN-ANSWER CHECK -------------------------------------------
    W = ds.windows
    controls = {
        "mean glucose (mg/dL)": np.array([float(np.mean(w)) for w in W]),
        "glucose SD":           np.array([float(np.std(w)) for w in W]),
        "% time > 180":         np.array([float(np.mean(w > 180)) for w in W]),
    }
    print("\n=== KNOWN-ANSWER CHECK — predict a glucose property from the embeddings (should be high) ===")
    ctrl_rows = []
    for name, yc in controls.items():
        r_ridge = grouped_cv_r2(X, yc, g_all, ridge)
        r_svr = grouped_cv_r2(X, yc, g_all, SVR(kernel="rbf", C=10.0))
        best = max(r_ridge, r_svr)
        ctrl_rows.append((name, r_ridge, r_svr, best))
        print(f"  {name:22s} Ridge={r_ridge:+.3f}  SVR={r_svr:+.3f}  best={best:+.3f}")

    # ---- 2) THE SHUFFLE TEST -------------------------------------------------
    print(f"\n=== SHUFFLE TEST — real cognition R² vs {args.n_perm} scrambled-score runs ===")
    rng = np.random.default_rng(C.RANDOM_STATE)
    perm_rows = []
    for target in ds.targets:
        m = ds.target_mask(target)
        Xt, yt, gt = X[m], ds.targets[target][m], g_all[m]
        real = grouped_cv_r2(Xt, yt, gt, ridge)
        null = np.array([grouped_cv_r2(Xt, rng.permutation(yt), gt, ridge) for _ in range(args.n_perm)])
        p = (np.sum(null >= real) + 1) / (args.n_perm + 1)
        p95 = float(np.percentile(null, 95))
        perm_rows.append((target, real, float(null.mean()), float(null.std()), p95, p))
        print(f"  {target.replace('_cognitive_score',''):9s} real={real:+.3f}  "
              f"scrambled={null.mean():+.3f}±{null.std():.3f}  95th={p95:+.3f}  p={p:.3f}")

    verdicts = []
    best_ctrl = max(b for *_, b in ctrl_rows)
    if all(b > 0.5 for _, _, _, b in ctrl_rows):
        verdicts.append("✅ Known-answer check: the embeddings + pipeline predict every glucose property well.")
    elif best_ctrl > 0.3:
        verdicts.append(
            f"⚠️ Known-answer check is PARTIAL: the best glucose property reaches R²={best_ctrl:.3f} "
            "(so the pipeline does extract real information), but not all of them clear 0.5. "
            "Absolute level is recovered poorly because Chronos-Bolt subtracts each series' own mean and "
            "divides by its own standard deviation before the encoder sees it. Do NOT report this as "
            "a clean pass — say which properties are recovered and which are not."
        )
    else:
        verdicts.append(
            f"❌ Known-answer check FAILED: best glucose property only R²={best_ctrl:.3f}. "
            "Investigate the pipeline before interpreting any cognition result."
        )
    if all(p > 0.05 for *_, p in perm_rows):
        verdicts.append(
            "✅ Shuffle test: scrambled scores do about as well as the real ones for every score "
            "(p > 0.05), so there is no measurable relationship between the glucose and the score."
        )
    for v in verdicts:
        print("\n" + v)

    # ---- markdown ------------------------------------------------------------
    out = Path(args.out) if args.out else RESULTS_DIR / f"rigor_{'real' if args.real else 'synthetic'}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# Rigor checks — the known-answer check + the shuffle test",
        "",
        f"- **Data**: {'REAL' if args.real else 'SYNTHETIC'}  ·  **Encoder**: "
        f"`{args.model if args.encoder=='chronos' else 'mock'}`  ·  **Shuffle rounds**: {args.n_perm}",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "- Model: `StandardScaler → PCA(32) → Ridge(alpha=10)`, subject-grouped 5-fold CV (same for every row).",
        "",
        "## 1. Known-answer check — can the SAME embeddings predict a property of the *glucose*?",
        "*(These should be high. If they are, but the cognitive scores stay near zero, then the low accuracy "
        "reflects the data rather than a broken pipeline.)*",
        "",
        "| glucose target | Ridge R² | SVR R² | best |",
        "|---|--:|--:|--:|",
    ]
    for name, rr, rs, b in ctrl_rows:
        L.append(f"| {name} | {rr:.3f} | {rs:.3f} | **{b:.3f}** |")
    L += [
        "",
        "## 2. The shuffle test — do we do any better than scrambled scores?",
        f"*Randomly reassign the scores to the wrong sessions {args.n_perm}× (destroying any real relationship) and "
        "recompute grouped-CV R² each time. `p` = the fraction of scrambled runs that did at least as well as the "
        "real one. **p > 0.05 means scrambled scores do about as well as the real ones — i.e. no measurable "
        "relationship.*** Note the scrambled runs average about −0.05 rather than 0, because R² is measured against "
        "each test fold's own average.",
        "",
        "| score | real R² | scrambled mean ± sd | scrambled 95th %ile | p-value |",
        "|---|--:|--:|--:|--:|",
    ]
    for t, real, nm, ns, p95, p in perm_rows:
        L.append(f"| {t.replace('_cognitive_score','')} | {real:.3f} | {nm:.3f} ± {ns:.3f} | {p95:.3f} | **{p:.3f}** |")
    L += ["", "### Verdict", ""] + [f"- {v}" for v in verdicts]
    if not args.real:
        L += ["", "> ⚠️ SYNTHETIC data — plumbing check only."]
    out.write_text("\n".join(L) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
