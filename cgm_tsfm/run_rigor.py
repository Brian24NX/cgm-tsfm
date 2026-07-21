"""Rigor checks that make the 'low prediction accuracy' conclusion convincing
(the advisor: "if it turns out no signal, we want very concrete evidence").

Two checks, both reusing the frozen Chronos embeddings + subject-grouped CV:

  1) POSITIVE CONTROL — predict a *glucose-derived* property (mean glucose, SD,
     % time high) from the SAME embeddings under the SAME grouped CV. These
     SHOULD be highly predictable (the embedding encodes the curve). If they are
     but cognition is ~0, the null is real — not a broken pipeline / dead
     embeddings.

  2) PERMUTATION TEST — shuffle the cognitive scores many times and recompute
     grouped-CV R². If the real R² sits inside this random-chance distribution
     (large p-value), there is no signal beyond chance.

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
    ap = argparse.ArgumentParser(description="Rigor checks: positive control + permutation test")
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

    # ---- 1) POSITIVE CONTROL -------------------------------------------------
    W = ds.windows
    controls = {
        "mean glucose (mg/dL)": np.array([float(np.mean(w)) for w in W]),
        "glucose SD":           np.array([float(np.std(w)) for w in W]),
        "% time > 180":         np.array([float(np.mean(w > 180)) for w in W]),
    }
    print("\n=== POSITIVE CONTROL — predict a glucose property from the embeddings (should be HIGH) ===")
    ctrl_rows = []
    for name, yc in controls.items():
        r_ridge = grouped_cv_r2(X, yc, g_all, ridge)
        r_svr = grouped_cv_r2(X, yc, g_all, SVR(kernel="rbf", C=10.0))
        best = max(r_ridge, r_svr)
        ctrl_rows.append((name, r_ridge, r_svr, best))
        print(f"  {name:22s} Ridge={r_ridge:+.3f}  SVR={r_svr:+.3f}  best={best:+.3f}")

    # ---- 2) PERMUTATION TEST -------------------------------------------------
    print(f"\n=== PERMUTATION TEST — cognition R² vs {args.n_perm} shuffled-label runs ===")
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
              f"chance={null.mean():+.3f}±{null.std():.3f}  95th={p95:+.3f}  p={p:.3f}")

    verdicts = []
    if all(b > 0.5 for _, _, _, b in ctrl_rows):
        verdicts.append("✅ Positive control passes: the embeddings + pipeline DO predict glucose properties well.")
    if all(p > 0.05 for *_, p in perm_rows):
        verdicts.append("✅ Permutation test: cognition R² is NOT above chance (p>0.05 for every score) → no signal beyond chance.")
    for v in verdicts:
        print("\n" + v)

    # ---- markdown ------------------------------------------------------------
    out = Path(args.out) if args.out else RESULTS_DIR / f"rigor_{'real' if args.real else 'synthetic'}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# Rigor checks — positive control + permutation test",
        "",
        f"- **Data**: {'REAL' if args.real else 'SYNTHETIC'}  ·  **Encoder**: "
        f"`{args.model if args.encoder=='chronos' else 'mock'}`  ·  **Permutations**: {args.n_perm}",
        f"- **Generated**: {datetime.now().isoformat(timespec='seconds')}",
        "- Model: `StandardScaler → PCA(32) → Ridge(alpha=10)`, subject-grouped 5-fold CV (same for every row).",
        "",
        "## 1. Positive control — can the SAME embeddings predict a *glucose* property?",
        "*(Sanity: these should be HIGH. If they are but cognition ≈ 0, the null is real — not a broken pipeline.)*",
        "",
        "| glucose target | Ridge R² | SVR R² | best |",
        "|---|--:|--:|--:|",
    ]
    for name, rr, rs, b in ctrl_rows:
        L.append(f"| {name} | {rr:.3f} | {rs:.3f} | **{b:.3f}** |")
    L += [
        "",
        "## 2. Permutation test — is the cognition R² better than chance?",
        f"*Shuffle each score {args.n_perm}× and recompute grouped-CV R². `p` = fraction of shuffles ≥ the real R². "
        "**p > 0.05 ⇒ the real R² is within the random-chance range ⇒ no signal beyond chance.***",
        "",
        "| score | real R² | chance mean ± sd | chance 95th %ile | p-value |",
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
