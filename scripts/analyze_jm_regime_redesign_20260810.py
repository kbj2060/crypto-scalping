"""Stage 3a of the JM-only regime3 redesign: read the sweep reports and print the decision tables.

Three questions, in the order that decides whether anything downstream is worth running:

  1. Does the redesign axis carry information at all? -> VAL->OOS Spearman rank correlation over
     every gated cell. If tuning on VAL does not rank OOS, the winner is a draw from noise and
     nothing below matters. This project has closed several regime lines on exactly that finding.
  2. Which feature panel wins, per asset, judged at each panel's OWN best hyperparameters?
  3. Does the per-asset label basis (frozen vs quantile-matched) change the answer?

Balanced accuracy is reported against the ADX/slope/BB rule label because that is the contract the
downstream sidecar's state->class matrix is defined against -- but it is partly tautological:
`wide24` literally contains bb_width_z / hma_slope / macd_hist, i.e. the label's own ingredients,
while `jm6`/`jm9` are pure price-return panels that cannot reconstruct an ADX rule by design. So
the economic-separation t-stat (bull-predicted minus bear-predicted forward 1h return) is printed
beside it every time: a panel that wins on agreement but not on separation has only learned to
re-derive the rule, which is worth nothing downstream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import LABEL_BASES, SELECTION_WINDOW  # noqa: E402
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR, PANELS  # noqa: E402

SEL = SELECTION_WINDOW


def gated(rows: list[dict], basis: str) -> list[dict]:
    return [r for r in rows
            if r["label_basis"] == basis
            and r[f"{SEL}_median_run_bars"] >= 12.0
            and r[f"{SEL}_min_class_coverage"] >= 0.05]


def fmt(r: dict) -> str:
    return (f"{r['panel']:>14} {r['scaler']:>8} k={r['k']} lpd={r['lambda_per_dim']:<5g} "
            f"T={r['temperature_ratio']:<5g} | VAL bal={r[f'{SEL}_balanced_accuracy']:.4f} "
            f"run={r[f'{SEL}_median_run_bars']:>5.0f} cov={r[f'{SEL}_min_class_coverage']:.3f} "
            f"sep_t={r[f'{SEL}_economic_separation_tstat']:>6.2f} | "
            f"OOS bal={r['oos_balanced_accuracy']:.4f} "
            f"run={r['oos_median_run_bars']:>5.0f} sep_t={r['oos_economic_separation_tstat']:>6.2f}")


def main() -> None:
    prep = json.loads((OUT_DIR / "prep_report.json").read_text())
    for asset in ("btc", "eth"):
        path = OUT_DIR / f"sweep_{asset}_report.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        rep = json.loads(path.read_text())
        rows = rep["cells"]
        print("\n" + "=" * 118)
        print(f"{asset.upper()}  --  {len(rows)} scored cells  "
              f"(fit {rep['protocol']['fit_year']}, select {SEL}, OOS held out)")
        lb = prep["assets"][asset]["label_balance"]
        print(f"  label balance 2024  frozen: " + " ".join(f"{k}={v:.3f}" for k, v in lb["frozen"]["2024"].items())
              + "   qmatched: " + " ".join(f"{k}={v:.3f}" for k, v in lb["qmatched"]["2024"].items()))
        print("=" * 118)

        for basis in LABEL_BASES:
            g = gated(rows, basis)
            n_basis = sum(1 for r in rows if r["label_basis"] == basis)
            print(f"\n--- label basis: {basis}   ({len(g)}/{n_basis} cells pass "
                  f"persistence>=12 bars and min-class-coverage>=5% on VAL)")
            if not g:
                print("    no cell passes the gates")
                continue
            v = np.array([r[f"{SEL}_balanced_accuracy"] for r in g])
            o = np.array([r["oos_balanced_accuracy"] for r in g])
            from scipy.stats import spearmanr
            rho, p = spearmanr(v, o)
            print(f"    VAL->OOS rank transfer (balanced accuracy): spearman rho={rho:+.3f} "
                  f"p={p:.3g} over n={len(g)} gated cells")
            vs = np.array([r[f"{SEL}_economic_separation_tstat"] for r in g])
            os_ = np.array([r["oos_economic_separation_tstat"] for r in g])
            rho2, p2 = spearmanr(vs, os_)
            print(f"    VAL->OOS rank transfer (economic separation t): spearman rho={rho2:+.3f} p={p2:.3g}")

            print(f"\n    best cell per panel (each panel at its own best hyperparameters):")
            best = {}
            for pn in PANELS:
                cand = [r for r in g if r["panel"] == pn]
                if cand:
                    best[pn] = max(cand, key=lambda r: r[f"{SEL}_balanced_accuracy"])
            for pn, r in sorted(best.items(), key=lambda kv: -kv[1][f"{SEL}_balanced_accuracy"]):
                print("      " + fmt(r))

            print(f"\n    top 5 cells overall by VAL balanced accuracy:")
            for r in sorted(g, key=lambda r: -r[f"{SEL}_balanced_accuracy"])[:5]:
                print("      " + fmt(r))

            print(f"\n    top 5 cells overall by VAL economic separation t-stat:")
            for r in sorted(g, key=lambda r: -r[f"{SEL}_economic_separation_tstat"])[:5]:
                print("      " + fmt(r))


if __name__ == "__main__":
    main()
