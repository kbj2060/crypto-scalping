"""Consolidated decision table for the JM-only regime3 redesign.

Merges everything onto one comparison, per asset, all scored on identical windows, gates and
metrics:

  incumbent   the live 12-state HMM, and the 2026-08-09/08-10 JM lambda=4 / lambda=2 builds
  panel       six hand-authored feature panels x scaler x K x lambda_per_dim x temperature
  sparse      sparse jump model -- learned feature weights over all 130 candidates (unsupervised
              between-cluster criterion, the method's own)
  ranked      supervised ANOVA-F / mRMR nested top-m panels over the same 130 candidates

Gates (VAL only): median run >= 12 bars, min class coverage >= 5%. Selection metric: VAL balanced
accuracy against the ADX/slope/BB rule label. Economic separation is printed beside every number
because agreement with a rule label is not evidence of a tradeable regime split -- and on these
assets the two criteria do not pick the same winner.

The VAL->OOS rank-transfer figures decide how much any of this can be trusted: a criterion whose
VAL ranking does not predict its own OOS ranking cannot be used to choose a model, whatever the
headline number says.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    LABEL_BASES, MIN_CLASS_COVERAGE, MIN_MEDIAN_RUN_BARS, SELECTION_WINDOW,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402

SEL = SELECTION_WINDOW
SOURCES_ = {"panel": "sweep_{a}_report.json",
            "sparse": "sparse_{a}_report.json",
            "ranked": "ranked_{a}_report.json"}


def load_cells(asset: str) -> dict[str, list[dict]]:
    out = {}
    for family, tmpl in SOURCES_.items():
        p = OUT_DIR / tmpl.format(a=asset)
        if p.exists():
            out[family] = json.loads(p.read_text())["cells"]
    return out


def gate(rows: list[dict], basis: str) -> list[dict]:
    return [r for r in rows
            if r["label_basis"] == basis
            and r[f"{SEL}_median_run_bars"] >= MIN_MEDIAN_RUN_BARS
            and r[f"{SEL}_min_class_coverage"] >= MIN_CLASS_COVERAGE]


def describe(family: str, r: dict) -> str:
    if family == "panel":
        return f"{r['panel']}/{r['scaler']} k={r['k']} lpd={r['lambda_per_dim']:g} T={r['temperature_ratio']:g}"
    if family == "sparse":
        return (f"sparseJM/{r['scaler']} k={r['k']} lam={r['lambda']:g} kappa={r['kappa']:g} "
                f"({r['n_selected']} feats) T={r['temperature_ratio']:g}")
    return (f"{r['ranking']}top{r['top_m']}/{r['scaler']} k={r['k']} "
            f"lpd={r['lambda_per_dim']:g} T={r['temperature_ratio']:g}")


def line(tag: str, r: dict) -> str:
    return (f"    {tag:<58} VAL bal={r[f'{SEL}_balanced_accuracy']:.4f} "
            f"run={r[f'{SEL}_median_run_bars']:>4.0f} cov={r[f'{SEL}_min_class_coverage']:.3f} "
            f"sep_t={r[f'{SEL}_economic_separation_tstat']:>6.2f} | "
            f"OOS bal={r['oos_balanced_accuracy']:.4f} "
            f"run={r['oos_median_run_bars']:>4.0f} sep_t={r['oos_economic_separation_tstat']:>6.2f}")


def main() -> None:
    base = json.loads((OUT_DIR / "baseline_report.json").read_text())
    summary: dict = {}
    for asset in ("btc", "eth"):
        fams = load_cells(asset)
        if not fams:
            continue
        print("\n" + "=" * 124)
        print(f"{asset.upper()}   consolidated  (fit 2024 | select {SEL} 2025-09..12 | OOS 2026-01..03 held out)")
        print("=" * 124)
        summary[asset] = {}
        for basis in LABEL_BASES:
            print(f"\n--- label basis: {basis}")
            print("    INCUMBENT")
            for name, entry in base.get(asset, {}).items():
                w = entry[basis]
                r = {f"{SEL}_balanced_accuracy": w["val"]["balanced_accuracy"],
                     f"{SEL}_median_run_bars": w["val"]["median_run_bars"],
                     f"{SEL}_min_class_coverage": w["val"]["min_class_coverage"],
                     f"{SEL}_economic_separation_tstat": w["val"]["economic_separation_tstat"],
                     "oos_balanced_accuracy": w["oos"]["balanced_accuracy"],
                     "oos_median_run_bars": w["oos"]["median_run_bars"],
                     "oos_economic_separation_tstat": w["oos"]["economic_separation_tstat"]}
                print(line(name, r))

            entry_out = {}
            print("    REDESIGN -- best gated cell per search family, by VAL balanced accuracy")
            for family, rows in fams.items():
                g = gate(rows, basis)
                if not g:
                    print(f"    {family:<58} no cell passes the gates")
                    continue
                best = max(g, key=lambda r: r[f"{SEL}_balanced_accuracy"])
                print(line(f"{family}: {describe(family, best)}", best))
                bsep = max(g, key=lambda r: r[f"{SEL}_economic_separation_tstat"])
                rho_b = spearmanr([r[f"{SEL}_balanced_accuracy"] for r in g],
                                  [r["oos_balanced_accuracy"] for r in g])
                rho_s = spearmanr([r[f"{SEL}_economic_separation_tstat"] for r in g],
                                  [r["oos_economic_separation_tstat"] for r in g])
                entry_out[family] = {
                    "n_gated": len(g), "best_by_balanced_accuracy": best,
                    "best_by_economic_separation": bsep,
                    "val_oos_rank_rho_balanced_accuracy": float(rho_b.statistic),
                    "val_oos_rank_rho_economic_separation": float(rho_s.statistic),
                }
            print("    REDESIGN -- best gated cell per search family, by VAL economic separation")
            for family, rows in fams.items():
                g = gate(rows, basis)
                if g:
                    print(line(f"{family}: {describe(family, entry_out[family]['best_by_economic_separation'])}",
                               entry_out[family]["best_by_economic_separation"]))
            print("    VAL->OOS rank transfer over gated cells (does the criterion choose?)")
            for family, e in entry_out.items():
                print(f"      {family:<10} n={e['n_gated']:<5} "
                      f"balanced_accuracy rho={e['val_oos_rank_rho_balanced_accuracy']:+.3f}   "
                      f"economic_separation rho={e['val_oos_rank_rho_economic_separation']:+.3f}")
            summary[asset][basis] = entry_out

    (OUT_DIR / "consolidated_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nconsolidated -> {OUT_DIR / 'consolidated_summary.json'}")


if __name__ == "__main__":
    main()
