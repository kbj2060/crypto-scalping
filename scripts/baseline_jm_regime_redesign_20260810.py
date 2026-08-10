"""Incumbent baselines for the JM-only regime3 redesign, scored on the SAME windows and metrics.

Without this the sweep's numbers are uninterpretable: "VAL balanced accuracy 0.86" only means
something next to what the models actually in (or near) production score on the identical VAL/OOS
slices, against the identical label. Every baseline here is read from its already-emitted
per-bar probability CSV, so nothing is refitted and nothing can drift.

Baselines:
  live HMM           the 12-state sticky Gaussian HMM on wide24 that is wired live for BTC
  jm lambda=4        the 2026-08-09 HMM->JM swap (BTC shadow-closed, ETH shadow-only)
  jm lambda=2        the 2026-08-10 BTC lambda refit
Scored under both label bases so the comparison against the redesign is like-for-like.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_BASES, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM,
    SOURCES, _num, _read, labels_for, quantile_matched_label_config, reference_label_quantiles,
    slice_window, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402

SUP = ROOT / "data/ensemble/supervised"
BASELINES = {
    "btc": {
        "live_hmm_wide24": {
            y: SUP / f"btc_regime3_current_hmm_sensitive_wide24_20260708/btc_features_{y}_regime3_current_sensitive_hmm_wide24.csv"
            for y in ("2024", "2025", "2026")},
        "jm_lambda4_20260809": {
            y: SUP / f"btc_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv"
            for y in ("2024", "2025", "2026")},
        "jm_lambda2_20260810": {
            y: SUP / f"btc_regime3_current_hmm_jmlam2_20260810_{y}_maskedname.csv"
            for y in ("2024", "2025", "2026")},
    },
    "eth": {
        "jm_lambda4_20260809": {
            y: SUP / f"eth_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv"
            for y in ("2024", "2025", "2026")},
    },
}


def _prob_columns(frame: pd.DataFrame) -> list[str] | None:
    for prefix in (f"{PREFIX_STEM}_wide24_", f"{PREFIX_STEM}_hmm_wide24_"):
        cols = [f"{prefix}{c}_prob" for c in CLASSES3]
        if all(c in frame.columns for c in cols):
            return cols
    cands = [c for c in frame.columns if c.endswith("_bull_prob")]
    if cands:
        stem = cands[0][: -len("bull_prob")]
        cols = [f"{stem}{c}_prob" for c in CLASSES3]
        if all(c in frame.columns for c in cols):
            return cols
    return None


def main() -> None:
    ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
    report: dict = {}
    for asset, models in BASELINES.items():
        frames = {y: _read(p) for y, p in SOURCES[asset].items()}
        cfgs = {"frozen": dict(LABEL_CONFIGS[LABEL_MODE]),
                "qmatched": quantile_matched_label_config(frames[FIT_YEAR], ref_q)}
        labels = {b: {y: labels_for(f, cfgs[b]) for y, f in frames.items()} for b in LABEL_BASES}
        closes = {y: _num(f, "close").ffill().bfill().to_numpy() for y, f in frames.items()}
        report[asset] = {}
        print(f"\n=== {asset.upper()} incumbent baselines ===")
        for name, paths in models.items():
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                print(f"  [skip] {name}: missing {missing[0]}")
                continue
            entry: dict = {}
            for basis in LABEL_BASES:
                windows = {}
                for wname, (yr, start, end) in EVAL_WINDOWS.items():
                    pf = _read(paths[yr])
                    cols = _prob_columns(pf)
                    if cols is None:
                        raise ValueError(f"{paths[yr]} has no recognisable class-probability columns")
                    # align on timestamp so a baseline emitted on a different row set is not
                    # silently compared against mismatched labels
                    base = frames[yr][["timestamp"]].copy()
                    base["_row"] = np.arange(len(base))
                    merged = base.merge(pf[["timestamp"] + cols], on="timestamp", how="inner")
                    rows = merged["_row"].to_numpy()
                    m = slice_window(frames[yr]["timestamp"], start, end)
                    keep = m[rows]
                    rows, proba = rows[keep], merged[cols].to_numpy()[keep]
                    windows[wname] = window_metrics(np.argmax(proba, axis=1).astype(np.int64),
                                                    labels[basis][yr][rows], closes[yr][rows])
                    windows[wname]["aligned_rows"] = int(len(rows))
                entry[basis] = windows
                v, o = windows["val"], windows["oos"]
                print(f"  {name:>22} [{basis:>8}] VAL bal={v['balanced_accuracy']:.4f} "
                      f"run={v['median_run_bars']:>4.0f} cov={v['min_class_coverage']:.3f} "
                      f"sep_t={v['economic_separation_tstat']:>6.2f} | "
                      f"OOS bal={o['balanced_accuracy']:.4f} run={o['median_run_bars']:>4.0f} "
                      f"sep_t={o['economic_separation_tstat']:>6.2f}  (n_val={v['aligned_rows']})")
            report[asset][name] = entry
    path = OUT_DIR / "baseline_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nbaseline report -> {path}")


if __name__ == "__main__":
    main()
