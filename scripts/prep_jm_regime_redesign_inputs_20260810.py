"""Stage 1 of the JM-only regime3 redesign: build the cached, already-scaled model inputs.

The sweep runs hundreds of JM fits across 12 (panel, scaler) combinations per asset. Re-reading a
220MB year CSV inside every worker would dominate both wall time and memory, and -- more
importantly -- would risk the scaler being refit on something other than the 2024 window. So the
scaling contract is pinned exactly once, here:

  * every feature panel is built causally from the raw frame,
  * the scaler and the NaN-fill medians are fit on the 2024 file ONLY,
  * 2024/2025/2026 are all transformed with those frozen 2024 statistics,
  * the frozen column list for the data-derived `wide24_decorr` panel is chosen on 2024 only.

Labels are cached under both bases (see the lib's label-basis note), together with close prices
and timestamps so the sweep can slice the VAL/OOS windows after decoding, never before.

Outputs: data/ensemble/reports/jm_redesign_20260810/cache/{asset}__{panel}__{scaler}.npz
         data/ensemble/reports/jm_redesign_20260810/prep_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, FIT_YEAR, LABEL_BASES, LABEL_CONFIGS, LABEL_MODE, LABEL_REF_ASSET, PANEL_DIMS,
    SOURCES, _num, _read, apply_scale, build_panel, choose_decorr_cols, fit_scale, labels_for,
    quantile_matched_label_config, reference_label_quantiles,
)

OUT_DIR = ROOT / "data/ensemble/reports/jm_redesign_20260810"
CACHE_DIR = OUT_DIR / "cache"
SCALERS = ("robust", "standard")
PANELS = ("jm6", "jm9", "jm9_perp", "wide24_decorr", "state12", "wide24")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"panels": list(PANELS), "scalers": list(SCALERS), "assets": {}}

    ref_fit = _read(SOURCES[LABEL_REF_ASSET][FIT_YEAR])
    ref_q = reference_label_quantiles(ref_fit)
    report["label_reference_asset"] = LABEL_REF_ASSET
    report["label_reference_quantiles"] = ref_q
    report["label_frozen_config"] = dict(LABEL_CONFIGS[LABEL_MODE])
    print(f"[label] reference quantile positions from {LABEL_REF_ASSET} {FIT_YEAR}: "
          + ", ".join(f"{k}={v:.4f}" for k, v in ref_q.items()))

    for asset in SOURCES:
        print(f"\n=== {asset} ===")
        frames = {}
        for year, path in SOURCES[asset].items():
            frames[year] = _read(path)
            print(f"  loaded {year}: {len(frames[year]):,} rows")
        fit_frame = frames[FIT_YEAR]

        cfgs = {
            "frozen": dict(LABEL_CONFIGS[LABEL_MODE]),
            "qmatched": quantile_matched_label_config(fit_frame, ref_q),
        }
        labels = {b: {y: labels_for(f, cfgs[b]) for y, f in frames.items()} for b in LABEL_BASES}
        asset_rep: dict = {"label_configs": cfgs, "label_balance": {}, "panels": {}}
        for b in LABEL_BASES:
            asset_rep["label_balance"][b] = {}
            for y, yv in labels[b].items():
                share = np.bincount(yv, minlength=3) / len(yv)
                asset_rep["label_balance"][b][y] = {CLASSES3[i]: float(share[i]) for i in range(3)}
                print(f"  label[{b}] {y}: " + " ".join(f"{CLASSES3[i]}={share[i]:.3f}" for i in range(3)))

        decorr_cols = choose_decorr_cols(fit_frame)
        asset_rep["wide24_decorr_cols"] = decorr_cols
        print(f"  wide24_decorr kept {len(decorr_cols)}/24 columns")

        closes = {y: _num(f, "close").ffill().bfill().to_numpy(dtype=np.float64) for y, f in frames.items()}
        stamps = {y: f["timestamp"].to_numpy(dtype="datetime64[ns]") for y, f in frames.items()}

        for panel in PANELS:
            dc = decorr_cols if panel == "wide24_decorr" else None
            panels_by_year = {y: build_panel(f, panel, decorr_cols=dc) for y, f in frames.items()}
            cols = panels_by_year[FIT_YEAR][1]
            asset_rep["panels"][panel] = {"dim": len(cols), "cols": cols}
            print(f"  panel {panel}: d={len(cols)}")
            for scaler_kind in SCALERS:
                x_fit, scaler, medians, clip_bounds = fit_scale(panels_by_year[FIT_YEAR][0], scaler_kind)
                payload = {
                    "cols": np.asarray(cols, dtype=object),
                    "dim": np.int64(len(cols)),
                }
                for y in frames:
                    xv = x_fit if y == FIT_YEAR else apply_scale(
                        panels_by_year[y][0], scaler, medians, clip_bounds)
                    payload[f"x_{y}"] = xv.astype(np.float64)
                    payload[f"close_{y}"] = closes[y]
                    payload[f"ts_{y}"] = stamps[y]
                    for b in LABEL_BASES:
                        payload[f"y_{b}_{y}"] = labels[b][y].astype(np.int8)
                np.savez_compressed(CACHE_DIR / f"{asset}__{panel}__{scaler_kind}.npz", **payload)
                joblib.dump({"scaler": scaler, "medians": medians, "cols": cols,
                             "clip_bounds": clip_bounds},
                            CACHE_DIR / f"{asset}__{panel}__{scaler_kind}__scaler.joblib")
        report["assets"][asset] = asset_rep

    (OUT_DIR / "prep_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nprep report -> {OUT_DIR / 'prep_report.json'}")
    print(f"cache -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
