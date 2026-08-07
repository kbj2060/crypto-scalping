"""소프트사이징 정제 라운드: quadratic 감쇠, threshold-ramp, chop+stability 결합
세 가지를 베이스라인/균등배분 대조군과 함께 비교.

전부 진짜 fresh-forward bar-by-bar, 라이브 설정(duration-gate off, notional_mult=1.5) 유지.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dataclasses

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_fresh_window_20260713 as fw  # noqa: E402

OUT_DIR = ROOT / "data/research"
NEW_END = "2026-07-13"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"
STAB_PATH = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv"
STAB_COL = "regime3_cmamba_h6_sidecar_stability_score"


def _load_stability():
    df = pd.read_csv(STAB_PATH, usecols=["timestamp", STAB_COL])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


SIZING_FNS = {
    "linear_floor0": lambda chop, stab: max(0.0, 1.0 - chop),
    "quadratic_floor0": lambda chop, stab: max(0.0, 1.0 - chop) ** 2,
    "ramp_0.2_0.6": lambda chop, stab: 1.0 if chop < 0.2 else (0.0 if chop >= 0.6 else 1.0 - (chop - 0.2) / 0.4),
    "chop_x_stability": lambda chop, stab: max(0.0, 1.0 - chop) * (stab if stab is not None else 1.0),
}


def run_variant(fn_name: str | None, stability_frame: pd.DataFrame | None):
    fn = SIZING_FNS.get(fn_name) if fn_name else None
    native = fw.native
    eth_retest = fw.eth_retest
    _orig_load = eth_retest.load_frame_current
    _orig_cand = native._candidate_for_asset

    stab_ts_to_val = None
    if stability_frame is not None:
        stab_ts_to_val = dict(zip(stability_frame["timestamp"], stability_frame[STAB_COL]))

    def _patched_load(start, end):
        return _orig_load(start, NEW_END)

    def _sized_candidate(world, asset, ts):
        c = _orig_cand(world, asset, ts)
        if c is None or fn is None or asset != "eth":
            return c
        aw = world[asset]
        i = aw["ts_to_i"].get(ts)
        if i is None or i >= len(aw["frame"]):
            return c
        chop = aw["frame"][CHOP_COL].iloc[i]
        if pd.isna(chop):
            return c
        stab = None
        if stab_ts_to_val is not None:
            stab = stab_ts_to_val.get(pd.Timestamp(ts))
            if stab is None or pd.isna(stab):
                stab = 1.0
        mult = fn(float(chop), stab)
        mult = max(0.0, min(1.0, float(mult)))
        new_notional = c.notional * mult
        return dataclasses.replace(c, notional=new_notional, leverage=new_notional / max(c.margin, 1e-12))

    eth_retest.load_frame_current = _patched_load
    native._candidate_for_asset = _sized_candidate
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}
        world = native._build_world("oos", device)
        metrics, ledger, timeline, diag = fw._replay_concurrent_entry_floor(
            world, device=device, cap_mode="scale",
            asset_shares={"eth": 1.0, "btc": 0.0, "sol": 0.0},
            asset_notional_multipliers={"eth": 1.5, "btc": 1.0, "sol": 1.0},
            enabled_assets=("eth",), entry_floor=None,
        )
    finally:
        eth_retest.load_frame_current = _orig_load
        native._candidate_for_asset = _orig_cand
    return metrics, ledger


def main():
    report = {"stage": "omega461_softsize_refine", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    stability_frame = _load_stability()

    print("=== Baseline ===")
    b_m, _ = run_variant(None, None)
    print(json.dumps(b_m["portfolio"], indent=2, default=str))
    report["baseline"] = b_m["portfolio"]

    variants = {}
    for tag in SIZING_FNS:
        needs_stab = tag == "chop_x_stability"
        print(f"\n=== {tag} ===")
        m, ledger = run_variant(tag, stability_frame if needs_stab else None)
        print(json.dumps(m["portfolio"], indent=2, default=str))
        ledger.to_csv(OUT_DIR / f"omega461_eth_refine_{tag}_20260719.csv", index=False)
        variants[tag] = m["portfolio"]

    report["variants"] = variants
    b_pnl, b_mdd = report["baseline"]["pnl"], report["baseline"]["mdd"]
    improved = [tag for tag, m in variants.items() if m["pnl"] > b_pnl and m["mdd"] > b_mdd]
    report["variants_strictly_improving_both_vs_baseline"] = improved
    best_tag = max(variants, key=lambda t: variants[t]["pnl"])
    report["best_by_pnl"] = {"tag": best_tag, **variants[best_tag]}

    out_json = ROOT / "docs/test_designs_duckdb_live_20260719/results/omega461_softsize_refine_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
