"""통제 실험: chop_prob 정보 없이 그냥 균등하게 notional을 절반(또는 다른 고정비율)으로
줄이면 soft-sizing과 같은 개선이 나오는지 확인. 만약 그렇다면 '레짐 정보'의 가치가 아니라
순수 레버리지 축소(변동성 드래그/Kelly 초과레버리지 문제)일 뿐이라는 뜻.
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

FLAT_MULTIPLIERS = [0.481, 0.3, 0.7]  # 0.481 = actual mean(notional_soft)/mean(notional_base) from prior run


def run_variant(*, flat_mult: float | None, use_chop: bool):
    native = fw.native
    eth_retest = fw.eth_retest
    _orig_load = eth_retest.load_frame_current
    _orig_cand = native._candidate_for_asset

    def _patched_load(start, end):
        return _orig_load(start, NEW_END)

    def _sized_candidate(world, asset, ts):
        c = _orig_cand(world, asset, ts)
        if c is None or asset != "eth":
            return c
        if use_chop:
            aw = world[asset]
            i = aw["ts_to_i"].get(ts)
            if i is None or i >= len(aw["frame"]):
                return c
            chop = aw["frame"][CHOP_COL].iloc[i]
            if pd.isna(chop):
                return c
            mult = max(0.0, 1.0 - float(chop))
        else:
            mult = flat_mult
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
    report = {"stage": "omega461_flatsize_control", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}

    print("=== Chop-based soft-size (reference, should match prior run) ===")
    chop_m, chop_ledger = run_variant(flat_mult=None, use_chop=True)
    print(json.dumps(chop_m["portfolio"], indent=2, default=str))
    report["chop_based_softsize"] = chop_m["portfolio"]

    variants = {}
    for fm in FLAT_MULTIPLIERS:
        print(f"\n=== FLAT multiplier (no chop info): {fm} ===")
        m, ledger = run_variant(flat_mult=fm, use_chop=False)
        print(json.dumps(m["portfolio"], indent=2, default=str))
        variants[f"flat_{fm}"] = m["portfolio"]

    report["flat_multiplier_variants"] = variants
    out_json = ROOT / "docs/test_designs_duckdb_live_20260719/results/omega461_flatsize_control_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
