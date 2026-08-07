"""핵심 질문: chop 기반 소프트사이징의 '추가 가치'(균등배분 대비)가 최근 홀드아웃
구간(2026-05-01+)에서도 유지되는가, 아니면 1~4월 좋은 구간에만 있던 우연인가?

이미 확인: 전체 구간에서는 균등0.481배(+70.87%/-17.23%) < chop기반(+108.13%/-16.46%).
하지만 이 '추가분'이 홀드아웃에서도 나오는지는 아직 검정 안 함.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import dataclasses

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_fresh_window_20260713 as fw  # noqa: E402

NEW_END = "2026-07-13"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"
HOLDOUT_START = pd.Timestamp("2026-05-01")
FLAT_MULT = 0.481  # matches mean notional ratio of the chop-based approach, for apples-to-apples


def run_variant(mode: str, entry_floor):
    native = fw.native
    eth_retest = fw.eth_retest
    _orig_load = eth_retest.load_frame_current
    _orig_cand = native._candidate_for_asset

    def _patched_load(start, end):
        return _orig_load(start, NEW_END)

    def _sized_candidate(world, asset, ts):
        c = _orig_cand(world, asset, ts)
        if c is None or asset != "eth" or mode == "baseline":
            return c
        if mode == "flat":
            mult = FLAT_MULT
        else:  # chop
            aw = world[asset]
            i = aw["ts_to_i"].get(ts)
            if i is None or i >= len(aw["frame"]):
                return c
            chop = aw["frame"][CHOP_COL].iloc[i]
            if pd.isna(chop):
                return c
            mult = max(0.0, 1.0 - float(chop))
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
            enabled_assets=("eth",), entry_floor=entry_floor,
        )
    finally:
        eth_retest.load_frame_current = _orig_load
        native._candidate_for_asset = _orig_cand
    return metrics, ledger


def main():
    report = {"stage": "holdout_chop_vs_flat", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    print(f"=== HOLDOUT slice only (entries >= {HOLDOUT_START}) ===\n")

    print("--- baseline ---")
    b_m, _ = run_variant("baseline", HOLDOUT_START)
    print(json.dumps(b_m["portfolio"], indent=2, default=str))

    print("\n--- flat multiplier (no chop info) ---")
    f_m, _ = run_variant("flat", HOLDOUT_START)
    print(json.dumps(f_m["portfolio"], indent=2, default=str))

    print("\n--- chop-based soft-sizing ---")
    c_m, _ = run_variant("chop", HOLDOUT_START)
    print(json.dumps(c_m["portfolio"], indent=2, default=str))

    report["baseline"] = b_m["portfolio"]
    report["flat_control"] = f_m["portfolio"]
    report["chop_based"] = c_m["portfolio"]
    extra_pnl = c_m["portfolio"]["pnl"] - f_m["portfolio"]["pnl"]
    extra_mdd = c_m["portfolio"]["mdd"] - f_m["portfolio"]["mdd"]
    report["chop_extra_value_vs_flat_on_holdout"] = {"extra_pnl_pp": extra_pnl, "extra_mdd_pp": extra_mdd}
    print(f"\n=== Chop's extra value beyond flat-multiplier, ON HOLDOUT SLICE: PnL {extra_pnl:+.1f}pp, MDD {extra_mdd:+.1f}pp ===")

    out_json = ROOT / "docs/test_designs_duckdb_live_20260719/results/omega461_holdout_chop_vs_flat_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)


if __name__ == "__main__":
    main()
