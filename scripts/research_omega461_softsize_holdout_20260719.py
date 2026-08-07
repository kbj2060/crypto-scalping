"""선형(floor=0) 소프트사이징이 최근 구간(2026-05-01~07-12)에서도 베이스라인을 이기는지 확인.

주의: 이건 완전한 블라인드 홀드아웃이 아니다 -- 이 방식(선형, floor=0)은 애초에
전체 구간(2026-01~07) 성과를 보고 4개 후보 중 골랐으므로, 이 최근 슬라이스도 그
선택 과정에 이미 포함돼 있었다. 진짜 편향 없는 확인은 2026-07-13 이후 진짜 새 데이터가
필요하다 (아직 존재하지 않음). 이번 체크는 "최근 슬라이스에서도 견고한가"를 보는
약한 확인일 뿐, 최종 승격 근거가 될 수 없다.
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

OUT_DIR = ROOT / "data/research"
NEW_END = "2026-07-13"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"
HOLDOUT_START = pd.Timestamp("2026-05-01")


def run_variant(use_softsize: bool, entry_floor):
    native = fw.native
    eth_retest = fw.eth_retest
    _orig_load = eth_retest.load_frame_current
    _orig_cand = native._candidate_for_asset

    def _patched_load(start, end):
        return _orig_load(start, NEW_END)

    def _sized_candidate(world, asset, ts):
        c = _orig_cand(world, asset, ts)
        if c is None or not use_softsize or asset != "eth":
            return c
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
    report = {"stage": "omega461_softsize_holdout_slice", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["caveat"] = (
        "NOT a blind holdout -- the linear/floor0 formula was already selected by looking at "
        "performance across the full 2026-01~07 window, which includes this slice. This only "
        "checks recent-slice robustness, not genuine unseen generalization."
    )
    report["holdout_start"] = str(HOLDOUT_START)

    print("=== Baseline, entries restricted to 2026-05-01+ ===")
    b_m, b_ledger = run_variant(False, HOLDOUT_START)
    print(json.dumps(b_m["portfolio"], indent=2, default=str))
    report["baseline_holdout_slice"] = b_m["portfolio"]

    print("\n=== Soft-size (linear, floor0), entries restricted to 2026-05-01+ ===")
    s_m, s_ledger = run_variant(True, HOLDOUT_START)
    print(json.dumps(s_m["portfolio"], indent=2, default=str))
    report["softsize_holdout_slice"] = s_m["portfolio"]

    b_pnl, b_mdd = b_m["portfolio"]["pnl"], b_m["portfolio"]["mdd"]
    s_pnl, s_mdd = s_m["portfolio"]["pnl"], s_m["portfolio"]["mdd"]
    report["verdict"] = (
        "HOLDS UP on recent slice (both pnl and mdd improved)"
        if (s_pnl > b_pnl and s_mdd > b_mdd) else
        "DOES NOT clearly hold up on recent slice -- mixed or reversed result"
    )

    out_json = ROOT / "docs/test_designs_duckdb_live_20260719/results/omega461_softsize_holdout_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
