"""하드 게이트(REJECTED) 다음 단계: chop_prob로 포지션 '크기'만 조절하는 소프트 버전.

하드 게이트는 모델이 이미 좋다고 판단한 신호를 통째로 버려서 실패했다 (정보 중복 파괴).
이 버전은 진입 자체는 그대로 두고(모델의 판단을 존중), notional만
size_multiplier = max(floor, 1 - chop_prob) 로 줄인다 -- 레짐이 나쁠수록 작게 베팅,
좋을수록 원래 크기 그대로. 완전 재학습 없이 기존 학습된 신호를 존중하면서 레짐 정보를
사이징에 반영하는 절충안.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_fresh_window_20260713 as fw  # noqa: E402
import dataclasses  # noqa: E402

OUT_DIR = ROOT / "data/research"
NEW_END = "2026-07-13"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"

# floor = minimum size multiplier even at chop_prob=1.0 (0.0 = can go to zero, i.e. pure linear dampening)
VARIANTS = [
    {"tag": "linear_floor0", "floor": 0.0},
    {"tag": "linear_floor0.3", "floor": 0.3},
    {"tag": "linear_floor0.5", "floor": 0.5},
]


def run_variant(floor: float | None):
    native = fw.native
    eth_retest = fw.eth_retest

    _orig_load_frame_current = eth_retest.load_frame_current
    _orig_candidate_for_asset = native._candidate_for_asset

    def _patched_load(start: str, end: str) -> pd.DataFrame:
        return _orig_load_frame_current(start, NEW_END)

    def _softsized_candidate(world, asset, ts):
        c = _orig_candidate_for_asset(world, asset, ts)
        if c is None or floor is None or asset != "eth":
            return c
        aw = world[asset]
        i = aw["ts_to_i"].get(ts)
        if i is None or i >= len(aw["frame"]):
            return c
        chop = aw["frame"][CHOP_COL].iloc[i]
        if pd.isna(chop):
            return c
        mult = max(float(floor), 1.0 - float(chop))
        new_notional = c.notional * mult
        return dataclasses.replace(c, notional=new_notional, leverage=new_notional / max(c.margin, 1e-12))

    eth_retest.load_frame_current = _patched_load
    native._candidate_for_asset = _softsized_candidate
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}
        world = native._build_world("oos", device)
        metrics, ledger, timeline, diag = fw._replay_concurrent_entry_floor(
            world, device=device, cap_mode="scale",
            asset_shares={"eth": 1.0, "btc": 0.0, "sol": 0.0},
            asset_notional_multipliers={"eth": 1.5, "btc": 1.0, "sol": 1.0},
            enabled_assets=("eth",),
            entry_floor=None,
        )
    finally:
        eth_retest.load_frame_current = _orig_load_frame_current
        native._candidate_for_asset = _orig_candidate_for_asset

    return metrics, ledger


def main():
    report = {"stage": "omega461_regime_softsize", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["method_note"] = (
        "Soft position-sizing dampening by chop_prob (entries NOT vetoed, only notional scaled: "
        "mult = max(floor, 1 - chop_prob)). Same live-matching config (duration-gate off, "
        "ETH notional_multiplier=1.5) and genuine bar-by-bar fresh-forward replay as the hard-gate test."
    )

    print("=== Baseline (reference, same numbers as hard-gate test) ===")
    baseline_metrics, baseline_ledger = run_variant(None)
    print(json.dumps(baseline_metrics["portfolio"], indent=2, default=str))
    report["baseline"] = baseline_metrics["portfolio"]

    variants = {}
    for v in VARIANTS:
        print(f"\n=== Soft-size variant: {v['tag']} (floor={v['floor']}) ===")
        m, ledger = run_variant(v["floor"])
        print(json.dumps(m["portfolio"], indent=2, default=str))
        ledger.to_csv(OUT_DIR / f"omega461_eth_softsize_{v['tag']}_20260719.csv", index=False)
        variants[v["tag"]] = m["portfolio"]

    report["variants"] = variants
    b_pnl, b_mdd = report["baseline"]["pnl"], report["baseline"]["mdd"]
    improved = [tag for tag, m in variants.items() if m["pnl"] > b_pnl and m["mdd"] > b_mdd]
    report["baseline_pnl_mdd"] = [b_pnl, b_mdd]
    report["variants_strictly_improving_both"] = improved
    report["verdict"] = (
        f"ACCEPTED -- {improved} beat baseline on BOTH pnl and mdd"
        if improved else "REJECTED -- no soft-size variant strictly improves both pnl and mdd vs baseline"
    )

    out_json = ROOT / "docs/test_designs_duckdb_live_20260719/results/omega461_regime_softsize_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
