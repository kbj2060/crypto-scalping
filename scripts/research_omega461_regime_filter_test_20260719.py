"""사용자 요청: Omega4.6.1(ETH, 5분봉)에 Sigma6 스타일 "장기 레짐 관점"을 이식해서 테스트.

Sigma6가 Sigma5(추세추종)에 추가했던 것과 동일한 아이디어를 재사용:
Regime3 'current' HMM nowcast의 chop_prob가 높으면(횡보장) 신규 진입을 아예 안 함.
Omega의 ETH 5분봉 프레임(ext_frame)에는 이미 Sigma6가 쓰는 것과 동일한 소스 파일
(regime3_current_hmm_sensitive_balancedish_20260530 wide24 overlay)이 5분봉 해상도로
병합되어 있음을 확인함 -- 추가 데이터 병합 불필요.

방식: native._candidate_for_asset()를 몽키패치해서 ETH 신규 진입 시점에만 chop_prob 게이트를
추가. 기존 포지션의 청산 로직(_try_close)은 전혀 건드리지 않음 -- Sigma6 원본과 동일하게
"진입만 막고 청산은 그대로" 설계.

라이브와 동일한 설정(duration-gate off, ETH notional_multiplier=1.5) 위에 레짐 필터만 추가.
진짜 fresh-forward bar-by-bar 실행 (저장된 원장 재사용 없음).
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

OUT_DIR = ROOT / "data/research"
NEW_END = "2026-07-13"
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"

# grid matching Sigma6's own sweep (reg_thr candidates); no val/OOS split re-selection here --
# reusing Sigma6's already-published winning thresholds as a direct transplant, not re-tuned.
REG_THR_GRID = [0.34, 0.42, 0.50]


def run_variant(reg_thr: float | None):
    native = fw.native
    eth_retest = fw.eth_retest

    _orig_load_frame_current = eth_retest.load_frame_current
    _orig_candidate_for_asset = native._candidate_for_asset

    def _patched_load(start: str, end: str) -> pd.DataFrame:
        return _orig_load_frame_current(start, NEW_END)

    def _gated_candidate(world, asset, ts):
        if reg_thr is not None and asset == "eth":
            aw = world[asset]
            i = aw["ts_to_i"].get(ts)
            if i is not None and i < len(aw["frame"]):
                chop = aw["frame"][CHOP_COL].iloc[i]
                if pd.notna(chop) and float(chop) >= reg_thr:
                    return None  # chop regime -- block new entries only, exits unaffected
        return _orig_candidate_for_asset(world, asset, ts)

    eth_retest.load_frame_current = _patched_load
    native._candidate_for_asset = _gated_candidate
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

    return metrics, ledger, world


def main():
    report = {"stage": "omega461_regime_filter_transplant", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["method_note"] = (
        "Genuine bar-by-bar fresh-forward replay (native._candidate_for_asset monkeypatched to gate "
        "ETH entries on regime3_current_sensitive_wide24_chop_prob, same source Sigma6 uses, already "
        "merge-asof'd onto the 5m ETH frame at native resolution -- no new data merging needed). "
        "Live-matching config preserved (duration-gate off, ETH notional_multiplier=1.5). "
        "Exits are NOT gated -- only new-entry candidate generation, matching Sigma6's own design."
    )

    print("=== Baseline: no regime filter (re-run for apples-to-apples comparison) ===")
    baseline_metrics, baseline_ledger, world = run_variant(None)
    print(json.dumps(baseline_metrics, indent=2, default=str))
    baseline_ledger.to_csv(OUT_DIR / "omega461_eth_baseline_freshforward_20260719.csv", index=False)
    report["baseline_no_filter"] = baseline_metrics

    variants = {}
    for reg_thr in REG_THR_GRID:
        print(f"\n=== Regime filter: chop_prob < {reg_thr} ===")
        m, ledger, _ = run_variant(reg_thr)
        print(json.dumps(m, indent=2, default=str))
        ledger.to_csv(OUT_DIR / f"omega461_eth_regimefilter_{str(reg_thr).replace('.', '')}_20260719.csv", index=False)
        variants[f"reg_thr_{reg_thr}"] = m

    report["regime_filter_variants"] = variants

    # verdict: any variant beat baseline on BOTH pnl and mdd?
    b_pnl, b_mdd = baseline_metrics["portfolio"]["pnl"], baseline_metrics["portfolio"]["mdd"]
    improved = []
    for tag, m in variants.items():
        p, d = m["portfolio"]["pnl"], m["portfolio"]["mdd"]
        if p > b_pnl and d > b_mdd:  # mdd less negative = better
            improved.append(tag)
    report["baseline_pnl_mdd"] = [b_pnl, b_mdd]
    report["variants_strictly_improving_both"] = improved
    report["verdict"] = (
        f"ACCEPTED -- {len(improved)} regime-filter variant(s) beat baseline on BOTH pnl and mdd: {improved}"
        if improved else
        "REJECTED -- no regime-filter variant strictly improves both pnl and mdd vs baseline"
    )

    out_json = OUT_DIR.parent.parent / "docs/test_designs_duckdb_live_20260719/results/omega461_regime_filter_transplant_20260719.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("\nWROTE", out_json)
    print(json.dumps({k: v for k, v in report.items() if k not in ("regime_filter_variants",)}, indent=2, default=str))
    print(json.dumps(variants, indent=2, default=str))


if __name__ == "__main__":
    main()
