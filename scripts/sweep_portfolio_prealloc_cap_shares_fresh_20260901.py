#!/usr/bin/env python3
"""A4 cap/asset_shares 재스윕 — fresh 데이터(2026-07-01~08-30)에서.

`docs/model_contracts/portfolio_concurrent_3asset_gate_off_cap_sweep_20260712.md`가 이미 스윕한
바로 그 그리드(total_notional_cap 7점, asset_shares 4조합)를, 그 문서 자신이 "heavily peeked
window" 경고와 함께 요구했던 2026-07+ 구간(2026-08-31 A4 확인 세션이 처음 확보)에 재적용한다.
새 자유파라미터 발명 없음 — 07-12 그리드 그대로 재사용.

07-12 문서의 핵심 발견(오염된 구간에서 재현): 60/25/15(ETH비중 더 높음)가 50/30/20(사용자
요청 우선순위)보다 PnL·MDD 둘 다 더 좋았음("share ranking flips" under gate-off). 이게 진짜
정보인지 아니면 그 문서 자신이 경고한 반복조회 아티팩트였는지가 이번 재스윕의 핵심 질문.

Base config(고정, CURRENT_BASELINE과 동일): duration_gate=off, eth_notional_multiplier=1.5,
btc/sol=1.0. `replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831.py`의
`_replay_concurrent_entry_floor`/`_compute_new_end`/`load_frame_current` 몽키패치를 그대로
재사용(중복 구현 안 함).

fresh_window은 2026-07-01~08-30로 고정(A4 확인 세션이 이미 소진한 바로 그 구간 — 09-30 예약
윈도우를 추가로 더 건드리지 않는다).

research_diagnostic_not_live_wired. trading_bot.py/portfolio_risk.py 변경 없음.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_prealloc_eth15x_fresh_confirmation_20260831 as base  # noqa: E402

native = base.native
eth_retest = base.eth_retest
_replay_concurrent_entry_floor = base._replay_concurrent_entry_floor
_compute_new_end = base._compute_new_end
_json_default = base._json_default
FRESH_CUTOFF = base.FRESH_CUTOFF
ASSET_NOTIONAL_MULT = base.ASSET_NOTIONAL_MULT

OUT_DIR = ROOT / "tmp/causal_regen_20260516/sweep_portfolio_prealloc_cap_shares_fresh_20260901"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_concurrent_3asset_prealloc_cap_shares_sweep_fresh_20260901.md"

# 07-12 gate_off_cap_sweep 그리드 그대로 재사용 (신규 자유파라미터 없음)
CAP_GRID: list[float | None] = [None, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
SHARE_GRID: dict[str, dict[str, float]] = {
    "50_30_20": {"eth": 0.5, "btc": 0.3, "sol": 0.2},
    "40_35_25": {"eth": 0.4, "btc": 0.35, "sol": 0.25},
    "33_33_33": {"eth": 1.0 / 3, "btc": 1.0 / 3, "sol": 1.0 / 3},
    "60_25_15": {"eth": 0.6, "btc": 0.25, "sol": 0.15},
}
BASE_SHARE_NAME = "50_30_20"
BASE_CAP = 3.0


def _run_point(world_val, world_oos, device, *, cap: float | None, shares: dict[str, float]) -> dict[str, Any]:
    cap_mode = "prealloc" if cap is not None else "scale"
    out: dict[str, Any] = {}
    val_metrics, _, _, val_diag = _replay_concurrent_entry_floor(
        world_val, device=device, cap_mode=cap_mode, total_notional_cap=cap,
        asset_shares=shares, asset_notional_multipliers=ASSET_NOTIONAL_MULT,
        enabled_assets=("eth", "sol", "btc"), entry_floor=None,
    )
    out["validation"] = {"metrics": val_metrics, "diagnostics": val_diag}
    for split_name, entry_floor in (("oos_extended", None), ("fresh_window", FRESH_CUTOFF)):
        metrics, _, _, diag = _replay_concurrent_entry_floor(
            world_oos, device=device, cap_mode=cap_mode, total_notional_cap=cap,
            asset_shares=shares, asset_notional_multipliers=ASSET_NOTIONAL_MULT,
            enabled_assets=("eth", "sol", "btc"), entry_floor=entry_floor,
        )
        out[split_name] = {"metrics": metrics, "diagnostics": diag}
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_end = _compute_new_end()

    _orig_load_frame_current = eth_retest.load_frame_current

    def _patched(start: str, end: str) -> pd.DataFrame:  # noqa: ARG001
        return _orig_load_frame_current(start, new_end)

    eth_retest.load_frame_current = _patched
    try:
        device = eth_retest.DEVICE
        native.DURATION_THRESHOLDS = {k: -999.0 for k in native.DURATION_THRESHOLDS}

        print("stage=build_world split=validation", flush=True)
        world_val = native._build_world("validation", device)
        print("stage=build_world split=oos", flush=True)
        world_oos = native._build_world("oos", device)
        print(f"worlds built. oos n={len(world_oos['timestamps'])}", flush=True)

        results: dict[str, Any] = {"cap_sweep": {}, "share_sweep": {}}

        base_shares = SHARE_GRID[BASE_SHARE_NAME]
        print("\n=== cap_sweep (shares fixed at 50/30/20) ===", flush=True)
        for cap in CAP_GRID:
            label = "uncapped" if cap is None else str(cap)
            print(f"  point cap={label}", flush=True)
            results["cap_sweep"][label] = _run_point(world_val, world_oos, device, cap=cap, shares=base_shares)

        print("\n=== share_sweep (cap fixed at 3.0) ===", flush=True)
        for share_name, shares in SHARE_GRID.items():
            if share_name == BASE_SHARE_NAME:
                results["share_sweep"][share_name] = results["cap_sweep"]["3.0"]
                continue
            print(f"  point shares={share_name}", flush=True)
            results["share_sweep"][share_name] = _run_point(world_val, world_oos, device, cap=BASE_CAP, shares=shares)
    finally:
        eth_retest.load_frame_current = _orig_load_frame_current

    report = {
        "method": "sweep_portfolio_prealloc_cap_shares_fresh_20260901",
        "base_config": "duration_gate=off, eth_notional_multiplier=1.5, btc/sol_notional_multiplier=1.0",
        "grid_source": "docs/model_contracts/portfolio_concurrent_3asset_gate_off_cap_sweep_20260712.md (reused verbatim, not re-invented)",
        "cap_grid": [("uncapped" if c is None else c) for c in CAP_GRID],
        "share_grid": SHARE_GRID,
        "fresh_cutoff": str(FRESH_CUTOFF),
        "data_extended_through": new_end,
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    def row(label: str, d: dict) -> str:
        m = d["fresh_window"]["metrics"]["portfolio"]
        mv = d["validation"]["metrics"]["portfolio"]
        mo = d["oos_extended"]["metrics"]["portfolio"]
        return (f"{label:12s} fresh: PnL={m['pnl']:8.2f}% MDD={m['mdd']:7.2f}% WR={m['wr']:6.1%} n={m['trades']:3d} | "
                f"oos_ext: PnL={mo['pnl']:8.2f}% MDD={mo['mdd']:7.2f}% | val: PnL={mv['pnl']:8.2f}% MDD={mv['mdd']:7.2f}%")

    print("\n=== cap_sweep summary ===")
    for label, d in results["cap_sweep"].items():
        print(row(f"cap={label}", d))
    print("\n=== share_sweep summary ===")
    for label, d in results["share_sweep"].items():
        print(row(f"shares={label}", d))

    print(f"\nreport written: {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
