#!/usr/bin/env python3
"""ETH **하네스 대조군** -- BTC 경제성게이트가 0/672로 전멸한 게 진짜인지 코드 결함인지 가른다.

BTC 게이트(`gate_btc_evidence_signals_trailing_economics_20260902.py`)가 7신호 96셀 전부
VAL+OOS 동시양수 0을 냈다. 결과가 너무 완전해서 하네스 자체를 의심해야 한다.

**같은 그리드·같은 비용·같은 시뮬레이터**로 ETH를 돌린다. ETH는 이미 통과 사실이 알려져 있다:

    liquidity_sweep_topdown  SL=4.0 ARM=2.0 Trail=0.1 -> VAL +10.70bp / OOS +14.49bp
    orthogonal_combo         SL=4.0 ARM=0.5 Trail=0.1 -> VAL  +9.36bp / OOS +15.13bp
    smt_divergence           SL=4.0 ARM=2.0 Trail=0.1 -> VAL  +7.00bp / OOS  +6.18bp

이 셀들이 재현되면 BTC의 0/672는 **진짜**다. 재현 안 되면 하네스가 깨진 것이다.
ETH fires CSV/HORIZON은 `research_evidence_signals_costgate_flip_audit_20260901.py`에서 그대로 가져왔다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_S = importlib.util.spec_from_file_location(
    "btcgate", ROOT / "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)
run_grid, SL_GRID, ARM_GRID, TRAIL_GRID = _g.run_grid, _g.SL_GRID, _g.ARM_GRID, _g.TRAIL_GRID
HOLDOUT_START = _g.HOLDOUT_START

KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/eth_harness_control.json"

# 배포된 ETH 신호 4종 (경제성게이트 통과 이력 있음)
ETH = {
    "liquidity_sweep_topdown": (
        "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv", 30,
        (4.0, 2.0, 0.1), (10.70, 14.49)),
    "orthogonal_combo": (
        "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features_H24_GAP12_ALLFIRES.csv", 24,
        (4.0, 0.5, 0.1), (9.36, 15.13)),
    "smt_divergence": (
        "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv", 72,
        (4.0, 2.0, 0.1), (7.00, 6.18)),
    "fib_extension_exhaustion": (
        "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv", 20,
        (3.5, 0.5, 0.1), (15.15, 3.00)),
}


def log(m): print(f"[eth-control] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    kl = pd.read_csv(KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    log(f"ETH 5m klines {len(kl):,}행 -- BTC와 동일 그리드/비용/시뮬레이터")
    rep, reproduced = {"signals": {}}, 0
    for name, (csv, H, known, expect) in ETH.items():
        p = ROOT / csv
        if not p.exists():
            log(f"⚠️{name}: fires CSV 없음 -- 건너뜀"); continue
        fires = pd.read_csv(p, parse_dates=["timestamp"])
        if fires["timestamp"].dt.tz is not None:
            fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
        fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)
        cells, ns = run_grid(kl, fires, H)
        passing = [c for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0]
        genuine = [c for c in passing
                   if c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]]
        hit = next((c for c in cells if (c["sl"], c["arm"], c["trail"]) == known), None)
        log("")
        log(f"=== {name} (H={H}) fires {len(fires):,} 후보 VAL {ns['val']}/OOS {ns['oos']} ===")
        log(f"  동시양수 {len(passing)}/96  진짜 {len(genuine)}  "
            f"ARM>=1.0 진짜 {len([c for c in genuine if c['arm']>=1.0])}")
        if hit:
            ok = hit["val_fwd_bp"] > 0 and hit["oos_fwd_bp"] > 0
            reproduced += int(ok)
            log(f"  ⭐알려진 셀 SL={known[0]} ARM={known[1]} Trail={known[2]}: "
                f"VAL {hit['val_fwd_bp']:+.2f} / OOS {hit['oos_fwd_bp']:+.2f}bp  "
                f"(문서값 {expect[0]:+.2f}/{expect[1]:+.2f})  {'✅재현' if ok else '❌미재현'}")
        rep["signals"][name] = {"n_passing_96": len(passing), "n_genuine": len(genuine),
                                "known_cell": list(known),
                                "known_cell_result": hit, "documented": list(expect)}
    log("")
    log(f"=== 하네스 판정: 알려진 셀 {reproduced}/{len(ETH)} 재현 ===")
    log("  ⇒ " + ("✅하네스 정상 -- BTC 0/672는 진짜 결과다"
                  if reproduced >= 2 else "❌하네스 의심 -- BTC 결과 무효"))
    rep["reproduced"] = reproduced
    rep["harness_ok"] = reproduced >= 2
    rep["runtime_sec"] = round(time.time()-t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
