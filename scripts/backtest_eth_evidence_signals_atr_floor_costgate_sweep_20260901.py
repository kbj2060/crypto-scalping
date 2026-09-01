#!/usr/bin/env python3
"""증거신호 6종 -- 저ATR 구간 제외(ATR 하한) 경제성 게이트 균일 스윕.

## 왜 이 조사를 하는가 (그리고 앞선 진단과 어떻게 다른가)

2026-09-01 ATR-라벨 진단(docs/homer/README.md 5.9)절)은 **라벨 문턱(K x ATR)**이 거래비용
아래로 내려가는지를 봤고, demarker_extreme 1개만 결함으로 판정했다. 그런데 demarker 경제성
재실행 뒤 사용자가 "다른 신호도 이런 개선이 가능한가"를 묻는 과정에서 **두 질문이 다르다**는 게
드러났다:

  - **라벨 결함**(K x ATR < 비용): K가 낮은 demarker만 해당.
  - **경제성 기회**(저ATR 구간에서 고정비용 10bp가 수익을 잠식): K와 무관하게 **ATR 자체**의
    문제 -- SL/ARM/Trail이 전부 ATR 배수인데 비용만 고정이기 때문.

ATR 자체 분포를 보니 6개 신호 **전부** 저ATR 구간에서 상당히 많이 거래하고 있었다
(ATR<25bp 비중: smt 52.7% / liquidity_sweep 48.6% / fib 45.1% / taker 38.3% /
orthogonal 34.9% / str_z 21.1%). 따라서 하한이 K x ATR이 아니라 **ATR(bp) 자체**여야 공정한
비교가 되고, 6개 전부에 개선 여지가 있을 수 있다.

## 방법 (각 신호의 기존 게이트와 동일, 필터만 추가)

각 신호의 기존 gridsearch 스크립트들이 전부 같은 구조를 쓴다 -- 사전 빌드된 fires CSV
(`pos`/`side`/`atr_pct`) + `core.causal_futures_backtest.simulate_single_position` +
MARGIN_FRACTION=0.30/LEVERAGE=3.0/ROUNDTRIP_COST_RATE=0.001 + SL 6 x ARM 4 x Trail 4 = 96조합 +
purged_decision_mask 기반 VAL/OOS. 그 구조를 그대로 재사용하고 **ATR 하한 필터**와
**방향뒤집기 대조군**만 추가한다.

방향뒤집기는 통과 조합 전체에 적용한다(단일 config만 검사하면 오판 --
feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901, fib_extension_exhaustion가
바로 이걸로 경제성 클레임이 철회된 전례가 있다).

⚠️ VAL+OOS만 사용(HOLDOUT 미터치). 라벨/라이브 코드 변경 없음. 하한 도입은 이 조사 결과만으로
결정하지 않는다 -- 각 신호의 HOLDOUT 소진 상태를 함께 고려해야 한다.

Run with the quant_ai conda env (CPU only):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/backtest_eth_evidence_signals_atr_floor_costgate_sweep_20260901.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_evidence_signals_atr_floor_costgate_20260901"

MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

ATR_FLOORS_BP = [0, 20, 25, 30]

SIGNALS = {
    "taker_delta_z_climax": {
        "fires": "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv",
        "horizon": 24},
    "short_term_return_z": {
        "fires": "data/labels/eth_5m_short_term_return_z_metalabel_20260829/eth_5m_short_term_return_z_metalabel_features.csv",
        "horizon": 12},
    "liquidity_sweep": {
        "fires": "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv",
        "horizon": 30},
    "orthogonal_combo": {
        "fires": "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features_H24_GAP12_ALLFIRES.csv",
        "horizon": 24},
    "smt_divergence": {
        "fires": "data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv",
        "horizon": 72},
    "fib_extension_exhaustion": {
        "fires": "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv",
        "horizon": 20},
}


def log(msg: str) -> None:
    print(f"[evsig_atr_floor] {msg}", flush=True)


def run_grid(ts, open_px, high, low, close, dec, scores, atr, horizon, vm, om) -> pd.DataFrame:
    tp_ph = np.full(len(dec), 999.0)
    rows = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": trail}
                ok = True
                for wname, mask in (("val", vm), ("oos", om)):
                    res = simulate_single_position(
                        timestamps=ts, open_px=open_px, high=high, low=low, close=close,
                        decision_indices=dec[mask], scores=scores[mask], tp_moves=tp_ph[mask],
                        sl_moves=(sl * atr)[mask], upper_threshold=1.0, lower_threshold=-1.0,
                        horizon_bars=horizon, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                        arm_moves=(arm * atr)[mask], trail_moves=(trail * atr)[mask])
                    led = res.ledger
                    n = int(len(led))
                    avg = float(led["trade_return"].mean() * 1e4) if n else float("nan")
                    wr = float((led["price_move"] > 0).mean()) if n else float("nan")
                    row[f"{wname}_n"], row[f"{wname}_avg_bp"] = n, round(avg, 3)
                    row[f"{wname}_win_rate"] = round(wr, 4)
                    if not (n > 0 and avg > 0):
                        ok = False
                row["both_positive"] = ok
                rows.append(row)
    return pd.DataFrame(rows)


def sweep_signal(name: str, cfg: dict, klines: pd.DataFrame) -> dict:
    fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
    horizon = cfg["horizon"]

    ts = klines["timestamp"]
    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))
    dec_all = fires["pos"].to_numpy(dtype=np.int64)
    sc_all = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
    atr_all = fires["atr_pct"].to_numpy(dtype=float)
    atr_bp_all = atr_all * 1e4

    ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=horizon)
    eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
    vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())

    log(f"\n=== {name} (H={horizon}) 후보 {len(fires)}건, ATR중앙값 {np.median(atr_bp_all):.1f}bp ===")
    out = {}
    for floor in ATR_FLOORS_BP:
        keep = atr_bp_all >= floor
        dec, sc, atr = dec_all[keep], sc_all[keep], atr_all[keep]
        vm = np.array([d in vset for d in dec])
        om = np.array([d in oset for d in dec])
        if vm.sum() < 30 or om.sum() < 30:
            log(f"  ATR>={floor:2d}bp: 표본부족(val={vm.sum()} oos={om.sum()}) -- 스킵")
            continue
        real = run_grid(ts, open_px, high, low, close, dec, sc, atr, horizon, vm, om)
        flip = run_grid(ts, open_px, high, low, close, dec, -sc, atr, horizon, vm, om)
        fmap = {(r.sl, r.arm, r.trail): r for r in flip.itertuples()}

        passing = real[real["both_positive"]]
        genuine = []
        for r in passing.itertuples():
            f = fmap[(r.sl, r.arm, r.trail)]
            if (r.val_avg_bp - f.val_avg_bp) > 0 and (r.oos_avg_bp - f.oos_avg_bp) > 0 \
               and f.val_avg_bp < 0 and f.oos_avg_bp < 0:
                genuine.append({"sl": r.sl, "arm": r.arm, "trail": r.trail,
                                "val_bp": r.val_avg_bp, "oos_bp": r.oos_avg_bp,
                                "val_n": int(r.val_n), "oos_n": int(r.oos_n),
                                "val_win": r.val_win_rate, "oos_win": r.oos_win_rate,
                                "gap_val": round(r.val_avg_bp - f.val_avg_bp, 2),
                                "gap_oos": round(r.oos_avg_bp - f.oos_avg_bp, 2)})
        best = max(genuine, key=lambda g: min(g["val_bp"], g["oos_bp"])) if genuine else None
        log(f"  ATR>={floor:2d}bp: 후보 {keep.sum():5d}/{len(keep)} ({keep.mean()*100:4.1f}%) "
            f"val_n={vm.sum():4d} oos_n={om.sum():4d} | 양수 {len(passing):2d}/96 진짜 {len(genuine):2d}"
            + (f" | best VAL={best['val_bp']:+6.2f} OOS={best['oos_bp']:+6.2f} "
               f"(SL{best['sl']}/ARM{best['arm']}/Tr{best['trail']}, win {best['val_win']:.0%}/{best['oos_win']:.0%})"
               if best else " | 진짜 조합 없음"))
        out[f"atr_floor_{floor}"] = {
            "n_candidates": int(keep.sum()), "pct_kept": round(float(keep.mean()) * 100, 1),
            "val_n": int(vm.sum()), "oos_n": int(om.sum()),
            "n_both_positive": int(len(passing)), "n_genuine": len(genuine), "best": best,
        }
    return {"horizon": horizon, "n_fires": int(len(fires)),
            "atr_bp_median": round(float(np.median(atr_bp_all)), 1), "by_floor": out}


def main() -> int:
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    log(f"{len(klines)} klines loaded")
    results = {n: sweep_signal(n, c, klines) for n, c in SIGNALS.items()}

    log("\n=== 요약: ATR 하한이 각 신호를 개선하는가 (best 조합의 min(VAL,OOS) bp) ===")
    header = " ".join(("F" + str(f)).rjust(9) for f in ATR_FLOORS_BP)
    log(f"  {'signal':26s} {'ATR중앙':>7s} " + header)
    for n, r in results.items():
        cells = []
        for f in ATR_FLOORS_BP:
            e = r["by_floor"].get(f"atr_floor_{f}")
            b = e["best"] if e else None
            cells.append(f"{min(b['val_bp'], b['oos_bp']):+9.2f}" if b else f"{'-':>9s}")
        log(f"  {n:26s} {r['atr_bp_median']:7.1f} " + " ".join(cells))

    report = {
        "signal": "evidence_signals_atr_floor_costgate_sweep", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "holdout_touched": False, "live_code_changed": False, "label_changed": False,
            "engine": "core.causal_futures_backtest.simulate_single_position (각 신호 기존 게이트와 동일)",
            "constants": {"margin_fraction": MARGIN_FRACTION, "leverage": LEVERAGE,
                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE},
            "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
            "floor_on": "ATR itself (bp), NOT K*ATR -- economics scales with ATR while cost is fixed",
            "direction_flip_applied_to": "all VAL+OOS-positive combos",
        },
        "atr_floors_bp": ATR_FLOORS_BP, "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "atr_floor_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_DIR / 'atr_floor_sweep_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
