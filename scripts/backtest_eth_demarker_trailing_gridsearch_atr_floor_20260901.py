#!/usr/bin/env python3
"""demarker_extreme 경제성 게이트 -- 저ATR 구간 제외(절대 bp 하한) 재실행.

## 배경 (docs/homer/README.md 5.9)절)

2026-09-01 증거신호 8종 ATR-라벨 진단에서 **demarker_extreme 1개만 결함** 판정:
K=0.70/horizon=8로 8개 중 최저라, 저변동성 10% 구간에서 라벨이 요구하는 최소 자격 움직임이
**8.9bp로 왕복비용 10bp보다 작다**(hit이어도 순손실). 전체 발동의 8.7%가 비용 미만, 43.9%가
비용 2배 미만.

5.9)절 2번 원칙: **하한 도입 여부는 진단이나 분류 AUC가 아니라 경제성 게이트로 판단하라**
(V_REBOUND에서 분류 +0.021이 경제성으로 전이되지 않은 전례). 이 스크립트가 그 판정을 한다 --
demarker가 기록한 "96/96 조합 VAL+OOS 동시양수" 통과가 저ATR 구간을 빼면 어떻게 되는가.

## 방법 (기존 게이트와 완전 동일, 필터만 추가)

`backtest_eth_kalman_demarker_trailing_gridsearch_20260831.py`의 엔진/상수/그리드를 그대로
재사용한다 -- `core.causal_futures_backtest.simulate_single_position`, MARGIN_FRACTION=0.30 /
LEVERAGE=3.0 / ROUNDTRIP_COST_RATE=0.001(10bp), SL 6 x ARM 4 x Trail 4 = 96조합,
purged_decision_mask 기반 VAL/OOS, cluster-anchored 후보 무조건 진입(score=+-1).

**추가한 것 두 가지뿐**:
1. **ATR 하한 필터**: 후보 중 `K * atr_pct * 1e4 >= FLOOR`인 것만 남긴다. FLOOR를
   0(=기존 게이트 재현) / 10 / 20 / 30bp로 스윕.
2. **방향뒤집기 대조군**: scores 부호를 뒤집어 같은 그리드를 돌린다. ARM이 그리드 낮은 쪽인
   조합은 방향 실력과 무관하게 봉노이즈만으로 승률을 만들 수 있으므로
   (feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901), **통과 조합 전체에**
   적용해야 한다 -- 단일 config만 뒤집으면 오판(fib_extension_exhaustion 전례).

`kalman_deviation_meanrev`는 같은 진단에서 OK(저ATR 구간 3.10x)였으므로 **대조군으로 함께**
돌린다 -- 하한이 demarker만 바꾸고 kalman은 거의 안 바꾼다면 진단이 맞았다는 방증이 된다.

⚠️ VAL+OOS만 사용(HOLDOUT 미터치, 기존 게이트와 동일 컨벤션). 라벨/라이브 코드 변경 없음.

Run with the quant_ai conda env (CPU only):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/backtest_eth_demarker_trailing_gridsearch_atr_floor_20260901.py
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
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity, rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    HOLDOUT_START, OOS_START, VAL_START, build_fires, load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS, build_indicator_frame,
)

OUT_DIR = ROOT / "data/research/eth_demarker_atr_floor_costgate_20260901"

# constants verbatim from the original gate
MARGIN_FRACTION = 0.30
LEVERAGE = 3.0
ROUNDTRIP_COST_RATE = 0.001
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5},
}

FLOORS_BP = [0, 10, 20, 30]


def log(msg: str) -> None:
    print(f"[demarker_atr_floor] {msg}", flush=True)


def run_grid(ts_full, open_px, high, low, close, dec_idx, scores, atr, horizon,
             val_mask, oos_mask) -> pd.DataFrame:
    tp_placeholder = np.full(len(dec_idx), 999.0)
    rows = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for trail in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": trail}
                ok = True
                for wname, mask in (("val", val_mask), ("oos", oos_mask)):
                    res = simulate_single_position(
                        timestamps=ts_full, open_px=open_px, high=high, low=low, close=close,
                        decision_indices=dec_idx[mask], scores=scores[mask],
                        tp_moves=tp_placeholder[mask], sl_moves=(sl * atr)[mask],
                        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
                        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                        arm_moves=(arm * atr)[mask], trail_moves=(trail * atr)[mask],
                    )
                    led = res.ledger
                    n = int(len(led))
                    avg_bp = float(led["trade_return"].mean() * 10000) if n else float("nan")
                    wr = float((led["price_move"] > 0).mean()) if n else float("nan")
                    row[f"{wname}_n"] = n
                    row[f"{wname}_avg_bp"] = round(avg_bp, 3)
                    row[f"{wname}_win_rate"] = round(wr, 4)
                    if not (n > 0 and avg_bp > 0):
                        ok = False
                row["both_positive"] = ok
                rows.append(row)
    return pd.DataFrame(rows)


def gate_with_floor(name: str, klines, ind, trig_top, trig_bot, extremeness, feat_cols) -> dict:
    cfg = SIGNAL_CONFIG[name]
    horizon, K = cfg["horizon"], cfg["K"]
    fires = build_fires(klines, ind, trig_top, trig_bot, extremeness, feat_cols,
                        horizon, cfg["gap"], K).sort_values("pos").reset_index(drop=True)

    open_px, high, low, close = (klines[c].to_numpy() for c in ("open", "high", "low", "close"))
    ts_full = klines["timestamp"]
    atr_pct_all = ind["atr_pct"].to_numpy()

    dec_all = fires["pos"].to_numpy(dtype=np.int64)
    sc_all = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
    atr_all = atr_pct_all[dec_all]
    thr_bp_all = K * atr_all * 1e4  # 라벨이 요구하는 최소 자격 움직임(bp)

    eligible_val = purged_decision_mask(ts_full, start=VAL_START, end=OOS_START, horizon_bars=horizon)
    eligible_oos = purged_decision_mask(ts_full, start=OOS_START, end=HOLDOUT_START, horizon_bars=horizon)
    val_set, oos_set = set(np.flatnonzero(eligible_val).tolist()), set(np.flatnonzero(eligible_oos).tolist())

    log(f"\n=== {name} (H={horizon}, GAP={cfg['gap']}, K={K}) 후보 {len(fires)}건 ===")
    log(f"  최소자격움직임(bp) 중앙값={np.median(thr_bp_all):.1f}  p10={np.percentile(thr_bp_all,10):.1f}")

    out = {}
    for floor in FLOORS_BP:
        keep = thr_bp_all >= floor
        dec, sc, atr = dec_all[keep], sc_all[keep], atr_all[keep]
        vm = np.array([d in val_set for d in dec])
        om = np.array([d in oos_set for d in dec])
        if vm.sum() < 20 or om.sum() < 20:
            log(f"  FLOOR={floor:2d}bp: 후보부족(val={vm.sum()} oos={om.sum()}) -- 스킵")
            continue

        real = run_grid(ts_full, open_px, high, low, close, dec, sc, atr, horizon, vm, om)
        flip = run_grid(ts_full, open_px, high, low, close, dec, -sc, atr, horizon, vm, om)
        fmap = {(r.sl, r.arm, r.trail): r for r in flip.itertuples()}

        passing = real[real["both_positive"]].copy()
        genuine = []
        for r in passing.itertuples():
            f = fmap[(r.sl, r.arm, r.trail)]
            gap_val = r.val_avg_bp - f.val_avg_bp
            gap_oos = r.oos_avg_bp - f.oos_avg_bp
            is_gen = bool(gap_val > 0 and gap_oos > 0 and f.val_avg_bp < 0 and f.oos_avg_bp < 0)
            genuine.append({"sl": r.sl, "arm": r.arm, "trail": r.trail,
                            "val_bp": r.val_avg_bp, "oos_bp": r.oos_avg_bp,
                            "val_n": int(r.val_n), "oos_n": int(r.oos_n),
                            "val_win": r.val_win_rate, "oos_win": r.oos_win_rate,
                            "flip_val_bp": f.val_avg_bp, "flip_oos_bp": f.oos_avg_bp,
                            "gap_val": round(gap_val, 2), "gap_oos": round(gap_oos, 2),
                            "genuine": is_gen})
        n_gen = sum(g["genuine"] for g in genuine)
        kept_pct = 100.0 * keep.mean()
        log(f"  FLOOR={floor:2d}bp: 후보 {keep.sum():4d}/{len(keep)} ({kept_pct:4.1f}%) "
            f"val_n={vm.sum():4d} oos_n={om.sum():4d}  "
            f"| 96조합 중 VAL+OOS양수 {len(passing):2d}  방향뒤집기통과 {n_gen:2d}")
        top = sorted([g for g in genuine if g["genuine"]],
                     key=lambda g: -min(g["val_bp"], g["oos_bp"]))[:3]
        for g in top:
            log(f"      SL={g['sl']}/ARM={g['arm']}/Tr={g['trail']}  "
                f"VAL={g['val_bp']:+7.2f}bp(n={g['val_n']},win={g['val_win']:.1%}) "
                f"OOS={g['oos_bp']:+7.2f}bp(n={g['oos_n']},win={g['oos_win']:.1%}) "
                f"| flip {g['flip_val_bp']:+.2f}/{g['flip_oos_bp']:+.2f} "
                f"gap {g['gap_val']:+.1f}/{g['gap_oos']:+.1f}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        real.assign(signal=name, floor_bp=floor).to_csv(OUT_DIR / f"{name}_floor{floor}_grid.csv", index=False)
        out[f"floor_{floor}"] = {
            "n_candidates": int(keep.sum()), "pct_kept": round(kept_pct, 1),
            "val_n": int(vm.sum()), "oos_n": int(om.sum()),
            "n_both_positive": int(len(passing)), "n_genuine_after_flip": int(n_gen),
            "genuine_top": top,
        }
    return {"K": K, "horizon": horizon, "n_fires": int(len(fires)),
            "threshold_bp_median": round(float(np.median(thr_bp_all)), 1), "by_floor": out}


def main() -> int:
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)

    results = {}
    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    results["demarker_extreme"] = gate_with_floor(
        "demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
        dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kdev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kz = rolling_zscore(kdev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kz.to_numpy()
    results["kalman_deviation_meanrev"] = gate_with_floor(
        "kalman_deviation_meanrev", klines, ind_kal, kz >= 2.0, kz <= -2.0,
        kz.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])

    report = {
        "signal": "demarker_kalman_atr_floor_costgate", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "holdout_touched": False, "live_code_changed": False, "label_changed": False,
            "engine": "core.causal_futures_backtest.simulate_single_position (기존 게이트와 동일)",
            "constants": {"margin_fraction": MARGIN_FRACTION, "leverage": LEVERAGE,
                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE},
            "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
            "purpose": ("5.9)절 2번 원칙 이행: demarker_extreme의 저ATR 결함에 대해 하한 도입 "
                        "여부를 분류가 아닌 경제성으로 판정. kalman은 진단 OK였던 대조군."),
        },
        "floors_bp": FLOORS_BP,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "atr_floor_costgate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_DIR / 'atr_floor_costgate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
