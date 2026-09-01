#!/usr/bin/env python3
"""증거신호 8종 x ATR하한 -- 정확도/승률/왜도까지 포함한 전체 지표 비교.

## 왜 필요한가

앞선 ATR 하한 스윕은 평균 bp와 총합만 봤다. 이 저장소는 **평균 bp만 보고 판단하면 안 된다**고
반복 경고해왔다 -- exit 스모크테스트의 비대칭 임계값 실험이 "평균 43.32bp인데 중앙값 -1.03bp"인
왜도 아티팩트였고(eth_rl_exit_gate_oracle_smoketest_20260901), ZDC도 "평균≪중앙값, 평균패가
평균승의 3배인 좌왜곡"이었다. 따라서 중앙값·승패 비대칭·꼬리 의존도를 함께 봐야 한다.

## ⭐측정 중 발견한 구분 (기존 스크립트들의 "win_rate"는 정확도였다)

기존 게이트들은 `(price_move > 0).mean()`을 win_rate로 보고해왔다. 이건 **비용 차감 전 방향
정확도**이지 실제 수익 여부가 아니다. 둘을 분리해 보고한다:

  - **방향정확도** = (price_move > 0)   -- 방향 콜이 맞았는가
  - **수익승률**   = (trade_return > 0) -- 비용까지 내고 실제로 벌었는가

## 지표

각 (신호 x 하한)의 **방향뒤집기를 통과한 best 조합**에서 원장 전체를 꺼내:
  n / 방향정확도 / 수익승률 / 평균bp / **중앙값bp** / 평균승 / 평균패 / 승패비 /
  profit factor / 총bp / 상위1% 기여도 / 중앙 보유봉수

⚠️ VAL+OOS만(HOLDOUT 미터치). 라벨/라이브 코드 변경 없음. 진단 전용.

Run with the quant_ai conda env (CPU only):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_evidence_signals_atr_floor_full_metrics_20260901.py
"""
from __future__ import annotations

import importlib.util
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

SWEEP = ROOT / "scripts/backtest_eth_evidence_signals_atr_floor_costgate_sweep_20260901.py"
_sspec = importlib.util.spec_from_file_location("evsig_sweep_fullmetrics", SWEEP)
_sw = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_sw)

DEM = ROOT / "scripts/backtest_eth_demarker_trailing_gridsearch_atr_floor_20260901.py"
_dspec = importlib.util.spec_from_file_location("dem_floor_fullmetrics", DEM)
_dm = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_dm)

SWEEP_REPORT = ROOT / "data/research/eth_evidence_signals_atr_floor_costgate_20260901/atr_floor_sweep_report.json"
OUT_JSON = ROOT / "data/research/eth_evidence_signals_atr_floor_costgate_20260901/full_metrics_report.json"

COST_BP = 10.0


def log(msg: str) -> None:
    print(f"[full_metrics] {msg}", flush=True)


def ledger_stats(led: pd.DataFrame) -> dict:
    if not len(led):
        return {}
    ret_bp = led["trade_return"].to_numpy() * 1e4
    pm = led["price_move"].to_numpy()
    wins, losses = ret_bp[ret_bp > 0], ret_bp[ret_bp <= 0]
    tot = float(ret_bp.sum())
    top1 = int(max(1, round(len(ret_bp) * 0.01)))
    top_sum = float(np.sort(ret_bp)[-top1:].sum())
    return {
        "n": int(len(led)),
        "dir_accuracy": round(float((pm > 0).mean()), 4),
        "profit_win_rate": round(float((ret_bp > 0).mean()), 4),
        "mean_bp": round(float(ret_bp.mean()), 2),
        "median_bp": round(float(np.median(ret_bp)), 2),
        "avg_win_bp": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_bp": round(float(losses.mean()), 2) if len(losses) else None,
        "win_loss_ratio": round(float(wins.mean() / abs(losses.mean())), 3) if len(wins) and len(losses) and losses.mean() != 0 else None,
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3) if len(losses) and losses.sum() != 0 else None,
        "total_bp": round(tot, 0),
        "top1pct_share_of_total": round(top_sum / tot, 3) if tot > 0 else None,
        "median_bars_held": float(np.median(led["bars_held"].to_numpy())),
    }


def eval_config(ts, o, h, l, c, dec, sc, atr, horizon, mask, sl, arm, trail) -> dict:
    res = simulate_single_position(
        timestamps=ts, open_px=o, high=h, low=l, close=c,
        decision_indices=dec[mask], scores=sc[mask], tp_moves=np.full(int(mask.sum()), 999.0),
        sl_moves=(sl * atr)[mask], upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=horizon,
        margin_fraction=_sw.MARGIN_FRACTION, leverage=_sw.LEVERAGE,
        roundtrip_cost_rate=_sw.ROUNDTRIP_COST_RATE,
        arm_moves=(arm * atr)[mask], trail_moves=(trail * atr)[mask])
    return ledger_stats(res.ledger)


def main() -> int:
    rep = json.loads(SWEEP_REPORT.read_text())
    klines = pd.read_csv(_sw.KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = klines["timestamp"]
    o, h, l, c = (klines[x].to_numpy() for x in ("open", "high", "low", "close"))

    results = {}
    for name, cfg in _sw.SIGNALS.items():
        fires = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        fires = fires.loc[fires["timestamp"] < _sw.HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        horizon = cfg["horizon"]
        dec_all = fires["pos"].to_numpy(dtype=np.int64)
        sc_all = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr_all = fires["atr_pct"].to_numpy(dtype=float)
        atr_bp = atr_all * 1e4
        ev = purged_decision_mask(ts, start=_sw.VAL_START, end=_sw.OOS_START, horizon_bars=horizon)
        eo = purged_decision_mask(ts, start=_sw.OOS_START, end=_sw.HOLDOUT_START, horizon_bars=horizon)
        vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())

        per_floor = {}
        for floor in _sw.ATR_FLOORS_BP:
            entry = rep["results"][name]["by_floor"].get(f"atr_floor_{floor}")
            if not entry or not entry.get("best"):
                continue
            b = entry["best"]
            keep = atr_bp >= floor
            dec, sc, atr = dec_all[keep], sc_all[keep], atr_all[keep]
            vm = np.array([d in vset for d in dec])
            om = np.array([d in oset for d in dec])
            per_floor[floor] = {
                "config": {"sl": b["sl"], "arm": b["arm"], "trail": b["trail"]},
                "val": eval_config(ts, o, h, l, c, dec, sc, atr, horizon, vm, b["sl"], b["arm"], b["trail"]),
                "oos": eval_config(ts, o, h, l, c, dec, sc, atr, horizon, om, b["sl"], b["arm"], b["trail"]),
            }
        results[name] = per_floor

    # demarker / kalman via their own fire-building path
    log("building demarker/kalman fires (build_fires path)...")
    kl2 = _dm.load_klines()
    ind = _dm.build_indicator_frame(kl2)
    dem = _dm.compute_demarker(kl2["high"], kl2["low"])
    ind_d = ind.copy(); ind_d["dem"] = dem.to_numpy()
    levels, _v = _dm.kalman_level_and_velocity(kl2["close"].to_numpy())
    kdev = pd.Series((kl2["close"].to_numpy() - levels) / levels, index=kl2.index)
    kz = _dm.rolling_zscore(kdev)
    ind_k = ind.copy(); ind_k["kalman_dev_z"] = kz.to_numpy()

    dem_rep = json.loads((ROOT / "data/research/eth_demarker_atr_floor_costgate_20260901/atr_floor_costgate_report.json").read_text())
    ts2 = kl2["timestamp"]
    o2, h2, l2, c2 = (kl2[x].to_numpy() for x in ("open", "high", "low", "close"))
    for name, (ind_x, ttop, tbot, extr, fcols) in {
        "demarker_extreme": (ind_d, dem >= 0.90, dem <= 0.10, dem.fillna(0.5).to_numpy(), _dm.FEATURE_COLUMNS + ["dem"]),
        "kalman_deviation_meanrev": (ind_k, kz >= 2.0, kz <= -2.0, kz.fillna(0.0).to_numpy(), _dm.FEATURE_COLUMNS + ["kalman_dev_z"]),
    }.items():
        cfgs = _dm.SIGNAL_CONFIG[name]
        horizon, K = cfgs["horizon"], cfgs["K"]
        fires = _dm.build_fires(kl2, ind_x, ttop, tbot, extr, fcols, horizon, cfgs["gap"], K).sort_values("pos").reset_index(drop=True)
        dec_all = fires["pos"].to_numpy(dtype=np.int64)
        sc_all = np.where(fires["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr_all = ind_x["atr_pct"].to_numpy()[dec_all]
        thr_bp = K * atr_all * 1e4  # demarker 리포트는 K*ATR 기준 하한을 썼으므로 그대로 맞춘다
        ev = purged_decision_mask(ts2, start=_dm.VAL_START, end=_dm.OOS_START, horizon_bars=horizon)
        eo = purged_decision_mask(ts2, start=_dm.OOS_START, end=_dm.HOLDOUT_START, horizon_bars=horizon)
        vset, oset = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
        per_floor = {}
        for floor in _dm.FLOORS_BP:
            e = dem_rep["results"][name]["by_floor"].get(f"floor_{floor}")
            if not e or not e.get("genuine_top"):
                continue
            b = e["genuine_top"][0]
            keep = thr_bp >= floor
            dec, sc, atr = dec_all[keep], sc_all[keep], atr_all[keep]
            vm = np.array([d in vset for d in dec]); om = np.array([d in oset for d in dec])
            per_floor[f"KxATR_{floor}"] = {
                "config": {"sl": b["sl"], "arm": b["arm"], "trail": b["trail"]},
                "val": eval_config(ts2, o2, h2, l2, c2, dec, sc, atr, horizon, vm, b["sl"], b["arm"], b["trail"]),
                "oos": eval_config(ts2, o2, h2, l2, c2, dec, sc, atr, horizon, om, b["sl"], b["arm"], b["trail"]),
            }
        results[name] = per_floor

    log("")
    log("=== 전체 지표 (각 신호의 best 조합, VAL 기준 / OOS는 괄호) ===")
    log(f"  {'signal':24s} {'floor':>7s} {'n':>5s} {'방향정확':>7s} {'수익승률':>7s} {'평균bp':>8s} {'중앙bp':>8s} {'승패비':>6s} {'PF':>5s} {'총bp':>7s} {'상위1%':>6s}")
    for name, pf in results.items():
        for floor, e in pf.items():
            v, ov = e["val"], e["oos"]
            if not v:
                continue
            log(f"  {name:24s} {str(floor):>7s} {v['n']:>5d} {v['dir_accuracy']:>7.1%} {v['profit_win_rate']:>7.1%} "
                f"{v['mean_bp']:>+8.2f} {v['median_bp']:>+8.2f} {str(v['win_loss_ratio']):>6s} {str(v['profit_factor']):>5s} "
                f"{v['total_bp']:>7.0f} {str(v['top1pct_share_of_total']):>6s}")
        log("")

    report = {
        "analysis": "evidence_signals_atr_floor_full_metrics", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"holdout_touched": False, "live_code_changed": False, "diagnostic_only": True,
                  "note": ("dir_accuracy = (price_move>0) = 비용 차감 전 방향 정확도 (기존 게이트들이 "
                           "'win_rate'로 보고해온 값). profit_win_rate = (trade_return>0) = 비용까지 "
                           "내고 실제로 번 비율. 둘은 다르다.")},
        "cost_bp": COST_BP, "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
