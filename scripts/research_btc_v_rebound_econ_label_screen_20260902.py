#!/usr/bin/env python3
"""BTC **경제적 결과 라벨** 스크리닝 -- ETH에서 통한 방식의 BTC 이식 + BTC 전용 튜닝.

## 배경

2026-09-02 ETH에서 확인된 것: 패턴 라벨("V자 모양")은 모델이 **이미 완성된 모양**을 고르게
만들어 경제성이 0이었고, 라벨을 **"이 진입이 비용 후 이익인가"**(트레일링 순손익>0)로 바꾸니
VAL/OOS/HOLDOUT을 전부 통과했다(HOLDOUT +6.09bp, 뒤집기 -4.15bp).
전문: docs/model_contracts/eth_v_rebound_econ_label_autotrade_spec_20260902.md

## ⚠️BTC로 그대로 이식하면 안 되는 것

메모리 실측: **"ETH 승자 S12_K3은 BTC에서 3/10 최하위"**(레짐 라벨). 파라미터는 자산 간
이식되지 않는다. 이 스크립트가 BTC 전용으로 다시 정하는 것:

  1. ⭐**라벨 exit 셀** -- ETH는 (SL 5.0, ARM 1.5, Trail 0.1). BTC는 ATR 특성이 다르다.
     **TRAIN에서만** 고른다(VAL에서 고르면 라벨이 평가셋을 훔쳐본다).
  2. 비용 -- BTC도 왕복 10bp(수수료 우대 가정 금지 규칙)
  3. 임계값/동시보유 -- 다음 단계(VAL 선정)에서

## 데이터

`data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv`
(277,191행, 2024-01-01~2026-08-20, bar-wide). Tier0 22 + 측면별 트리거.
⚠️교차자산 피쳐 없음 -- `FeatureEngineer`의 `close_btc` 슬롯 함정(BTC 주체면 그 슬롯에 ETH가
들어감)은 여기 해당 없음. `smt_divergence`도 BTC 후보에서 이미 제외돼 있다.

## 산출

TRAIN 기준 exit 셀 격자 스캔 -> 라벨률/기대값 -> **BTC 전용 라벨 셀 확정**.
이 스크립트는 라벨 정의만 정한다. 학습·검증은 다음 스크립트.

⚠️VAL/OOS/HOLDOUT 미터치(TRAIN만 사용). 라이브 코드 변경 없음.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_btc", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
sim_exit = _pf.sim_exit
SL_GRID, ARM_GRID, TRAIL_GRID = _pf.SL_GRID, _pf.ARM_GRID, _pf.TRAIL_GRID
# ⭐BTC 전용 확장: 1차 스캔에서 최적이 SL=5.0(ETH 격자의 최댓값)에 **붙어** 있었다.
# BTC ATR이 16bp로 ETH(~23bp)보다 30% 작아 같은 ATR 배수라도 손절이 절대폭으로 좁다 --
# 경계에 붙은 최적은 격자 밖에 진짜 최적이 있다는 신호이므로 넓혀서 다시 본다.
# ⚠️3차 교정(2026-09-02): 앞선 스캔이 SL=16을 골랐고 2단계가 OOS에서 -22.18bp로 참패했다.
# 원인은 선정 기준이다 -- BTC는 무작위 기준선이 음수라 "평균 최대"가 SL을 무한정 넓히는
# 방향으로 밀었고, 그 결과 승률 84~95%/손익비 0.089라는 **극단적 음의 왜도**(변동성 매도)가
# 만들어졌다. 구간에 따라 부호가 뒤집히는 프로파일이다(VAL +5.36 -> OOS -22.18).
# 손익비가 0.302->0.180으로 떨어지는 것이 경고였는데 놓쳤다.
# 교정: SL 상한을 8.0으로 묶고, 선정에 **손익비 하한**을 건다.
SL_GRID = tuple(x for x in (tuple(SL_GRID) + (6.0, 8.0)) if x <= 8.0)
# ⭐4차 교정(2026-09-02): 앞선 두 시도가 실패한 뒤 격자를 다시 보니 **ARM 상한이 1.5**였다.
# ARM은 이익 목표인데 BTC ATR 16bp에서 1.5x면 목표가 24bp -- 비용 10bp 대비 2.4:1뿐이다
# (ETH는 1.5x23=35bp로 3.5:1). 나는 SL(손실 폭)만 넓혔는데 방향이 반대였다.
# BTC가 수수료 바닥에 가깝다는 진단이 맞다면 **넓혀야 할 것은 ARM**이다.
ARM_GRID = tuple(ARM_GRID) + (2.5, 4.0, 6.0, 8.0)
PAYOFF_FLOOR = 0.25          # ETH 실측 0.30~0.39보다 낮게 두되 복권형(0.089)은 배제
# ⚠️2차 확장: 1차 확장(=12.0)에서도 최적이 경계에 붙었다. 청산사유를 확인해보니
# SL=12에서도 94.2%가 여전히 트레일링 스톱으로 나가고 보유 중앙값은 11->14봉뿐이라
# "buy-and-hold로 퇴화한다"는 가설은 반증됐다 -- 넓은 손절이 무장 전에 잘리던 거래를
# 실제로 살려내는 것이다. 내부 최적이 존재하는지 확인하려 더 넓힌다.
FORWARD_BARS = _pf.FORWARD_BARS

BTC_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
COST_BP = 10.0
ARTIFACT_FREE_MIN = 1.0
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
ETH_CELL = (5.0, 1.5, 0.1)
OUT = ROOT / "data/research/btc_v_rebound_econ_label_screen_20260902/report_arm_extended.json"

# ETH Tier0와 동일 구성 (side-dependent 2개는 아래서 유도)
BAR_FEATURES = ["atr", "atr_percentile_864", "range_width_pct", "hour_utc", "weekday",
                "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
                "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14",
                "pdi", "ndi", "bb_width_pctile", "rsi"]
TIER0 = ["is_downside", "sweep_penetration_atr", "flow_aligned_delta_z"] + BAR_FEATURES


def log(m): print(f"[btc-screen] {m}", flush=True)


def build_long() -> tuple[pd.DataFrame, dict]:
    """BTC bar-wide CSV -> (bar, side) long frame. ETH long_frame_for와 같은 유도 공식."""
    need = ["timestamp", "open", "high", "low", "close", "sweep_level_low",
            "sweep_level_high"] + BAR_FEATURES
    df = pd.read_csv(BTC_CSV, usecols=lambda c: c in set(need))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    log(f"bar-wide {len(df):,}행 ({df.timestamp.min()} ~ {df.timestamp.max()})")

    rows = []
    atr = df["atr"].to_numpy(dtype=float)
    dz = df["delta_z"].to_numpy(dtype=float)
    for side, is_down in (("bottom", True), ("top", False)):
        sub = pd.DataFrame({"timestamp": df["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        level = (df["sweep_level_low"] if is_down else df["sweep_level_high"]).to_numpy(dtype=float)
        pen = (level - df["low"].to_numpy()) if is_down else (df["high"].to_numpy() - level)
        with np.errstate(invalid="ignore", divide="ignore"):
            sub["sweep_penetration_atr"] = np.where(np.isfinite(atr) & (atr > 0), pen / atr, np.nan)
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for c in BAR_FEATURES:
            sub[c] = df[c].to_numpy()
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long["bar_idx"] = np.tile(np.arange(len(df)), 2)
    meta = {"bars": int(len(df)), "start": str(df.timestamp.min()), "end": str(df.timestamp.max())}
    return long, {"df": df, **meta}


def main() -> int:
    t0 = time.time()
    long, meta = build_long()
    df = meta.pop("df")
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    nb = len(df)

    long = long.dropna(subset=TIER0).reset_index(drop=True)
    long = long.loc[long["bar_idx"] + FORWARD_BARS + 1 < nb].reset_index(drop=True)
    tr = long.loc[long["timestamp"] < TRAIN_END].reset_index(drop=True)
    log(f"long frame {len(long):,}행 -> TRAIN {len(tr):,}행 (피쳐 {len(TIER0)}개)")
    assert tr["timestamp"].max() < TRAIN_END, "TRAIN 경계 위반"

    i_all = tr["bar_idx"].to_numpy().astype(int)
    sgn = np.where(tr["is_downside"].to_numpy() == 1, 1.0, -1.0)
    at = tr["atr"].to_numpy(dtype=float)
    atr_bp = at / c[i_all] * 1e4
    log(f"TRAIN ATR: 중앙 {np.median(atr_bp):.1f}bp  p25 {np.percentile(atr_bp,25):.1f}  "
        f"p75 {np.percentile(atr_bp,75):.1f}   (참고 ETH 중앙 ~23bp)")

    def net_for(cell):
        sl, arm, trv = cell
        out = np.full(len(tr), np.nan)
        for s_ in range(0, len(tr), CHUNK):
            e_ = min(s_ + CHUNK, len(tr))
            j = i_all[s_:e_]
            H = np.stack([h[x+1:x+1+FORWARD_BARS] for x in j])
            L = np.stack([l[x+1:x+1+FORWARD_BARS] for x in j])
            C = np.stack([c[x+1:x+1+FORWARD_BARS] for x in j])
            pn, _ = sim_exit(o[j+1], at[s_:e_], sgn[s_:e_], H, L, C, sl, arm, trv)
            out[s_:e_] = pn * 1e4 - COST_BP
        return out

    log("")
    log("=== TRAIN exit 셀 격자 (라벨 정의용, ARM>=1.0) ===")
    log(f"  {'SL':>5s} {'ARM':>5s} {'Trail':>6s} {'라벨률':>7s} {'중앙net':>9s} {'평균net':>9s} "
        f"{'승률':>6s} {'손익비':>7s}")
    results = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            if arm < ARTIFACT_FREE_MIN:
                continue
            for trv in TRAIL_GRID:
                v = net_for((sl, arm, trv))
                v = v[np.isfinite(v)]
                if len(v) < 1000:
                    continue
                w = v > 0
                r = {"cell": [sl, arm, trv], "label_rate": float(w.mean()),
                     "median_bp": float(np.median(v)), "mean_bp": float(v.mean()),
                     "win_rate": float(w.mean()),
                     "payoff": float(v[w].mean() / -v[~w].mean()) if w.any() and (~w).any() else None}
                results.append(r)
                log(f"  {sl:5.1f} {arm:5.2f} {trv:6.2f} {r['label_rate']:7.4f} "
                    f"{r['median_bp']:+8.2f}bp {r['mean_bp']:+8.2f}bp {r['win_rate']*100:5.1f}% "
                    f"{r['payoff'] if r['payoff'] else float('nan'):7.3f}")

    # 라벨용 셀 선정: **라벨률이 극단(0.5 미만/0.9 초과)이 아니면서 평균 net이 최대**
    # -- 라벨률이 너무 치우치면 학습 신호가 없다(ETH는 0.7585였다).
    # ⭐선정 기준 교정: 평균만 보면 왜도가 폭주한다(위 주석). 손익비 하한을 함께 건다.
    ok = [r for r in results if 0.55 <= r["label_rate"] <= 0.85
          and (r["payoff"] or 0) >= PAYOFF_FLOOR]
    if not ok:
        log(f"  ⚠️손익비>={PAYOFF_FLOOR} 조건을 만족하는 셀 없음 -- 하한 없이 재선정")
        ok = [r for r in results if 0.55 <= r["label_rate"] <= 0.85] or results
    best = max(ok, key=lambda r: r["mean_bp"])
    eth = next((r for r in results if tuple(r["cell"]) == ETH_CELL), None)

    log("")
    log("=== BTC 전용 라벨 셀 선정 (TRAIN only) ===")
    log(f"  ⭐선정: SL/ARM/Trail = {best['cell']}  라벨률 {best['label_rate']:.4f}  "
        f"평균 {best['mean_bp']:+.2f}bp  중앙 {best['median_bp']:+.2f}bp  손익비 {best['payoff']:.3f}")
    if eth:
        log(f"  (참고) ETH 셀 {list(ETH_CELL)}: 라벨률 {eth['label_rate']:.4f}  "
            f"평균 {eth['mean_bp']:+.2f}bp  중앙 {eth['median_bp']:+.2f}bp")
        log(f"  → {'동일' if tuple(best['cell'])==ETH_CELL else '⭐다름 -- 자산별 재튜닝이 실제로 필요했다'}")

    report = {"signal": "btc_v_rebound_econ_label_screen", "asset": "BTCUSDT",
              "scope": {"purpose": "BTC 전용 경제라벨 exit 셀 확정 (TRAIN only)",
                        "cost_bp": COST_BP, "forward_bars": FORWARD_BARS,
                        "artifact_free_min_arm": ARTIFACT_FREE_MIN,
                        "features": TIER0, "n_features": len(TIER0),
                        "train_rows": int(len(tr)), "val_oos_holdout_touched": False,
                        "live_code_changed": False,
                        "eth_reference_cell": list(ETH_CELL)},
              "data": meta,
              "train_atr_bp": {"median": round(float(np.median(atr_bp)), 2),
                               "p25": round(float(np.percentile(atr_bp, 25)), 2),
                               "p75": round(float(np.percentile(atr_bp, 75)), 2)},
              "grid": results, "selected_cell": best, "eth_cell_on_btc": eth,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
