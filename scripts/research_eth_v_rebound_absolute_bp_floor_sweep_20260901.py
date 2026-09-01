#!/usr/bin/env python3
"""V_REBOUND 라벨에 절대 bp 하한 추가 -- 스윕 (GBM 프록시, CPU only).

## 동기 (9-8 육안검증에서 사용자가 직접 발견)

사용자가 미포착 사건 예시 하나(2025-12-27 20:55)를 짚어 "이게 왜 트리거에 안 잡혔냐"고 물었고,
까보니 **9개 트리거 전부 문턱 근처에도 못 갔다** -- 애초에 극단적인 일이 없었기 때문. ATR이
1.471(=5.0bp)로 시장이 사실상 죽어 있었고, 그래서 라벨의 `fast_mult >= 1.5*ATR` 문턱이 겨우
7.5bp로 내려앉아 **8.5bp 노이즈에 "V자반등" 라벨이 붙었다**.

후속 측정: 죽은시장(ATR<10bp) 사건은 반등폭 중앙값이 포착 19.2bp / 미포착 18.1bp로 **왕복비용
10bp의 2배도 안 된다**(20bp 미만이 57~64%). 이건 미포착 사건만의 문제가 아니라 **지금 배포된
모델이 학습한 포착 사건에도 4.0% 들어있는, 라벨 자체의 선재 결함**이다.

## 이 스크립트가 하는 것

`is_v` 조건에 절대 하한을 AND로 추가:

    is_v = (fast_mult >= 1.5) AND (giveback <= 0.20) AND (fast_move_bp >= FLOOR)

FLOOR를 못 넘는 후보는 chop(0)이 아니라 **ambiguous(제외)**로 보낸다 -- 그 봉은 실제로 1.5xATR을
움직였으므로 "반등 실패"라고 가르치면 틀린 학습이 된다. v7b의 "애매하면 아예 제외" 철학과 동일.

## ⚠️ 평가 설계상의 제약 (반드시 지켜야 함)

FLOOR가 달라지면 **라벨 정의 자체가 달라지므로 각 FLOOR의 자기 AUC를 서로 직접 비교하면 안 된다**
(feedback_cross_model_auc_comparison_requires_matched_label_difficulty_20260901: 문제 난이도
차이를 성능 차이로 오인). 따라서:

  - 모든 모델을 **동일한 고정 타겟**(REF_FLOOR=30bp 라벨, 경제적으로 가장 의미있는 양성만)에서
    평가한다 -> 사과 대 사과.
  - 각 모델의 "자기 라벨 AUC"도 참고로 찍되 **비교 불가**로 명시한다.

⚠️ 진단 전용: 라이브 코드 변경 없음, OOS/HOLDOUT 미터치(TRAIN < 2025-09-01, VAL 2025-09-01~
2025-12-31). 전 봉 스코어링 설계(9-3) 위에서 측정.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_absolute_bp_floor_sweep_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

FEAS_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_fspec = importlib.util.spec_from_file_location("everybar_feas_floor_20260901", FEAS_SCRIPT)
_feas = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_feas)
_vs = _feas._vs

AUDIT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("event_audit_floor_20260901", AUDIT_SCRIPT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/absolute_bp_floor_sweep_report.json"

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
TRAIN_END = _feas.TRAIN_END
VAL_END = _feas.VAL_END
START = _feas.START
SEEDS = [20260901, 141592, 271828, 577215, 20260829]
W = _vs.W
FAST_BARS = _vs.FAST_BARS
FULL_BARS = _vs.FULL_BARS
ATR_MULT = _vs.ATR_MULT
CHOP_MULT = _vs.CHOP_MULT
T_SUSTAIN = _vs.T_SUSTAIN

FLOORS = [0, 10, 20, 30, 40]
REF_FLOOR = 30  # fixed evaluation target -- economically meaningful positives only
GAP = 12


def log(msg: str) -> None:
    print(f"[bp_floor_sweep] {msg}", flush=True)


def side_fields(sig: pd.DataFrame, is_down: bool) -> dict[str, np.ndarray]:
    close = sig["close"].to_numpy()
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    atr = sig["atr"].to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    extreme = low if is_down else high
    if is_down:
        fast_move = _vs.fwd_window(close, 1, FAST_BARS, "max") - extreme
        peak = _vs.fwd_window(high, 1, FULL_BARS, "max")
    else:
        fast_move = extreme - _vs.fwd_window(close, 1, FAST_BARS, "min")
        peak = _vs.fwd_window(low, 1, FULL_BARS, "min")
    end_price = _vs.shifted_at(close, FULL_BARS)
    full_high_max = _vs.fwd_window(high, 1, FULL_BARS, "max")
    full_low_min = _vs.fwd_window(low, 1, FULL_BARS, "min")
    with np.errstate(invalid="ignore", divide="ignore"):
        fast_mult = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        giveback = np.where(np.abs(denom) >= 1e-12,
                            (peak - end_price) / denom if is_down else (end_price - peak) / denom,
                            np.nan)
        fast_move_bp = fast_move / close * 10000.0
        atr_bp = pre_atr / close * 10000.0
    valid = (np.isfinite(pre_atr) & (pre_atr > 0) & np.isfinite(full_high_max)
             & np.isfinite(full_low_min) & np.isfinite(end_price))
    return {"fast_mult": fast_mult, "giveback": giveback, "fast_move_bp": fast_move_bp,
            "atr_bp": atr_bp, "valid": valid}


def status_with_floor(f: dict[str, np.ndarray], floor_bp: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        is_v = ((f["fast_mult"] >= ATR_MULT) & np.isfinite(f["giveback"])
                & (f["giveback"] <= T_SUSTAIN) & (f["fast_move_bp"] >= floor_bp))
        is_chop = f["fast_mult"] < CHOP_MULT
    return np.where(~f["valid"], "invalid", np.where(is_v, "v_rebound",
                    np.where(is_chop, "chop", "ambiguous")))


def main() -> int:
    t0 = time.time()
    log("building per-bar feature frame + triggers...")
    feat = _feas.build_all_bar_frame()
    eth = _vs.load_klines(_feas.ETH_CSV)
    btc = _vs.load_klines(_feas.BTC_CSV)
    impl = _vs.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()

    fields = {"bottom": side_fields(sig, True), "top": side_fields(sig, False)}
    statuses = {(side, fl): status_with_floor(fields[side], fl)
                for side in ("bottom", "top") for fl in FLOORS}

    # ---- descriptive: what does each floor remove? (event level, TRAIN+VAL) ----
    log("=== 하한별 라벨 분포 변화 (사건 단위, GAP=12) ===")
    in_span = ((sig["timestamp"] >= START) & (sig["timestamp"] < VAL_END)).to_numpy()
    desc = {}
    for fl in FLOORS:
        n_ev = n_bars = 0
        removed_atr = []
        for side in ("bottom", "top"):
            st = statuses[(side, fl)]
            base_st = statuses[(side, 0)]
            v_idx = np.flatnonzero((st == "v_rebound") & in_span)
            n_ev += len(_audit.cluster_events(v_idx, GAP))
            n_bars += len(v_idx)
            rem = np.flatnonzero((base_st == "v_rebound") & (st != "v_rebound") & in_span)
            removed_atr.extend(fields[side]["atr_bp"][rem].tolist())
        base_ev = desc[0]["events"] if 0 in desc else None
        desc[fl] = {
            "events": n_ev, "labeled_pos_bars": n_bars,
            "pct_of_floor0_events": round(n_ev / desc[0]["events"] * 100, 1) if base_ev else 100.0,
            "removed_events_atr_bp_median": round(float(np.median(removed_atr)), 1) if removed_atr else None,
        }
        log(f"  FLOOR={fl:2d}bp  사건 {n_ev:5d} ({desc[fl]['pct_of_floor0_events']:5.1f}% of 하한없음)  "
            f"양성봉 {n_bars:6d}  제거된 양성의 ATR중앙값={desc[fl]['removed_events_atr_bp_median']}bp")

    # ---- build long frame with all floors' labels ----
    log("building long frame (features + all floors' labels)...")
    rows = []
    for side, is_down in (("bottom", True), ("top", False)):
        d = pd.DataFrame({"timestamp": sig["timestamp"]})
        for fl in FLOORS:
            d[f"status_{fl}"] = statuses[(side, fl)]
        d["atr_bp"] = fields[side]["atr_bp"]
        d["fast_move_bp"] = fields[side]["fast_move_bp"]
        merged = d.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
        sub = pd.DataFrame({"timestamp": merged["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        level = merged["sweep_level_low"].to_numpy() if is_down else merged["sweep_level_high"].to_numpy()
        atr = merged["atr"].to_numpy(dtype=float)
        pen = (level - merged["low"].to_numpy()) if is_down else (merged["high"].to_numpy() - level)
        sub["sweep_penetration_atr"] = pen / atr
        sub["atr"] = atr
        sub["atr_percentile_864"] = merged["atr_percentile_864"].to_numpy()
        sub["range_width_pct"] = merged["range_width_pct"].to_numpy()
        sub["hour_utc"] = merged["hour_utc"].to_numpy()
        sub["weekday"] = merged["weekday"].to_numpy()
        dz = merged["delta_z"].to_numpy(dtype=float)
        sub["delta_z"] = dz
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        for col in ["p_fast", "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
                    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi",
                    "bb_width_pctile", "ret3_z", "rsi"]:
            sub[col] = merged[col].to_numpy()
        for fl in FLOORS:
            sub[f"status_{fl}"] = merged[f"status_{fl}"].to_numpy()
        sub["atr_bp"] = merged["atr_bp"].to_numpy()
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.loc[(long["timestamp"] >= START) & (long["timestamp"] < VAL_END)].reset_index(drop=True)
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    for fl in FLOORS:
        long[f"label_{fl}"] = np.where(long[f"status_{fl}"] == "v_rebound", 1.0,
                              np.where(long[f"status_{fl}"] == "chop", 0.0, np.nan))

    # fixed evaluation target: REF_FLOOR label on VAL
    ref_col = f"label_{REF_FLOOR}"
    va_ref = long.loc[(long["split"] == "VAL") & long[ref_col].notna()]
    log(f"고정 평가타겟: FLOOR={REF_FLOOR}bp 라벨, VAL n={len(va_ref)}, "
        f"base={va_ref[ref_col].mean():.4f}")

    results = {}
    for fl in FLOORS:
        tr = long.loc[(long["split"] == "TRAIN") & long[f"label_{fl}"].notna()]
        va_own = long.loc[(long["split"] == "VAL") & long[f"label_{fl}"].notna()]
        log(f"=== FLOOR={fl}bp 학습 (n={len(tr)}, base={tr[f'label_{fl}'].mean():.4f}) ===")
        ref_aucs, own_aucs = [], []
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(random_state=sd, max_iter=300, early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(tr[FEATURE_COLUMNS], tr[f"label_{fl}"].to_numpy())
            ref_aucs.append(roc_auc_score(va_ref[ref_col].to_numpy(),
                                          m.predict_proba(va_ref[FEATURE_COLUMNS])[:, 1]))
            own_aucs.append(roc_auc_score(va_own[f"label_{fl}"].to_numpy(),
                                          m.predict_proba(va_own[FEATURE_COLUMNS])[:, 1]))
        ra, oa = np.array(ref_aucs), np.array(own_aucs)
        results[f"floor_{fl}"] = {
            "train_n": int(len(tr)), "train_base_rate": round(float(tr[f"label_{fl}"].mean()), 4),
            "auc_on_FIXED_ref_target": {"mean": round(float(ra.mean()), 4), "std": round(float(ra.std()), 4)},
            "auc_on_own_label_NOT_COMPARABLE": {"mean": round(float(oa.mean()), 4), "std": round(float(oa.std()), 4),
                                                 "eval_n": int(len(va_own))},
        }
        log(f"  -> 고정타겟(FLOOR{REF_FLOOR}) AUC {ra.mean():.4f} ± {ra.std():.4f}   "
            f"| (참고,비교불가) 자기라벨 AUC {oa.mean():.4f} ± {oa.std():.4f}")

    base_ref = results[f"floor_0"]["auc_on_FIXED_ref_target"]["mean"]
    log("=== 고정타겟 기준, 하한없음(FLOOR=0) 대비 ===")
    for fl in FLOORS:
        d = results[f"floor_{fl}"]["auc_on_FIXED_ref_target"]["mean"] - base_ref
        results[f"floor_{fl}"]["delta_vs_floor0_on_ref_target"] = round(d, 4)
        log(f"  FLOOR={fl:2d}bp  delta={d:+.4f}  (std {results[f'floor_{fl}']['auc_on_FIXED_ref_target']['std']:.4f})")

    report = {
        "signal": "v_rebound_absolute_bp_floor_sweep", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "model": "HistGradientBoostingClassifier proxy (not TabPFN)",
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "population": "ALL bars x both sides (every-bar scoring design)",
            "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. {VAL_END}"},
            "purpose": ("Test adding an absolute bp floor to the V_REBOUND label, motivated by the "
                        "user spotting (in 9-8's chart) that the ATR-normalised label fires on ~8bp "
                        "noise when ATR collapses to ~5bp."),
        },
        "floor_definition": "is_v = (fast_mult>=1.5) AND (giveback<=0.20) AND (fast_move_bp>=FLOOR); sub-floor positives -> ambiguous (excluded), NOT chop",
        "evaluation_note": ("Cross-floor AUC on each floor's OWN label is NOT comparable (different "
                            "label difficulty). All models are therefore also scored on a single "
                            f"FIXED target: the FLOOR={REF_FLOOR}bp label on VAL."),
        "floors": FLOORS, "ref_floor": REF_FLOOR, "seeds": SEEDS,
        "label_distribution_by_floor": desc,
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
