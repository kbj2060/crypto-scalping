#!/usr/bin/env python3
"""매 봉 라벨링의 건전성 점검 -- 사건(event) 단위 클러스터링 감사.

9-5 결론("매 봉 설계에서는 라벨 생성에도 트리거가 필요 없다 -- label_side()는 순수 가격 산술")의
전제가 실제로 건전한지 검증한다. 트리거 없이 전 봉에 라벨을 붙이면 V자반등이 TRAIN+VAL에서
32,499건 나오는데(현행 9트리거 게이트는 7,016건=21.6%만 포착), 이 숫자가

  (a) 진짜로 트리거가 놓친 별개 사건들인가,   <- recall 이득이 진짜
  (b) 같은 실제 저점 주변 봉들이 중복 카운트된 것인가  <- recall 이득이 허수

를 가른다. 이 저장소는 연속형 신호에서 정확히 이 착시를 이미 겪었다
(eth_evidence_signal_v_rebound_live_window_check_20260901: kalman 원시 32봉 -> 실제 12사건,
"원시 발동봉 카운트가 독립 표본크기를 구조적으로 과대추정한다").

## 방법

라벨=v_rebound인 봉들을 GAP봉 이내 인접하면 같은 사건으로 묶는다(side별 독립). 각 사건에 대해
9트리거 중 하나라도 발동한 봉을 포함하는지 확인 -> 사건 단위 recall을 계산한다. GAP은 6/12/24로
민감도를 본다(단일 GAP 최적점을 신뢰하지 않는 이 저장소 관례).

추가로 미포착 사건들의 특성을 본다: 사건 길이 분포, fast_mult 분포(포착/미포착 비교) -- 미포착
쪽이 구조적으로 약한 사건들이면 "게이트가 약한 것만 놓쳤다"는 뜻이라 recall 이득의 가치가 낮다.

⚠️ 진단 전용: 라벨식 V0 그대로, 라이브 코드 변경 없음, OOS/HOLDOUT 미터치(TRAIN+VAL).

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VARIANT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_label_redesign_variant_screen_20260901.py"
_vspec = importlib.util.spec_from_file_location("label_variants_eventaudit_20260901", VARIANT_SCRIPT)
_vs = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vs)

OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/every_bar_label_event_audit_report.json"

GAP_GRID = [6, 12, 24]
ALL9 = _vs.ALL9
W = _vs.W
FAST_BARS = _vs.FAST_BARS
ATR_MULT = _vs.ATR_MULT


def log(msg: str) -> None:
    print(f"[event_audit] {msg}", flush=True)


def cluster_events(idx: np.ndarray, gap: int) -> list[np.ndarray]:
    """Consecutive labeled bars within `gap` of each other = one event."""
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > gap) + 1
    return np.split(idx, splits)


def fast_mult_array(sig: pd.DataFrame, is_down: bool) -> np.ndarray:
    """Reproduces label_side()'s fast_mult for diagnostics (same offsets as the label)."""
    close = sig["close"].to_numpy()
    atr = sig["atr"].to_numpy()
    extreme = (sig["low"] if is_down else sig["high"]).to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    if is_down:
        fm = _vs.fwd_window(close, 1, FAST_BARS, "max") - extreme
    else:
        fm = extreme - _vs.fwd_window(close, 1, FAST_BARS, "min")
    with np.errstate(invalid="ignore", divide="ignore"):
        return fm / pre_atr


def main() -> int:
    t0 = time.time()
    sig = _vs.build_base()
    n = len(sig)

    log("computing V0 labels on ALL bars, both sides...")
    st = {"bottom": _vs.label_variant(sig, True, "wick", 0),
          "top": _vs.label_variant(sig, False, "wick", 0)}
    fm = {"bottom": fast_mult_array(sig, True), "top": fast_mult_array(sig, False)}

    trig = {}
    for side in ("bottom", "top"):
        trig[side] = np.any([sig[f"{side}_{nm}"].fillna(False).to_numpy() for nm in ALL9], axis=0)

    results = {}
    for gap in GAP_GRID:
        log(f"=== GAP={gap} bars ({gap*5}분) ===")
        per_side = {}
        tot_ev = tot_cov = 0
        for side in ("bottom", "top"):
            v_idx = np.flatnonzero(st[side] == "v_rebound")
            events = cluster_events(v_idx, gap)
            covered = [ev for ev in events if trig[side][ev].any()]
            uncovered = [ev for ev in events if not trig[side][ev].any()]
            lens = np.array([len(ev) for ev in events])
            fm_cov = np.concatenate([fm[side][ev] for ev in covered]) if covered else np.array([])
            fm_unc = np.concatenate([fm[side][ev] for ev in uncovered]) if uncovered else np.array([])
            per_side[side] = {
                "raw_labeled_bars": int(len(v_idx)),
                "distinct_events": int(len(events)),
                "bars_per_event_mean": round(float(lens.mean()), 2) if len(lens) else None,
                "bars_per_event_median": float(np.median(lens)) if len(lens) else None,
                "events_covered_by_any_trigger": int(len(covered)),
                "events_uncovered": int(len(uncovered)),
                "event_level_recall": round(len(covered) / len(events), 4) if events else None,
                "bar_level_recall": round(float(trig[side][v_idx].mean()), 4) if len(v_idx) else None,
                "fast_mult_median_covered": round(float(np.nanmedian(fm_cov)), 3) if len(fm_cov) else None,
                "fast_mult_median_uncovered": round(float(np.nanmedian(fm_unc)), 3) if len(fm_unc) else None,
            }
            tot_ev += len(events)
            tot_cov += len(covered)
            p = per_side[side]
            log(f"  [{side}] 원시 라벨봉 {p['raw_labeled_bars']:6d} -> 별개사건 {p['distinct_events']:5d} "
                f"(사건당 평균 {p['bars_per_event_mean']}봉)")
            log(f"           트리거 포착사건 {p['events_covered_by_any_trigger']:5d} / 미포착 {p['events_uncovered']:5d} "
                f"-> 사건단위 recall {p['event_level_recall']}  (봉단위 {p['bar_level_recall']})")
            log(f"           fast_mult 중앙값: 포착 {p['fast_mult_median_covered']} vs 미포착 {p['fast_mult_median_uncovered']}")

        n_days = (sig["timestamp"].iloc[-1] - sig["timestamp"].iloc[0]).days
        results[f"gap_{gap}"] = {
            "per_side": per_side,
            "total_events": tot_ev, "total_covered": tot_cov,
            "combined_event_level_recall": round(tot_cov / tot_ev, 4) if tot_ev else None,
            "events_per_day": round(tot_ev / n_days, 2) if n_days else None,
        }
        log(f"  [합계] 별개사건 {tot_ev} (하루 {results[f'gap_{gap}']['events_per_day']}건), "
            f"사건단위 recall {results[f'gap_{gap}']['combined_event_level_recall']}")

    report = {
        "signal": "v_rebound_every_bar_label_event_audit", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "label_formula": "V0 unchanged (current live label_side())",
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "population": "ALL bars, TRAIN+VAL (timestamp < 2026-01-01)",
            "purpose": ("Verify the 9-5 premise that triggers are unnecessary for LABEL generation, "
                        "by checking whether every-bar labeling's 32,499 v_rebound bars are distinct "
                        "events (real recall gain) or clustered duplicates of the same bottoms."),
        },
        "gap_grid": GAP_GRID,
        "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
