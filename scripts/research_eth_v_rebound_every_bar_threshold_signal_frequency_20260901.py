#!/usr/bin/env python3
"""임계값별 **실제 화면 체감 빈도** 측정 — "0.60이면 신호가 너무 없나?"에 답하기 위한 것.

## 왜 기존 호출률로는 답이 안 되나

`backtest_eth_v_rebound_every_bar_tabpfn_costgate_threshold_20260901.py`의 호출률
(thr=0.60에서 VAL 1.31%)은 **라벨이 붙은 행(v_rebound 또는 chop) 기준**이다. 그런데 라이브 칩은
라벨 유무와 무관하게 **모든 봉**을 채점하고, 한 봉에 대해 bottom/top 두 점수 중 **높은 쪽만**
하나 보여준다. 그래서 화면 체감 빈도는 그 숫자와 다르다:
  - 분모가 다르다(라벨 있는 행 -> 전체 봉)
  - 봉당 2행이 1개 표시로 합쳐진다(max 집계)

이 스크립트는 **라이브가 실제로 하는 것과 동일한 집계**로 다시 센다:
  1. 동결 컨텍스트 + TabPFN으로 fit(라이브와 같은 fit)
  2. VAL/OOS의 **모든 봉**을 양방향 채점
  3. 봉마다 max(bottom, top) — 라이브 `best_by_pos` 로직과 동일
  4. 임계값별로 "며칠에 한 번", "하루 몇 개", "신호 간 간격", 그리고 칩의 48봉(4시간)
     히스토리 스트립에 **평균 몇 칸이 칠해지는지**까지 계산

## 판단 기준 (화면 신호로서)

너무 잦으면 신호 가치가 희석되고, 너무 드물면 화면이 계속 비어 재량 판단에 못 쓴다.
이 대시보드의 다른 증거신호 칩들과 비교 가능하도록 같은 단위(하루당 개수)로 낸다.

⚠️ VAL/OOS만 사용. HOLDOUT 미터치. 라이브 코드 변경 없음. 읽기 전용 측정.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_every_bar_threshold_signal_frequency_20260901.py
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

FEAS = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_spec = importlib.util.spec_from_file_location("everybar_feas_freq", FEAS)
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

import live_eth_sweep_v_rebound_signal_20260829 as _live  # noqa: E402

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
TRAIN_CONTEXT_CSV = _live.TRAIN_CONTEXT_CSV
LIVE_SEED = 20260829
HISTORY_BARS = _live.HISTORY_BARS  # 칩 히스토리 스트립 길이(48봉 = 4시간)

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")  # HOLDOUT은 이 이후 -- 여기서 잘라 미터치 보장

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
OUT_JSON = ROOT / "data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/signal_frequency.json"


def log(msg: str) -> None:
    print(f"[freq] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    log("building all-bar long frame (라벨 없는 봉도 전부 포함)...")
    _feas.VAL_END = OOS_END
    long = _feas.build_long_frame()
    # ⚠️핵심: label.notna() 필터를 **걸지 않는다**. 라이브는 라벨과 무관하게 모든 봉을 채점한다.
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    ctx = pd.read_csv(TRAIN_CONTEXT_CSV)
    log(f"동결 컨텍스트: {TRAIN_CONTEXT_CSV.name} n={len(ctx)}")
    clf = TabPFNClassifier(device="cuda", random_state=LIVE_SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())

    results = {}
    for split in ("VAL", "OOS"):
        s = long.loc[long["split"] == split].copy()
        s["proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]

        # 라이브 best_by_pos와 동일: 봉마다 bottom/top 중 확률 높은 쪽 하나
        per_bar = s.groupby("timestamp", sort=True)["proba"].max().sort_index()
        n_bars = len(per_bar)
        days = (per_bar.index.max() - per_bar.index.min()).total_seconds() / 86400
        labeled = s.loc[s["label"].notna()]
        log("")
        log(f"=== {split}: 채점봉 {n_bars:,}개 ({days:.0f}일), 라벨있는 행 {len(labeled):,}/{len(s):,} "
            f"({len(labeled)/len(s)*100:.1f}%) ===")

        rows = {}
        for thr in THRESHOLDS:
            fired = per_bar >= thr
            n_fire = int(fired.sum())
            per_day = n_fire / days
            # 신호 사이 간격(시간). 연속 발동은 한 덩어리로 묶어 "사건" 수도 같이 센다.
            idx = np.flatnonzero(fired.to_numpy())
            if len(idx) >= 2:
                gaps_bars = np.diff(idx)
                gap_median_h = float(np.median(gaps_bars) * 5 / 60)
                n_events = 1 + int((gaps_bars > 1).sum())  # 연속 발동 = 같은 사건
            else:
                gap_median_h, n_events = float("nan"), n_fire
            # 칩 히스토리 스트립(48봉=4시간)에 평균 몇 칸이 칠해지나
            strip_avg = float(fired.rolling(HISTORY_BARS).sum().mean())
            # 하루라도 신호가 하나도 없는 날의 비율
            by_day = fired.groupby(fired.index.date).any()
            dry_day_pct = float((~by_day).mean() * 100)

            rows[f"{thr:.2f}"] = {
                "n_bars_fired": n_fire, "pct_of_bars": round(n_fire / n_bars * 100, 2),
                "per_day": round(per_day, 2), "n_events": n_events,
                "events_per_day": round(n_events / days, 2),
                "median_gap_hours": round(gap_median_h, 1) if gap_median_h == gap_median_h else None,
                "strip_avg_colored_of_48": round(strip_avg, 1),
                "dry_day_pct": round(dry_day_pct, 1),
            }
            r = rows[f"{thr:.2f}"]
            log(f"  thr={thr:.2f}  발동봉 {n_fire:>5,}({r['pct_of_bars']:5.2f}%)  "
                f"하루 {r['per_day']:5.2f}봉 / 사건 {r['events_per_day']:4.2f}건  "
                f"간격중앙값 {r['median_gap_hours']}h  "
                f"4h스트립 평균 {r['strip_avg_colored_of_48']}/48칸  "
                f"신호없는날 {r['dry_day_pct']:4.1f}%")

        results[split] = {"n_bars": n_bars, "days": round(days, 1),
                          "labeled_row_pct": round(len(labeled) / len(s) * 100, 1),
                          "thresholds": rows}

    report = {
        "signal": "v_rebound_every_bar_threshold_signal_frequency", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"model": "TabPFN, 배포판과 동일 fit", "aggregation": "봉당 max(bottom,top) -- 라이브 best_by_pos와 동일",
                  "population": "ALL bars incl. unlabeled (라이브가 실제로 채점하는 모집단)",
                  "holdout_touched": False, "live_code_changed": False,
                  "purpose": "임계값별 화면 체감 빈도 -- '0.60이면 너무 드문가' 판단용"},
        "context_csv": str(TRAIN_CONTEXT_CSV.relative_to(ROOT)), "seed": LIVE_SEED,
        "history_bars": HISTORY_BARS, "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
