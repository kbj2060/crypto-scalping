#!/usr/bin/env python3
"""V자반등 9트리거 라이브 frozen context를 6,000행 서브샘플 -> 전체 TRAIN으로 교체.

## 왜

배포판은 TRAIN 전체가 아니라 6,000행 무작위 서브샘플을 frozen context로 쓴다. 그 선택의 이유는
성능이 아니라 **레이턴시**였다("전체TRAIN은 레이턴시상 너무 큼", 서버 실측 fit+predict 2.88s).

2026-09-01 서버 GPU 실측(`research_eth_v_rebound_pool_a_context_size_tabpfn_20260901.py`,
TabPFN 4시드, VAL 3,500건)에서 그 대가가 드러났다:

| 컨텍스트 | VAL AUC | 라이브 사이클 |
|---|---|---|
| 6,000 (현행) | 0.8204±0.0013 | 2.20s |
| 9,000 | 0.8236±0.0012 | 2.76s |
| 12,000 | 0.8262±0.0013 | 3.90s |
| **전체 17,969** | **0.8290±0.0006** | 6.59s |

- **+0.0086 AUC** (시드 std 0.0006~0.0013의 7~14배 = 확실한 실질 이득)
- 레이턴시 6.59s는 이 엔드포인트의 캐시 주기(`EVIDENCE_SIGNAL_CACHE_SECONDS = 60`)의 11%,
  봉 간격(5분)의 2% -- 원래 "너무 크다"는 판단은 실제 예산 대비로는 여유로웠다.
- **부수 효과: 인용 수치와 실제 배포 성능의 불일치가 해소된다.** 지금까지 인용돼온 VAL AUC
  0.8292는 cheap_gate가 **전체 TRAIN**으로 측정한 값인데(위 표의 full 0.8290과 일치), 실제
  배포된 6,000행 모델은 0.8204다. 약 0.009 부풀려진 수치가 계속 인용돼왔다.

## 하는 일

배포된 학습 데이터(`eth_5m_v_rebound_multitrigger_features_tier0.csv`)에서 기존 6,000행
컨텍스트와 **완전히 같은 스키마/필터**로 전체 TRAIN을 뽑아 새 CSV를 만든다. 서브샘플링만 뺀다.

- 라벨: outcome이 "V자반등"(1) / "지지/횡보"(0)인 행만 (애매 제외 -- 기존과 동일)
- 기간: timestamp < 2025-09-01 (TRAIN split)
- 컬럼: timestamp + label + FEATURES 23개 (기존 CSV와 byte-identical한 컬럼 순서)
- **재균형 안 함** -- 자연 라벨비율 그대로(검증 수치가 그 비율 기준이므로, 기존 결정과 동일)

## ⚠️ 라이브 스크립트에 필요한 동반 변경

`live_eth_sweep_v_rebound_signal_20260829.py`의 TabPFNClassifier 호출에
**`ignore_pretraining_limits=True`를 추가해야 한다** -- TabPFN 권장 상한이 1만행이라 17,961행
컨텍스트는 이 플래그 없이는 거부되거나 경고된다. 검증 파이프라인(cheap_gate/seed_stability/
holdout)은 이미 이 플래그를 쓰고 있었다.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/freeze_eth_v_rebound_multitrigger_train_context_full_20260901.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831"
FEATURES_CSV = DATA_DIR / "eth_5m_v_rebound_multitrigger_features_tier0.csv"
OLD_CONTEXT = DATA_DIR / "tabpfn_train_context_frozen_multitrigger_v1_20260831.csv"
NEW_CONTEXT = DATA_DIR / "tabpfn_train_context_frozen_multitrigger_full_20260901.csv"

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")


def log(msg: str) -> None:
    print(f"[freeze_full] {msg}", flush=True)


def main() -> int:
    old = pd.read_csv(OLD_CONTEXT)
    cols = list(old.columns)  # 기존 스키마를 그대로 따른다(순서 포함)
    log(f"기존 컨텍스트: n={len(old)}, 라벨비율={old['label'].mean():.4f}, 컬럼 {len(cols)}개")

    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    labeled = df.loc[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    labeled["label"] = (labeled["outcome"] == "V자반등").astype(int)
    feat_cols = [c for c in cols if c not in ("timestamp", "label")]
    labeled = labeled.dropna(subset=feat_cols)

    train = labeled.loc[labeled["timestamp"] < TRAIN_END].sort_values("timestamp").reset_index(drop=True)
    out = train[cols].copy()
    log(f"새 컨텍스트: n={len(out)}, 라벨비율={out['label'].mean():.4f} "
        f"({out['timestamp'].min()} ~ {out['timestamp'].max()})")

    # 정합성 확인 -- 기존 6,000행이 새 전체 집합의 부분집합이어야 한다(같은 모집단에서 뽑았으므로)
    old_ts = set(pd.to_datetime(old["timestamp"], utc=True))
    new_ts = set(out["timestamp"])
    missing = old_ts - new_ts
    log(f"self-check: 기존 6,000행 중 새 집합에 없는 행 = {len(missing)}건 (0이어야 정상)")
    if missing:
        log(f"  ⚠️예시: {sorted(missing)[:3]}")

    assert list(out.columns) == cols, "컬럼 스키마 불일치"
    out.to_csv(NEW_CONTEXT, index=False)
    log(f"saved -> {NEW_CONTEXT} ({NEW_CONTEXT.stat().st_size/1e6:.1f}MB)")

    report = {
        "old_context": {"path": str(OLD_CONTEXT.relative_to(ROOT)), "n": int(len(old)),
                        "label_rate": round(float(old["label"].mean()), 4)},
        "new_context": {"path": str(NEW_CONTEXT.relative_to(ROOT)), "n": int(len(out)),
                        "label_rate": round(float(out["label"].mean()), 4),
                        "first": str(out["timestamp"].min()), "last": str(out["timestamp"].max())},
        "schema_columns": cols,
        "self_check_old_rows_missing_from_new": len(missing),
        "rebalanced": False,
        "measured_gain": {"val_auc_6000": 0.8204, "val_auc_full": 0.8290, "delta": 0.0086,
                          "live_cycle_sec_6000": 2.20, "live_cycle_sec_full": 6.59,
                          "source": "research_eth_v_rebound_pool_a_context_size_tabpfn_20260901.py (서버 GPU, TabPFN 4시드)"},
        "requires_live_change": "TabPFNClassifier(..., ignore_pretraining_limits=True) -- 1만행 상한 우회",
    }
    (DATA_DIR / "train_context_full_20260901_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    log("report saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
