#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- 라이브 frozen context 생성 (전체 봉 TRAIN에서 무작위 18,000행).

## 근거

서버 GPU 실측(`research_eth_v_rebound_every_bar_tabpfn_confirm_20260901.py`, TabPFN 3시드,
VAL 전체봉 15,000행 평가):

| 샘플링 | 6,000 | 12,000 | 18,000 |
|---|---|---|---|
| random | 0.6822 | 0.6907 | **0.6942** |
| stratified | 0.6811 | 0.6894 | 0.6926 |

- **random_18000이 최고**: AUC 0.6942 -- GBM 프록시(0.6953) 대비 −0.0011로 사실상 동일.
  후보풀 학습을 전체봉에 적용했을 때(0.5287) 대비 **+0.1655**.
- 층화추출은 무작위와 유의차 없음(라벨비율 0.1462 vs 0.1463 -- 큰 풀에서 무작위로 뽑으면
  자연비율이 저절로 재현되므로 층화가 추가로 하는 일이 없다). **재균형 안 함** -- 검증 수치가
  자연비율 기준이므로.
- 라이브 사이클 6.56s는 현행 배포판(전체 TRAIN 17,969행, 6.72s)과 같은 수준이고
  이 엔드포인트 캐시 주기(60s)의 11%.
- ⚠️**event-first 샘플링은 쓰지 않는다** -- 사건당 첫 봉만 뽑으면 가장 약하고 비대표적인
  양성만 남아 GBM에서 −0.097로 크게 나빠졌다(2026-09-01 실측).

## 현행(후보풀) 컨텍스트와의 차이

|  | 현행 | 이 스크립트 |
|---|---|---|
| 모집단 | 9트리거 발동봉만(전체의 23.45%) | **전체 봉** |
| TRAIN | 17,969행(전부) | 182,969행에서 무작위 18,000행 |
| 라벨비율 | 32.53% | 14.63% |

라벨비율이 절반 이하로 떨어지는 것은 정상이다 -- 후보풀은 극단이벤트만 모아 놓은 편중
모집단이었고, 전체 봉이 실제 라이브가 마주하는 모집단이다. **다만 확률의 의미가 바뀌므로
0.5 임계값을 그대로 재사용할지는 별도 판단이 필요하다.**

⚠️ TRAIN(< 2025-09-01)만 사용. OOS/HOLDOUT 미터치.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/freeze_eth_v_rebound_every_bar_train_context_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FEAS = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_spec = importlib.util.spec_from_file_location("everybar_feas_freeze", FEAS)
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)

OUT_DIR = ROOT / "data/labels/eth_5m_v_rebound_every_bar_20260901"
OUT_CSV = OUT_DIR / "tabpfn_train_context_frozen_every_bar_20260901.csv"
OLD_CONTEXT = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/tabpfn_train_context_frozen_multitrigger_full_20260901.csv"

CONTEXT_N = 18000
SEED = 20260829  # 라이브 추론 시드와 동일(현행 관례)
FEATURE_COLUMNS = _feas.FEATURE_COLUMNS


def log(msg: str) -> None:
    print(f"[freeze_every_bar] {msg}", flush=True)


def main() -> int:
    old = pd.read_csv(OLD_CONTEXT)
    cols = list(old.columns)  # 기존 스키마(순서 포함)를 그대로 따른다 -- 라이브 코드 무변경 재사용
    log(f"기존(후보풀) 컨텍스트: n={len(old)}, 라벨비율={old['label'].mean():.4f}, 컬럼 {len(cols)}개")

    log("building all-bar long frame...")
    long = _feas.build_long_frame()
    long = long.loc[long["label"].notna()].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    tr = long.loc[long["split"] == "TRAIN"].reset_index(drop=True)
    log(f"전체 봉 TRAIN: n={len(tr)}, 라벨비율={tr['label'].mean():.4f}")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
    ctx = tr.iloc[idx].copy()
    log(f"무작위 추출: n={len(ctx)}, 라벨비율={ctx['label'].mean():.4f} "
        f"(모집단 {tr['label'].mean():.4f} 대비 {ctx['label'].mean()-tr['label'].mean():+.4f})")

    missing = [c for c in cols if c not in ctx.columns]
    if missing:
        log(f"  ⛔ 스키마 불일치, 누락 컬럼: {missing}")
        return 1
    out = ctx[cols].copy()

    # self-check: 라이브가 쓰는 피쳐에 NaN이 없어야 한다(검증 파이프라인과 동일 기준)
    n_nan = int(out[FEATURE_COLUMNS].isna().any(axis=1).sum())
    log(f"self-check: NaN 행 {n_nan}건 (0이어야 정상)")
    if n_nan:
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    log(f"saved -> {OUT_CSV} ({OUT_CSV.stat().st_size/1e6:.1f}MB)")

    report = {
        "context": {"path": str(OUT_CSV.relative_to(ROOT)), "n": int(len(out)),
                    "label_rate": round(float(out["label"].mean()), 4),
                    "first": str(out["timestamp"].min()), "last": str(out["timestamp"].max())},
        "population": {"train_n": int(len(tr)), "train_label_rate": round(float(tr["label"].mean()), 4),
                       "scope": "ALL bars x both sides (every-bar scoring), TRAIN < 2025-09-01"},
        "previous_context": {"path": str(OLD_CONTEXT.relative_to(ROOT)), "n": int(len(old)),
                             "label_rate": round(float(old["label"].mean()), 4),
                             "scope": "9-trigger candidate pool only (23.45% of bars)"},
        "sampling": {"method": "uniform random", "seed": SEED, "rebalanced": False,
                     "rejected_alternatives": {
                         "stratified": "무작위와 유의차 없음(AUC 0.6926 vs 0.6942)",
                         "event_first": "GBM에서 −0.097 -- 가장 약한 양성만 남음"}},
        "measured": {"tabpfn_val_auc": 0.6942, "tabpfn_val_auc_std": 0.0023,
                     "gbm_proxy_reference": 0.6953,
                     "candidate_pool_model_on_all_bars": 0.5287,
                     "live_cycle_sec": 6.56,
                     "source": "research_eth_v_rebound_every_bar_tabpfn_confirm_20260901.py"},
        "schema_columns": cols, "self_check_nan_rows": n_nan,
        "caveat": ("라벨비율이 32.53%->14.63%로 바뀌므로 확률의 의미가 달라진다. "
                   "0.5 임계값을 그대로 재사용할지는 별도 판단 필요."),
    }
    (OUT_DIR / "context_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log("report saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
