#!/usr/bin/env python3
"""
ETH 분포적 회귀(distributional regression) 라벨 빌더.

Michańków(2025) "Forecasting Probability Distributions of Financial Returns with Deep
Neural Networks"(arXiv:2508.18921) 레시피: 이산 배리어/threshold 없이, 각 시점의 미래
실현 로그수익률(연속값)을 라벨로 삼고 모델이 그 위에 Normal/Student-t/skewed-Student-t
같은 분포 모수를 custom NLL loss로 직접 추정(CRPS/LPS로 평가)한다. "배리어 선택"이라는,
이 저장소가 반복 실패해온 축 자체를 제거하는 게 핵심 매력이다.

후보①(Directional-Change, build_eth_directional_change_triple_barrier_labels_20260819.py)/
②(CUSUM, build_eth_cusum_triple_barrier_labels_20260819.py)와 동일한 원시 bar를 쓰지만,
방법론이 근본적으로 달라 아래 3가지가 해당 없다:
  - 이벤트 샘플링(DC/CUSUM): "어느 시점을 볼지 고르는" 개념 자체가 없다 — 고정 horizon
    전방수익률 방식이라 사실상 매 bar(all_bars와 동일 사고방식)가 라벨을 가진다.
  - 배리어 자동튜닝(calibrate_barriers): 배리어가 없으니 튜닝할 것도 없다.
  - 이산화(zigzag_action {0,1,2} 매핑): 이 라벨은 discretize하지 않는다 — 그 자체가 이
    방법론의 핵심 매력이라 여기서 다시 이산화하면 의미가 없다.

**따라서 이 산출물은 기존 `_read_labels()`/zigzag_action 계약과 스키마가 다르다 — 그대로
읽히지 않는다.** direction_head를 classification에서 regression(분포 모수 출력)으로
바꾸는 별도 학습 스크립트가 있어야 실제로 쓸 수 있으며, 그건 이 스크립트의 범위 밖이다.
분포족(Normal/Student-t/skewed-Student-t) 선택과 NLL/CRPS 손실함수도 학습 시점 결정이라
라벨 자체에는 필요 없다 — 이 스크립트는 순수 realized forward log-return만 만든다.

라벨만 만든다 — TabM 학습/백테스트/promotion 판정은 이 스크립트의 범위 밖이다.
docs/label_methodology_survey_20260815.md: 이 저장소의 40개 이상 선행 라벨 방법론이
전부 "학습 가능하나 방향 edge 없음"으로 수렴했다 — 이 스크립트의 산출 통계만으로는
edge를 주장할 수 없다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/eth_distributional_regression_return_labels_20260819"

REQUIRED_RAW_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

# 후보①/②의 calibrate_barriers가 두 이벤트 샘플링 방식 모두에서 반복해서 골랐던
# max_hold=24를 포함해, 비교용으로 몇 개 horizon을 같이 계산해둔다. 어느 horizon이
# 맞는지는 라벨 빌드 단계가 아니라 학습 단계에서 판단할 문제라 여기선 선택하지 않는다.
DEFAULT_HORIZONS = (12, 24, 48, 96)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_frame(path: Path, *, expected_year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    missing = sorted(REQUIRED_RAW_COLS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = (
        frame[sorted(REQUIRED_RAW_COLS)]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(expected_year)]:
        raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame


def _parse_int_list(text: str) -> tuple:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--input-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--input-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--horizons", type=str, default=",".join(str(h) for h in DEFAULT_HORIZONS),
                    help="쉼표구분 bar horizon 목록(5분봉 기준). 각각 별도 fwd_logret_h{H} 컬럼으로 계산.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = build_args()
    horizons = _parse_int_list(args.horizons)

    # --- 1. 입력 로드 (후보①②와 동일한 원시 bar, 동일한 _read_frame 패턴) ---
    frames = {
        2024: _read_frame(args.input_2024, expected_year=2024),
        2025: _read_frame(args.input_2025, expected_year=2025),
        2026: _read_frame(args.input_2026, expected_year=2026),
    }

    # --- 2. concat: 각 horizon의 forward return이 연도 경계를 자연스럽게 넘어가야 하므로
    # (2024년 12월 말 bar도 2025년 1월 데이터로 유효한 forward return을 가질 수 있다) per-year
    # 독립 처리보다 concat이 낫다 — 후보①②보다 오히려 더 강한 근거(워밍업/배리어절단 문제가
    # 아니라 "매년 끝 H개 bar를 불필요하게 버리는" 실질적 정보손실을 막는다).
    full = pd.concat([frames[2024], frames[2025], frames[2026]], ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)
    if full["timestamp"].duplicated().any():
        raise RuntimeError("concat 이후 timestamp 중복 발견 — 연도 파일 경계가 겹침")
    if not full["timestamp"].is_monotonic_increasing:
        raise RuntimeError("concat 이후 timestamp가 단조증가하지 않음")

    # --- 3. horizon별 실현 로그수익률 계산 (causal: bar t의 라벨은 t+H의 미래 종가를 봄 —
    # 학습 타겟이므로 미래를 보는 게 정의 자체이며, live feature로는 절대 쓰면 안 됨) ---
    log_close = np.log(full["close"].to_numpy())
    n = len(full)
    out = pd.DataFrame({"timestamp": full["timestamp"]})
    for h in horizons:
        fwd = np.full(n, np.nan, dtype=np.float64)
        if h < n:
            fwd[: n - h] = log_close[h:] - log_close[: n - h]
        out[f"fwd_logret_h{h}"] = fwd
    out["_year"] = out["timestamp"].dt.year

    # --- 4. 연도별 저장 + horizon별 분포 진단 ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_year_summary: dict[str, Any] = {}
    for year in (2024, 2025, 2026):
        year_out = out[out["_year"] == year].drop(columns="_year").reset_index(drop=True)
        out_path = args.out_dir / f"fwd_return_labels_{year}.csv"
        year_out.to_csv(out_path, index=False)

        horizon_stats = {}
        for h in horizons:
            col = year_out[f"fwd_logret_h{h}"].dropna()
            horizon_stats[str(h)] = {
                "n_valid": int(len(col)),
                "n_nan": int(year_out[f"fwd_logret_h{h}"].isna().sum()),
                "mean": float(col.mean()) if len(col) else None,
                "std": float(col.std()) if len(col) else None,
                "skew": float(col.skew()) if len(col) else None,
                "kurtosis_excess": float(col.kurtosis()) if len(col) else None,
                "p05": float(col.quantile(0.05)) if len(col) else None,
                "p50": float(col.quantile(0.50)) if len(col) else None,
                "p95": float(col.quantile(0.95)) if len(col) else None,
            }
        per_year_summary[str(year)] = {
            "n_rows": int(len(year_out)),
            "horizon_stats": horizon_stats,
            "output_path": str(out_path),
        }
        print(f"[{year}] n={len(year_out):,} bar -> {out_path}")
        for h in horizons:
            hs = horizon_stats[str(h)]
            print(f"    h={h:>3}bar  mean={hs['mean']*100:+.4f}%  std={hs['std']*100:.4f}%  "
                  f"skew={hs['skew']:+.3f}  kurt_excess={hs['kurtosis_excess']:+.3f}  nan={hs['n_nan']}")

    # --- 5. report.json ---
    report = {
        "type": "distributional_regression_forward_return_labels",
        "source": "arXiv:2508.18921 (Michańków 2025) — 배리어 없는 realized forward log-return, "
                   "분포족/손실함수는 학습 시점 결정(라벨 자체엔 없음)",
        "params": {
            "horizons_bars": list(horizons),
            "event_sampling": "all_bars(해당없음 — 매 bar가 라벨을 가짐, DC/CUSUM 이벤트 개념 없음)",
            "discretization": "없음 — 연속값 그대로, zigzag_action 계약과 다름",
            "processing_scope": "concat",
            "input_paths": {
                "2024": str(args.input_2024), "2025": str(args.input_2025), "2026": str(args.input_2026),
            },
        },
        "per_year_summary": per_year_summary,
        "known_limitations": [
            "이 산출물은 zigzag_action{0,1,2} 계약과 스키마가 달라 기존 _read_labels()로 바로"
            " 읽히지 않는다 — direction_head를 classification에서 regression(분포 모수 출력)으로"
            " 바꾸는 별도 학습 스크립트가 필요하며, 이는 이 스크립트의 범위 밖이다.",
            "horizon={12,24,48,96}bar 중 어느 것이 적절한지, 어떤 분포족(Normal/Student-t/"
            "skewed-Student-t)을 쓸지는 라벨 빌드 단계가 아니라 학습 단계에서 결정해야 한다"
            " — 이 report의 horizon_stats(mean/std/skew/kurtosis)는 그 결정을 위한 사전 진단"
            " 자료일 뿐, 특정 선택을 추천하지 않는다.",
            "이 report의 라벨 통계만으로는 방향 edge를 주장할 수 없다"
            "(docs/label_methodology_survey_20260815.md — 40개 이상의 선행 라벨 방법론이 전부"
            " '학습 가능하나 방향 edge 없음'으로 수렴) — edge 판정은 별도의 TabM 학습 +"
            " Fresh-Forward 평가 단계의 몫이다.",
        ],
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
