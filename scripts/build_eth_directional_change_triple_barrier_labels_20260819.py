#!/usr/bin/env python3
"""
ETH Directional-Change/intrinsic-time 이벤트 샘플링 + triple-barrier 라벨 빌더.

Razmi & Barak(arXiv:2501.06032) 방법론을 core/event_label_engine.py의
event_method='directional_change'로 실제 ETH 데이터에 처음 적용한다(저장소 전체에서
이 이벤트 방식으로 실전 라벨을 생성한 이력이 이전에 없었음 — 유일한 실행 이력은 엔진
자체의 __main__ 스모크테스트뿐이었다).

기존 zigzag_action_labels_{year}.csv와 동일한 원시 bar(data/splits/year_oos/
training_features_{year}.csv)를 입력으로 써서, 이벤트 샘플링 로직만 다른 통제된 비교가
되게 한다. 배리어는 calibrate_barriers() grid search로 자동튜닝하고(매직넘버 배제),
출력은 이벤트가 발생한 bar에만 라벨을 남기는 sparse 모드로 만든다
(2026-08-19 사용자 승인 — /home/kbj20/.claude/plans/1-velvety-whistle.md 참고).

라벨만 만든다 — TabM 학습/백테스트/promotion 판정은 이 스크립트의 범위 밖이다.
docs/label_methodology_survey_20260815.md: 이 저장소의 40개 이상 선행 라벨 방법론이
전부 "학습 가능하나 방향 edge 없음"으로 수렴했다 — 이 스크립트의 산출 통계만으로는
edge를 주장할 수 없다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/eth_directional_change_triple_barrier_labels_20260819"

REQUIRED_RAW_COLS = {"timestamp", "open", "high", "low", "close", "volume"}

# 표준 3-way 라벨({-1,0,+1}) -> zigzag_action 계약({0=CASH,1=LONG,2=SHORT}).
# +1(상방 배리어 선착)->LONG, -1(하방 배리어 선착)->SHORT, 0(terminal return 부호가
# 정확히 0인 극희귀 케이스)->CASH.
LABEL_TO_ZIGZAG_ACTION = {1: 1, -1: 2, 0: 0}

DIAGNOSTIC_COLS = [
    "touch_type", "realized_ret", "vol_at_entry", "bars_held",
    "trend_tstat", "trend_slope", "trend_horizon",
    "weight_uniqueness", "weight_return_attr", "weight",
]


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


def _load_engine():
    """core/__init__.py가 python-binance 의존 binance_client를 즉시 import해서
    이 dev 셸에서 `import core.*`가 ModuleNotFoundError로 깨진다(확인됨) —
    scripts/diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py
    33-38행과 동일하게 importlib 직접 로드로 우회한다."""
    spec = importlib.util.spec_from_file_location(
        "event_label_engine", ROOT / "core" / "event_label_engine.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _parse_float_grid(text: str) -> tuple:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_int_grid(text: str) -> tuple:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--input-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--input-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--dc-theta", type=float, default=0.004, help="DC 확정 임계값(비율). 엔진 기본값 0.004=0.4%%.")
    p.add_argument("--vol-span", type=int, default=100, help="EWMA 변동성 span(엔진 기본값).")
    p.add_argument("--calib-pt-mult-grid", type=str, default="1.0,1.5,2.0,3.0")
    p.add_argument("--calib-sl-mult-grid", type=str, default="1.0,1.5,2.0,3.0")
    p.add_argument("--calib-max-hold-grid", type=str, default="24,48,96")
    p.add_argument("--calib-target-balance", type=float, default=0.30)
    p.add_argument("--calib-min-events", type=int, default=200)
    p.add_argument("--min-events-per-year", type=int, default=50,
                    help="이 미만이면 RuntimeError(조용히 degenerate 라벨 출력 금지).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = build_args()
    engine = _load_engine()

    # --- 1. 입력 로드 (build_zigzag_action_labels_v2_20260604.py의 _read_frame 패턴) ---
    frames = {
        2024: _read_frame(args.input_2024, expected_year=2024),
        2025: _read_frame(args.input_2025, expected_year=2025),
        2026: _read_frame(args.input_2026, expected_year=2026),
    }

    # --- 2. concat: DC/triple-barrier 둘 다 순방향 전용 알고리즘이라 연도 경계를 이어붙여도
    # 룩어헤드가 생기지 않는다. per-year로 쪼개면 매년 변동성 워밍업 손실이 반복되고 연말
    # 이벤트가 그 해 마지막 bar에서 배리어가 강제로 잘리는 부작용이 있어 concat이 더 안전하다.
    full = pd.concat([frames[2024], frames[2025], frames[2026]], ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)
    if full["timestamp"].duplicated().any():
        raise RuntimeError("concat 이후 timestamp 중복 발견 — 연도 파일 경계가 겹침")
    if not full["timestamp"].is_monotonic_increasing:
        raise RuntimeError("concat 이후 timestamp가 단조증가하지 않음")

    # --- 3. DC 이벤트 + 배리어 자동튜닝(calibrate_barriers) + 라벨 생성 ---
    vol = engine.ewma_volatility(full["close"], span=args.vol_span)
    event_idx, _event_dir = engine.directional_change_events(full["close"].to_numpy(), args.dc_theta)
    print(f"[dc] theta={args.dc_theta} -> {len(event_idx):,}개 이벤트 검출 (전체 {len(full):,} bar)")

    best_barrier = engine.calibrate_barriers(
        full, event_idx, vol,
        pt_mult_grid=_parse_float_grid(args.calib_pt_mult_grid),
        sl_mult_grid=_parse_float_grid(args.calib_sl_mult_grid),
        max_hold_grid=_parse_int_grid(args.calib_max_hold_grid),
        target_balance=args.calib_target_balance,
        min_events=args.calib_min_events,
    )
    print(f"[calibrate_barriers] pt_mult={best_barrier.pt_mult} sl_mult={best_barrier.sl_mult} max_hold={best_barrier.max_hold}")

    cfg = engine.LabelEngineConfig(
        event_method="directional_change",
        dc_theta=args.dc_theta,
        vol_method="ewma",
        vol_span=args.vol_span,
        barrier=best_barrier,
    )
    labels = engine.generate_labels(full, cfg)
    print(f"[generate_labels] {len(labels):,}건 라벨 생성")

    # --- 4. 라벨 매핑 + event_time 기준 연도 재분할 ---
    labels = labels.copy()
    labels["zigzag_action"] = labels["label"].map(LABEL_TO_ZIGZAG_ACTION).astype("int64")
    if labels["zigzag_action"].isna().any():
        raise RuntimeError("label -> zigzag_action 매핑 실패(예상 못한 label 값 존재)")
    labels["_year"] = pd.to_datetime(labels["event_time"]).dt.year

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_year_summary: dict[str, Any] = {}
    for year in (2024, 2025, 2026):
        year_labels = labels[labels["_year"] == year].copy()
        n_events = len(year_labels)
        if n_events < args.min_events_per_year:
            raise RuntimeError(
                f"{year}: 이벤트 {n_events}건 < 최소 {args.min_events_per_year} — "
                f"dc_theta/배리어 그리드 재검토 필요(조용히 degenerate 라벨 출력 금지)"
            )

        out_cols = ["event_time", "zigzag_action"] + DIAGNOSTIC_COLS
        out_frame = (
            year_labels[out_cols]
            .rename(columns={"event_time": "timestamp"})
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        # 파일명을 zigzag_action_labels_{year}.csv 계약과 정확히 맞춘다 — _read_labels()가
        # 이 파일명을 하드코딩해서 찾기 때문에(scripts/train_eval_omega4_3head_parent72_loose_
        # entry_quality_20260620.py:70), 다른 이름을 쓰면 --direction-label-dir로 이 디렉토리를
        # 가리켜도 소비자가 파일을 못 찾는다. 이 라벨 계열은 이미 별도 out_dir에 격리되므로
        # zig075/h48qual 프로덕션 라벨과의 충돌은 디렉토리 분리만으로 충분히 방지된다.
        out_path = args.out_dir / f"zigzag_action_labels_{year}.csv"
        out_frame.to_csv(out_path, index=False)

        label_dist = year_labels["zigzag_action"].value_counts(normalize=True).sort_index()
        touch_dist = year_labels["touch_type"].value_counts(normalize=True)
        class_fracs = {int(k): float(v) for k, v in label_dist.items()}
        min_class_frac = min(class_fracs.get(1, 0.0), class_fracs.get(2, 0.0))
        per_year_summary[str(year)] = {
            "n_events": int(n_events),
            "label_dist": {str(k): v for k, v in class_fracs.items()},
            "touch_type_dist": {str(k): float(v) for k, v in touch_dist.items()},
            "mean_bars_held": float(year_labels["bars_held"].mean()),
            "timeout_frac": float((year_labels["touch_type"] == "timeout").mean()),
            "min_class_frac": min_class_frac,
            "output_path": str(out_path),
        }
        print(f"[{year}] n={n_events:,} dist={class_fracs} -> {out_path}")

    # --- 5. report.json ---
    report = {
        "type": "directional_change_triple_barrier_labels",
        "source": "arXiv:2501.06032 (Razmi & Barak) via core/event_label_engine.py event_method='directional_change'",
        "params": {
            "dc_theta": args.dc_theta,
            "vol_method": "ewma",
            "vol_span": args.vol_span,
            "barrier_mode": "calibrated",
            "calibrated_barrier": {
                "pt_mult": best_barrier.pt_mult,
                "sl_mult": best_barrier.sl_mult,
                "max_hold": best_barrier.max_hold,
            },
            "calib_grids": {
                "pt_mult_grid": _parse_float_grid(args.calib_pt_mult_grid),
                "sl_mult_grid": _parse_float_grid(args.calib_sl_mult_grid),
                "max_hold_grid": _parse_int_grid(args.calib_max_hold_grid),
                "target_balance": args.calib_target_balance,
                "min_events": args.calib_min_events,
            },
            "event_density_mode": "sparse",
            "processing_scope": "concat",
            "input_paths": {
                "2024": str(args.input_2024), "2025": str(args.input_2025), "2026": str(args.input_2026),
            },
        },
        "per_year_summary": per_year_summary,
        "total_events": int(len(labels)),
        "known_limitations": [
            "generate_labels()가 directional_change_events()의 event_dir(반전 방향) 반환값을 버린다"
            "(engine 595행: `event_idx, _ = directional_change_events(...)`) — 최종 라벨의 방향은 DC"
            " 반전 방향이 아니라 그 이후 triple-barrier 해소 방향(어느 배리어에 먼저 닿았는지)으로"
            " 결정된다. 'Directional-Change' 이름과 달리 DC 자체는 순수 샘플링 시점 결정 역할만 한다.",
            "sparse 출력이므로 하류 학습 파이프라인의 _align()(inner-join)과 결합 시 표본이 이벤트"
            " 수준으로 조용히 축소된다 — 정확한 축소 비율은 스모크테스트 단계에서 측정한다.",
            "label==0(zigzag_action=CASH 대응)은 실측상 사실상 발생하지 않는다 — 이 라벨 계열은"
            " zigzag_action의 CASH 우세 분포와 근본적으로 다른 분포 형태를 가진다.",
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
