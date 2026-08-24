#!/usr/bin/env python3
"""dc_theta 스윕 — 라벨 경제성만 확인, 학습은 아직 안 함 (cusum_k 스윕과 동일한 "싸게 먼저
본다" 절차).

배경: memory(`eth_tabm_label_logic_retest_initiative_20260819`)에 이미 기록된 미실행
권장사항 — theta=0.002(더 촘촘하게)는 이미 시도해 기각됐고("빈도를 높이는 방향은 경제성을
개선 못 함"), "향후엔 theta를 낮추는 게 아니라 올리는 쪽(예 0.008~0.015)을 봐야 함"이라고
적어두고 실행은 안 한 상태였다. `build_eth_directional_change_triple_barrier_labels_20260819.py`
(기존 검증된 빌더)와 완전히 같은 이벤트검출+calibrate_barriers+generate_labels 로직을
그대로 재사용하되, 여러 theta를 한 프로세스 안에서 순회하며 라벨 CSV는 안 쓰고 요약 통계만
낸다(각 theta마다 파일 I/O할 필요 없음 -- 실제 채택할 값이 정해지면 그때 정식 빌더로 다시
CSV를 만든다).

비용 기준: FEE_RATE=5bp/side, SLIP_RATE=2bp/side, MAKER_FEE_MULT=0.20
(train_eval_omega1_2_tabm_diffusion_risk_20260603.py:47-49, 이전에 이미 이 축에서 검증된
실제 코드 기준) -> 왕복비용 6bp(양쪽메이커)~14bp(양쪽테이커). TP폭은 pt_mult*vol_at_entry."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RAW_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
COST_BEST_BP = 6.0   # 양쪽 메이커
COST_WORST_BP = 14.0  # 양쪽 테이커
THETA_GRID = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]  # 0.004=기존(참고용), 0.002는 이미 기각


def _load_engine():
    spec = importlib.util.spec_from_file_location("event_label_engine", ROOT / "core" / "event_label_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_frame(path: Path, *, expected_year: int) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    frame = frame[sorted(REQUIRED_RAW_COLS)].dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    assert years == [int(expected_year)], f"{path}: {years}"
    return frame


def main() -> None:
    engine = _load_engine()
    frames = {
        2024: _read_frame(ROOT / "data/splits/year_oos/training_features_2024.csv", expected_year=2024),
        2025: _read_frame(ROOT / "data/splits/year_oos/training_features_2025.csv", expected_year=2025),
        2026: _read_frame(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv", expected_year=2026),
    }
    full = pd.concat([frames[2024], frames[2025], frames[2026]], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    vol = engine.ewma_volatility(full["close"], span=100)

    results = []
    for theta in THETA_GRID:
        event_idx, _ = engine.directional_change_events(full["close"].to_numpy(), theta)
        n_events_raw = len(event_idx)
        best_barrier = engine.calibrate_barriers(
            full, event_idx, vol,
            pt_mult_grid=(1.0, 1.5, 2.0, 3.0), sl_mult_grid=(1.0, 1.5, 2.0, 3.0), max_hold_grid=(24, 48, 96),
            target_balance=0.30, min_events=200,
        )
        cfg = engine.LabelEngineConfig(event_method="directional_change", dc_theta=theta, vol_method="ewma", vol_span=100, barrier=best_barrier)
        labels = engine.generate_labels(full, cfg)

        tp_width_bp = (best_barrier.pt_mult * labels["vol_at_entry"] * 10000.0)
        sl_width_bp = (best_barrier.sl_mult * labels["vol_at_entry"] * 10000.0)
        n = len(labels)
        avg_spacing_min = float((labels["event_time"].max() - labels["event_time"].min()).total_seconds() / 60.0 / max(n - 1, 1))
        mean_bars_held = float(labels["bars_held"].mean())
        avg_hold_min = mean_bars_held * 5.0  # 5분봉

        row = {
            "theta": theta, "n_events": int(n), "n_events_raw_dc": int(n_events_raw),
            "pt_mult": best_barrier.pt_mult, "sl_mult": best_barrier.sl_mult, "max_hold": best_barrier.max_hold,
            "tp_width_bp_median": float(tp_width_bp.median()), "tp_width_bp_p10": float(tp_width_bp.quantile(0.10)),
            "tp_width_bp_p90": float(tp_width_bp.quantile(0.90)),
            "pct_below_worst_14bp": float((tp_width_bp < COST_WORST_BP).mean() * 100),
            "pct_below_best_6bp": float((tp_width_bp < COST_BEST_BP).mean() * 100),
            "avg_spacing_min": avg_spacing_min, "avg_hold_min": avg_hold_min,
            "spacing_lt_hold": avg_spacing_min < avg_hold_min,
            "label_dist": {int(k): float(v) for k, v in labels["label"].value_counts(normalize=True).items()},
        }
        results.append(row)
        print(f"theta={theta}: n={n:,} pt_mult={best_barrier.pt_mult} sl_mult={best_barrier.sl_mult} "
              f"max_hold={best_barrier.max_hold} TPwidth_med={row['tp_width_bp_median']:.1f}bp "
              f"(p10={row['tp_width_bp_p10']:.1f}/p90={row['tp_width_bp_p90']:.1f}) "
              f"<14bp={row['pct_below_worst_14bp']:.1f}% <6bp={row['pct_below_best_6bp']:.1f}% "
              f"간격{avg_spacing_min:.1f}분/보유{avg_hold_min:.1f}분 "
              f"{'[간격<보유!]' if row['spacing_lt_hold'] else ''}", flush=True)

    out_path = ROOT / "tmp/eth_dc_theta_sweep_20260820.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
