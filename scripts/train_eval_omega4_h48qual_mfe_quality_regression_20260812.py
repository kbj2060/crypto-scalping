"""오라클 라벨 문헌 리서치 권장안 4 실행: h48qual `quality_head`를 MFE(Maximum Favorable
Excursion) 연속 회귀로 전환한 TabM 풀 학습. 사전 MI/R^2 게이트(scripts/verify_eth_h48qual_
mfe_quantile_label_mi_r2_gate_20260812.py, 2026-08-12)가 이 세션 유일하게 결정적으로 통과한
라벨×타겟 조합이라 이어서 진행한다 -- 강한 정규화 GBM 기준 VAL R2=+0.08/OOS R2=+0.14,
spearman VAL +0.28/OOS +0.39(둘 다 p<0.001), MAE/실현손익 confound 체크도 통과(MFE 높을수록
MAE 작고 실현손익과 spearman +0.43).

기존 quality-regression 학습 인프라(`train_eval_omega4_quality_regression_20260621.py`,
`ThreeHeadQualityRegTabM` 아키텍처, quantile-relative 게이팅 threshold)를 그대로 재사용하되
`_barrier_quality_targets`만 몽키패치 -- 원래는 매 행마다 omega._try_execution/exit_head.
_continue_to_barrier_net로 barrier replay를 다시 시뮬레이션해서 net_return-MAE_penalty를
계산하지만, 이번엔 build_omega1_2_triple_barrier_labels_20260619.py가 h48_conservative 배리어
계산의 부산물로 이미 저장해둔 tb_long_mfe_h48_conservative/tb_short_mfe_h48_conservative를
그대로 읽어써서 게이트 때 검증한 것과 정확히 같은 타겟을 재현한다(재시뮬레이션 없음, 값 일치
보장). 게이팅 자체는 quantile-relative(threshold=quantile(train 예측분포, q))라 MFE가 항상
>=0(기존 아키텍처가 기대하는 부호있는 [-1,+1] 스케일과 다름)이어도 그대로 작동 -- 스케일 조정
불필요, clip만 걸어 학습 안정성 확보.

FINAL12 피쳐 제약은 h48orig 재현판을 먼저 import해 얻는 전역 omega._load_omega_frames/
omega._numeric_feature_cols 몽키패치 체인(train_eval_omega4_3head_parent72_eth_h48qual_
final12_h384_20260811.py가 원 출처)을 그대로 재사용 -- 게이트와 동일 피쳐셋 보장."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# FINAL12 전역 몽키패치 체인 트리거 (omega._load_omega_frames/_numeric_feature_cols)
import train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811 as h48orig_mod  # noqa: E402,F401

import train_eval_omega4_quality_regression_20260621 as qreg  # noqa: E402

TB_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
MFE_COLS = ["timestamp", "tb_long_mfe_h48_conservative", "tb_short_mfe_h48_conservative"]
MFE_CLIP = 0.08  # 관측 p90 ~0.03의 넉넉한 상한 -- 학습 안정성용, 게이팅은 quantile-relative라 스케일 자체는 무관


def _barrier_quality_targets_mfe(frame: pd.DataFrame, *, fee, slip, cost_mult, mae_lambda, clip, mode):
    """게이트 스크립트와 동일 소스(사전계산 h48_conservative MFE)를 읽어 재시뮬레이션 없이 재현."""
    del fee, slip, cost_mult, mae_lambda, clip  # 원 시그니처 호환용, MFE 경로에선 미사용
    tb = pd.read_csv(TB_DIR / "train_triple_barrier_labels.csv", usecols=MFE_COLS, parse_dates=["timestamp"])
    tb = tb.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    m = frame[["timestamp"]].merge(tb, on="timestamp", how="left")
    assert len(m) == len(frame), "timestamp merge row count mismatch"
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    long_mfe = m["tb_long_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    short_mfe = m["tb_short_mfe_h48_conservative"].to_numpy(dtype=np.float64)
    raw = np.where(action == 1, long_mfe, np.where(action == 2, short_mfe, 0.0))
    raw = np.nan_to_num(raw, nan=0.0)
    target = np.clip(raw, 0.0, MFE_CLIP).astype(np.float32)
    active = int((action != 0).sum())
    active_vals = raw[action != 0]
    diag = {
        "active_rows": active,
        "target_source": "precomputed tb_long_mfe_h48_conservative/tb_short_mfe_h48_conservative (build_omega1_2_triple_barrier_labels_20260619.py, no re-simulation)",
        "target_clip": float(MFE_CLIP),
        "raw_mean": float(active_vals.mean()) if active else 0.0,
        "raw_p10": float(np.quantile(active_vals, 0.10)) if active else 0.0,
        "raw_p50": float(np.quantile(active_vals, 0.50)) if active else 0.0,
        "raw_p90": float(np.quantile(active_vals, 0.90)) if active else 0.0,
        "scaled_mean": float(target.mean()),
        "scaled_p70": float(np.quantile(target, 0.70)),
        "target_mode": "mfe_regression",
        "positive_rate": None,
    }
    return target, diag


qreg._barrier_quality_targets = _barrier_quality_targets_mfe

if __name__ == "__main__":
    defaults = [
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-target-mode", "regression"),
        ("--quality-quantile", "0.70"),
        ("--epochs", "4"),
        ("--max-train-rows", "30000"),
        ("--out-suffix", "h48qual_mfe_quality_reg_20260812"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(qreg.main())
