#!/usr/bin/env python3
"""TabM DC 5시드 딥앙상블(예측확률 평균) 실증검증.

⚠️ASWA가 아니라 딥앙상블(여러 독립 시드의 예측을 결합)을 구현한다 -- 사용자가 인용한 두
기법(ASWA/NASWA 대 bagging/deep ensembles)은 서로 다르다. ASWA는 "한 번의 학습 궤적 안"의
여러 체크포인트 가중치를 평균해 그 궤적의 진동을 매끈하게 만드는 기법인데, epoch30 테스트
(scripts/eth_directional_change_tabm_training_unified_single_model_epoch30test_20260820.py)에서
이미 각 시드가 patience=8 기준 9~10에폭 내에 하나의 고정 방향편향으로 수렴하고 그 뒤로 전혀
안 흔들린다는 걸 확인했다 -- ASWA가 다룰 "궤적 내 진동" 자체가 없다. 반면 우리 문제(시드마다
다른 편향에 수렴)에 대응하는 건 여러 독립 시드의 예측을 결합하는 딥앙상블/bagging 쪽이라,
이쪽을 구현한다.

재학습 없음 -- 기존 5개 시드(2026-08-19, epoch=2 스크리닝. epoch가 결과에 영향 없음은 이미
검증됨)의 oos_predictions_q045.csv에 있는 direction head 확률(dir_p_cash/long/short)을
bar별로 단순평균한 뒤 argmax -- 표준 딥앙상블(Lakshminarayanan et al. 스타일 predictive
distribution averaging) 방식이다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402 (build_dec, omega 재사용)

omega = base_nt.omega

SEEDS = [758616172, 810628369, 615897020, 176529615, 573123622]
P_COLS = [
    "omega1_regime3_expertdq_dir_p_cash",
    "omega1_regime3_expertdq_dir_p_long",
    "omega1_regime3_expertdq_dir_p_short",
]


def _seed_dir(seed: int) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_dc_dense_cashfill_unified_single_model_seed{seed}_20260819"


def main() -> None:
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(_seed_dir(seed) / "oos_predictions_q045.csv", parse_dates=["timestamp"], usecols=["timestamp", *P_COLS])
        df = df.rename(columns={c: f"{c}__{seed}" for c in P_COLS}).set_index("timestamp")
        frames.append(df)
    merged = pd.concat(frames, axis=1, join="inner")
    if len(merged) != len(frames[0]):
        raise RuntimeError(f"5시드 timestamp 정렬 불일치 -- inner join 후 {len(merged)}행, 개별 {len(frames[0])}행")
    print(f"5시드 OOS 확률 정렬 완료: {len(merged):,}행 (개별 시드와 동일 -- timestamp 완전일치 확인)", flush=True)

    avg = pd.DataFrame(index=merged.index)
    for c in P_COLS:
        avg[c] = merged[[f"{c}__{s}" for s in SEEDS]].mean(axis=1)
    ensemble_action = avg[P_COLS].to_numpy().argmax(axis=1)  # 0=cash,1=long,2=short (컬럼 순서와 일치)

    n_long = int((ensemble_action == 1).sum())
    n_short = int((ensemble_action == 2).sum())
    n_cash = int((ensemble_action == 0).sum())
    print(f"앙상블 argmax 분포: CASH={n_cash} LONG={n_long} SHORT={n_short} "
          f"(활성bar 중 LONG%={n_long/max(n_long+n_short,1)*100:.1f}%, "
          f"개별시드 범위는 raw신호기준 0~88% -- 예측대로 base rate 근처로 수렴하는지 확인용)", flush=True)

    # --- 조건부 방향정확도 (개별 시드와 동일 지표, 직접비교 가능) ---
    true_lbl = pd.read_csv(
        ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819/zigzag_action_labels_2026.csv",
        usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"],
    ).set_index("timestamp")
    joined = true_lbl.reindex(merged.index)
    y_true = joined["zigzag_action"].to_numpy()
    y_pred = ensemble_action
    both_active = (y_true != 0) & (y_pred != 0)
    n_both = int(both_active.sum())
    dir_match = float((y_true[both_active] == y_pred[both_active]).mean() * 100) if n_both else float("nan")
    print(f"\n조건부 방향정확도(앙상블): 교집합 n={n_both} 방향일치율={dir_match:.1f}% "
          f"(개별 5시드 범위: 48.2~51.4%, chance 기준선 ~50~52%)", flush=True)

    # --- 다수결(majority vote) 앙상블도 교차확인 ---
    indiv_actions = []
    for seed in SEEDS:
        df = pd.read_csv(_seed_dir(seed) / "oos_predictions_q045.csv", parse_dates=["timestamp"],
                          usecols=["timestamp", "omega1_regime3_expertdq_dir_action"]).set_index("timestamp")
        indiv_actions.append(df.reindex(merged.index)["omega1_regime3_expertdq_dir_action"].to_numpy())
    indiv_stack = np.stack(indiv_actions, axis=1)  # (n, 5)
    majority = np.array([np.bincount(row, minlength=3).argmax() for row in indiv_stack])
    maj_both_active = (y_true != 0) & (majority != 0)
    maj_match = float((y_true[maj_both_active] == majority[maj_both_active]).mean() * 100) if maj_both_active.sum() else float("nan")
    print(f"다수결 앙상블(교차확인): 교집합 n={int(maj_both_active.sum())} 방향일치율={maj_match:.1f}%", flush=True)

    # --- PnL: 앙상블 action을 그대로 트레이딩 신호로 사용 (N-HiTS의 build_dec와 동일 패턴) ---
    ohlc = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                        usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).set_index("timestamp")
    ohlc = ohlc.reindex(merged.index)
    if ohlc["close"].isna().any():
        raise RuntimeError("OHLC 정렬 후 결측 -- timestamp 커버리지 불일치")
    ohlc_reset = ohlc.reset_index()
    dec = base_nt.build_dec(ensemble_action)
    m = omega._metrics(ohlc_reset, dec, fee=base_nt._FEE, slip=base_nt._SLIP, cost_mult=base_nt.COST_MULTS["cost3"])
    print(f"\n앙상블 PnL(cost3, BASE_TEMPLATE 사이징): pnl={m['pnl']:+.2f} mdd={m['mdd']:+.2f} trades={m['trades']} "
          f"wr={m['wr']:.3f} L/S={m['long_entries']}/{m['short_entries']} exit={m['exit_reasons']}", flush=True)
    print(f"참고 -- 개별 5시드 OOS PnL(각자 원래 threshold, 사이징 다름이라 직접비교는 참고용): "
          f"+60.70, +23.64, +18.07, -3.88, -9.33", flush=True)

    out = {
        "seeds_ensembled": SEEDS, "method": "deep_ensemble_probability_averaging_argmax",
        "n_bars": int(len(merged)), "ensemble_long_pct_raw": n_long / max(n_long + n_short, 1) * 100,
        "conditional_direction_accuracy_pct": dir_match, "majority_vote_accuracy_pct": maj_match,
        "pnl_metrics": m,
    }
    out_path = ROOT / "tmp/eth_directional_change_tabm_deep_ensemble_verification_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}", flush=True)


if __name__ == "__main__":
    main()
