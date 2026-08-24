#!/usr/bin/env python3
"""TabM DC 딥앙상블 -- 두 번째 독립 5시드 배치로 앙상블 자체의 시드배치간 안정성 검증.

`eth_directional_change_tabm_deep_ensemble_verification_20260820.py`(1차 앙상블, seeds=
[758616172,810628369,615897020,176529615,573123622] -> LONG44.0%/조건부정확도50.7%/PnL+8.30)
와 완전히 동일한 방법(direction 확률 평균+argmax)을 **처음부터 새로 뽑은 5개 시드**에 적용한다
-- "앙상블을 다시 만들어도 비슷한 자리로 수렴하는가"를 실제로 확인하기 위해, 1차 배치와
겹치지 않는 5개 시드를 오늘 새로 학습시켰다(같은 unified_single_model 스크립트, epoch=2 --
epoch가 결과에 영향 없음은 이미 검증됨)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402

omega = base_nt.omega

SEEDS = [498893814, 405866927, 492015211, 108277116, 519733484]  # 1차 배치와 완전히 별개(신규 무작위)
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
    print(f"5시드(배치2) OOS 확률 정렬 완료: {len(merged):,}행", flush=True)

    avg = pd.DataFrame(index=merged.index)
    for c in P_COLS:
        avg[c] = merged[[f"{c}__{s}" for s in SEEDS]].mean(axis=1)
    ensemble_action = avg[P_COLS].to_numpy().argmax(axis=1)

    n_long = int((ensemble_action == 1).sum())
    n_short = int((ensemble_action == 2).sum())
    print(f"앙상블(배치2) argmax 분포: LONG={n_long} SHORT={n_short} "
          f"(활성bar 중 LONG%={n_long/max(n_long+n_short,1)*100:.1f}%, 1차 배치는 44.0%)", flush=True)

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
    print(f"조건부 방향정확도(앙상블 배치2): 교집합 n={n_both} 방향일치율={dir_match:.1f}% (1차 배치: 50.7%)", flush=True)

    ohlc = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                        usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).set_index("timestamp")
    ohlc = ohlc.reindex(merged.index)
    if ohlc["close"].isna().any():
        raise RuntimeError("OHLC 정렬 후 결측")
    dec = base_nt.build_dec(ensemble_action)
    m = omega._metrics(ohlc.reset_index(), dec, fee=base_nt._FEE, slip=base_nt._SLIP, cost_mult=base_nt.COST_MULTS["cost3"])
    print(f"앙상블(배치2) PnL(cost3): pnl={m['pnl']:+.2f} mdd={m['mdd']:+.2f} trades={m['trades']} "
          f"wr={m['wr']:.3f} L/S={m['long_entries']}/{m['short_entries']} exit={m['exit_reasons']} (1차 배치: pnl=+8.30 mdd=-17.33)", flush=True)

    out = {"seeds_ensembled": SEEDS, "method": "deep_ensemble_probability_averaging_argmax_batch2",
           "n_bars": int(len(merged)), "ensemble_long_pct_raw": n_long / max(n_long + n_short, 1) * 100,
           "conditional_direction_accuracy_pct": dir_match, "pnl_metrics": m}
    out_path = ROOT / "tmp/eth_directional_change_tabm_deep_ensemble_verification_batch2_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[report] {out_path}", flush=True)


if __name__ == "__main__":
    main()
