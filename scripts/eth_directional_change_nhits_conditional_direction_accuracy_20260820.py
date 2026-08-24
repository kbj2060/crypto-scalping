#!/usr/bin/env python3
"""N-HiTS 단일시드(502957522, 재현 확인됨)의 조건부 방향정확도 계산.

TabM 5시드에서 이미 확인한 것과 동일한 질문 -- "실제 라벨과 예측이 둘 다 CASH가 아닌 bar들만
놓고 보면 방향(LONG/SHORT) 일치율이 chance(라벨 자체의 LONG/SHORT 비율)보다 높은가?" --을
N-HiTS에도 대칭적으로 적용한다. 기존 분석 스크립트(..._breakdown_20260820.py)는 예측 배열을
CSV로 저장하지 않아 재계산이 필요 -- 학습 자체는 완전 재현 확인됨(동일 seed, 동일 VAL/OOS
pnl/trades/wr)이라 이번에도 동일 모델이 나온다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819  # noqa: F401,E402
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402

omega = base_nt.omega
SEED = 502957522


def main() -> None:
    device = base_nt._device("cpu")
    data = base_nt.load_panel_and_labels()
    window = int(base_nt.ARCH_DEFAULT_TRAIN.get("window", base_nt.DEFAULT_WINDOW))
    oos_mask = (data["panel"]["timestamp"] >= base_nt.OOS_START) & (data["panel"]["timestamp"] <= base_nt.OOS_END)
    oos_idx = base_nt._valid_indices(oos_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])

    r = base_nt._fit_one(
        "nhits", base_nt.ARCH_DEFAULT_PARAMS["nhits"], base_nt.ARCH_DEFAULT_TRAIN, seed=SEED,
        epochs=base_nt.MAX_EPOCHS_FINAL, patience=base_nt.PATIENCE_FINAL,
        use_gce=False, use_elr=False, use_mixup=False, data=data, device=device,
    )
    preds = base_nt._predict(r["model"], r["scaler_raw_std"], r["window"], oos_idx, data["y_dir_full"], data["y_qual_full"], device)
    y_pred = preds["direction"].astype(np.int64)
    y_true = data["y_dir_full"][oos_idx].astype(np.int64)

    n_true_active = int((y_true != 0).sum())
    true_long_pct = float((y_true[y_true != 0] == omega.ACTION_LONG).mean() * 100) if n_true_active else float("nan")
    print(f"실제 라벨(OOS 구간, N-HiTS valid_indices 기준) 이벤트bar={n_true_active} "
          f"LONG비율={true_long_pct:.1f}% SHORT비율={100-true_long_pct:.1f}% (chance 기준선)")

    both_active = (y_true != 0) & (y_pred != 0)
    n_both = int(both_active.sum())
    dir_match = float((y_true[both_active] == y_pred[both_active]).mean() * 100) if n_both else float("nan")
    recall_active = float((y_pred[y_true != 0] != 0).mean() * 100) if n_true_active else float("nan")
    print(f"실제+예측 둘다 active인 교집합 n={n_both} 방향일치율={dir_match:.1f}% "
          f"(chance~{true_long_pct if true_long_pct>50 else 100-true_long_pct:.0f}%)  |  "
          f"실제이벤트 중 CASH아닌예측 낸 비율(recall)={recall_active:.1f}%")


if __name__ == "__main__":
    main()
