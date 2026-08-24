#!/usr/bin/env python3
"""N-HiTS 단일시드(502957522) 재실행 -- 마이너스 구간 특징 분석용 상세 브레이크다운.

`eth_directional_change_nhits_single_seed_run_20260819.py`와 완전히 동일한 배선/아키텍처/
하이퍼파라미터를 쓰되, 이번엔 seed를 그날 실제로 뽑힌 값(502957522)으로 고정해 동일 모델을
재현한다(`_seed_everything()`이 random/numpy/torch를 전부 시드 고정, CPU라 non-determinism
소스 없음 -- 재현성 확인됨). 원본 스크립트는 pnl_vs_benchmarks()의 집계 dict(pnl/trades/wr만)만
저장하고 raw per-bar 예측을 버렸는데, TabM 쪽 시드별 분석(exit_reasons/long_entries/
short_entries/mdd 등 omega._metrics()의 전체 반환값)과 대칭적으로 비교하려면 이 상세 필드가
필요해서 다시 뽑는다. N-HiTS는 이 저장소에 N=1 시드뿐이라 "시드 간 비교"는 애초에 불가능하다
-- 이 스크립트는 그 유일한 런 내부의 방향편향/청산사유/월별 분포만 들여다본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819  # noqa: F401,E402 (와이어링 부작용)
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402

omega = base_nt.omega

SEED = 502957522  # 20260819 단일시드런에서 실제로 뽑혔던 값 -- 동일 모델 재현 목적


def main() -> None:
    device = base_nt._device("cpu")
    data = base_nt.load_panel_and_labels()

    arch = "nhits"
    print(f"stage=start arch={arch} seed={SEED} (20260819 런 재현, 상세 브레이크다운용)", flush=True)

    window = int(base_nt.ARCH_DEFAULT_TRAIN.get("window", base_nt.DEFAULT_WINDOW))
    val_mask = (data["panel"]["timestamp"] >= base_nt.VAL_START) & (data["panel"]["timestamp"] <= base_nt.VAL_END)
    oos_mask = (data["panel"]["timestamp"] >= base_nt.OOS_START) & (data["panel"]["timestamp"] <= base_nt.OOS_END)
    val_idx = base_nt._valid_indices(val_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])
    oos_idx = base_nt._valid_indices(oos_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])

    r = base_nt._fit_one(
        arch, base_nt.ARCH_DEFAULT_PARAMS[arch], base_nt.ARCH_DEFAULT_TRAIN, seed=SEED,
        epochs=base_nt.MAX_EPOCHS_FINAL, patience=base_nt.PATIENCE_FINAL,
        use_gce=False, use_elr=False, use_mixup=False, data=data, device=device,
    )
    print(f"학습 완료 epochs_ran={r['epochs_ran']} es_loss={r['es_loss']:.4f} "
          f"(20260819 런과 동일해야 재현 성공: epochs_ran=30 es_loss 근사)", flush=True)

    result: dict = {"arch": arch, "seed": SEED}
    for split_name, idx in (("VAL", val_idx), ("OOS", oos_idx)):
        preds = base_nt._predict(r["model"], r["scaler_raw_std"], r["window"], idx, data["y_dir_full"], data["y_qual_full"], device)
        direction_pred = preds["direction"]

        ohlc = data["panel"].iloc[idx][["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec = base_nt.build_dec(direction_pred)
        m = omega._metrics(ohlc, dec, fee=base_nt._FEE, slip=base_nt._SLIP, cost_mult=base_nt.COST_MULTS["cost3"])

        ts = pd.to_datetime(data["panel"]["timestamp"].to_numpy()[idx])
        action_names = np.where(direction_pred == omega.ACTION_LONG, "LONG",
                                 np.where(direction_pred == omega.ACTION_SHORT, "SHORT", "CASH"))
        month_tab = (
            pd.DataFrame({"month": ts.to_period("M"), "action": action_names})
            .query("action != 'CASH'")
            .groupby(["month", "action"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        for c in ("LONG", "SHORT"):
            if c not in month_tab.columns:
                month_tab[c] = 0
        month_tab["long_pct"] = (month_tab["LONG"] / (month_tab["LONG"] + month_tab["SHORT"]).clip(lower=1) * 100).round(1)

        n_long_signal = int((direction_pred == omega.ACTION_LONG).sum())
        n_short_signal = int((direction_pred == omega.ACTION_SHORT).sum())
        n_signal = max(n_long_signal + n_short_signal, 1)

        result[split_name] = {
            "n_bars": int(len(idx)),
            "raw_signal_long": n_long_signal, "raw_signal_short": n_short_signal,
            "raw_signal_long_pct": round(n_long_signal / n_signal * 100, 1),
            "metrics": m,
            "month_breakdown": {str(k): v.to_dict() for k, v in month_tab.iterrows()},
        }
        print(f"\n=== {split_name} n={len(idx)} raw_signal LONG={n_long_signal}({n_long_signal/n_signal*100:.0f}%) "
              f"SHORT={n_short_signal}({n_short_signal/n_signal*100:.0f}%) ===", flush=True)
        print(f"  metrics: pnl={m['pnl']:+.2f} mdd={m['mdd']:+.2f} trades={m['trades']} wr={m['wr']:.3f} "
              f"L/S={m['long_entries']}/{m['short_entries']} exit={m['exit_reasons']} tpd={m['trades_per_day']:.3f}", flush=True)
        print(month_tab, flush=True)

    out_path = base_nt.OUT_DIR / f"dc_regimefeature_single_seed_{arch}_seed{SEED}_breakdown_20260820.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}", flush=True)


if __name__ == "__main__":
    main()
