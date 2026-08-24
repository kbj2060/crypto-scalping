#!/usr/bin/env python3
"""N-HiTS 단일시드 본실행 -- 실제 VAL/OOS PnL 확인(사용자 승인: N-HiTS 단일시드 사전확인부터).

`train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819.py`의 배선(레짐피쳐 concat +
DC dense-cashfill 라벨 오버라이드, import 시점에 이미 적용됨)을 그대로 재사용하고,
`base_nt.stage_final()`의 로직을 N=1 시드로 복제한다(재구현 아님 -- N_SEEDS_FINAL=5 루프를
시드 1개로 줄인 것뿐, HP서치/isolation 없이 ARCH_DEFAULT_PARAMS/ARCH_DEFAULT_TRAIN 기본값과
GCE/ELR/mixup 전부 off를 씀 -- `base_nt.stage_sanity`와 같은 "튜닝 없는 사전확인" 철학).

sanity 스테이지(이미 통과 확인 -- ModernTCN 450초/N-HiTS 5초, 크래시 없음)는 크래시 여부만
봤을 뿐 PnL을 안 냈다. 이 스크립트는 N-HiTS만(ModernTCN은 CPU에서 sanity 수준 소표본도 450초라
전체데이터+MAX_EPOCHS_FINAL=30은 비현실적, 서버 GPU 필요 -- 사용자가 이미 이 축을 보류하기로
결정) 실제 VAL/OOS 기간에 PnL을 계산해 "쓸만한 신호가 있는지" 첫 데이터 포인트를 얻는다.
N=1이므로 결론 근거가 아니라 다음 단계(추가 시드 투자 여부) 판단용 스크리닝이다."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_regimefeature_dc_20260819  # noqa: F401,E402 (와이어링 부작용)
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402


def main() -> None:
    device = base_nt._device("cpu")
    data = base_nt.load_panel_and_labels()

    arch = "nhits"
    seed = random.SystemRandom().randint(1_000_000, 999_000_000)
    print(f"stage=start arch={arch} seed={seed} (단일시드, ARCH_DEFAULT_PARAMS/TRAIN 기본값, "
          f"GCE/ELR/mixup 전부 off)", flush=True)

    window = int(base_nt.ARCH_DEFAULT_TRAIN.get("window", base_nt.DEFAULT_WINDOW))
    val_mask = (data["panel"]["timestamp"] >= base_nt.VAL_START) & (data["panel"]["timestamp"] <= base_nt.VAL_END)
    oos_mask = (data["panel"]["timestamp"] >= base_nt.OOS_START) & (data["panel"]["timestamp"] <= base_nt.OOS_END)
    val_idx = base_nt._valid_indices(val_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])
    oos_idx = base_nt._valid_indices(oos_mask.to_numpy(), window, data["y_dir_full"], data["y_qual_full"])
    print(f"VAL n={len(val_idx)} ({base_nt.VAL_START.date()}~{base_nt.VAL_END.date()})  "
          f"OOS n={len(oos_idx)} ({base_nt.OOS_START.date()}~{base_nt.OOS_END.date()})", flush=True)

    t0 = time.time()
    r = base_nt._fit_one(
        arch, base_nt.ARCH_DEFAULT_PARAMS[arch], base_nt.ARCH_DEFAULT_TRAIN, seed=seed,
        epochs=base_nt.MAX_EPOCHS_FINAL, patience=base_nt.PATIENCE_FINAL,
        use_gce=False, use_elr=False, use_mixup=False, data=data, device=device,
    )
    print(f"학습 완료 ({time.time()-t0:.0f}s) epochs_ran={r['epochs_ran']} es_loss={r['es_loss']:.4f}", flush=True)

    result = {"arch": arch, "seed": seed, "epochs_ran": r["epochs_ran"], "es_loss": r["es_loss"],
              "n_features": len(base_nt.SEQ_COLS), "feature_cols": list(base_nt.SEQ_COLS)}
    for split_name, idx in (("VAL", val_idx), ("OOS", oos_idx)):
        preds = base_nt._predict(r["model"], r["scaler_raw_std"], r["window"], idx, data["y_dir_full"], data["y_qual_full"], device)
        cls = base_nt.classification_report(idx, preds, data["y_dir_full"], data["y_qual_full"])
        pnl = base_nt.pnl_vs_benchmarks(data["panel"], idx, preds["direction"])
        result[f"{split_name}_classification"] = cls
        result[f"{split_name}_pnl"] = pnl
        c3 = pnl["cost3"]
        print(f"{split_name}: n={len(idx)} dir_bacc={cls['direction_balanced_accuracy']:.4f} "
              f"model_pnl(cost3)={c3['model_pnl']:.2f} trades={c3['model_trades']} "
              f"always_short={c3['always_short_pnl']:.2f} always_long={c3['always_long_pnl']:.2f} "
              f"beats_short={c3['beats_always_short']} beats_long={c3['beats_always_long']}", flush=True)

    out_path = base_nt.OUT_DIR / f"dc_regimefeature_single_seed_{arch}_seed{seed}_20260819.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[report] {out_path}", flush=True)


if __name__ == "__main__":
    main()
