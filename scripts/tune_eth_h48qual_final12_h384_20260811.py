"""ETH h48qual(final12 피쳐 + h384 quality horizon) 파라미터 튜닝, v2.

v1(epochs=3|4|6 스윕)은 사후분석에서 버그가 드러났다: patience=8은 8에폭 연속 무개선이어야 발동하는데
epoch cap이 3~6이라 발동할 공간이 아예 없었다 -- 세 epoch 설정이 seed당 완전히 동일한 결과(같은
best_validation_loss, 소수점까지)를 냈다. epochs=30/시드2개 확인런에서 patience가 실제로 epoch
9~10에서 발동하는 걸 확인했다. 그래서 v2는 epochs를 더 이상 스윕하지 않는다 -- 40으로 넉넉히 고정하고
patience가 실제 정지 지점을 결정하게 둔다. 대신 그 확인런에서 seed=260620/481003 사이 VAL pnl이
(+13~14 vs 전구간 마이너스)로 크게 갈린 게 드러나서, 절약된 예산(epoch축 3칸->1칸)을 seed 5->15로
돌린다 -- 지금 진짜 열린 질문은 "epoch을 얼마나 돌리나"가 아니라 "이 신호가 seed 노이즈보다 큰가"이기
때문이다. 총 런 수는 v1과 동일하게 30(=1 epoch x 2 train_rows x 15 seeds)으로 유지.

Swept:
    seed          기존 5개(260620 base 기본시드 포함, 무작위추출, 고정간격 아님) + 신규 무작위 10개,
                  총 15개 -- seed-diversity 게이트가 요구하는 N>=5을 훨씬 상회.
    max_train_rows 30000 | 45000
    epochs        40으로 고정(스윕 아님) -- patience=8이 실제 정지 지점을 정함(확인런에서 9~10에폭
                  관찰, 40은 그보다 넉넉한 상한).

Selection, VAL only (OOS는 여기서 선택 기준으로 안 쓴다):
    (train_rows, quality_threshold)별로 VAL pnl/mdd를 15시드 평균 낸 뒤, 평균 VAL mdd >= -8.0을
    만족하는 것 중 평균 VAL pnl이 최고인 조합을 고른다. v1에서는 0/30이 이 floor를 통과 못 했다 --
    이번에도 0개면 그 사실을 그대로 보여주고 전체에서 pnl 최고를 고른다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(Path.home() / "anaconda3/envs/quant_ai/bin/python")
OUT_DIR = ROOT / "data/ensemble/reports/eth_h48qual_final12_h384_20260811"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
SCRIPT = ROOT / "scripts/train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py"

SEEDS = (260620, 481003, 26611, 903174, 155827,  # tune_btc_parent_regime_jmredesign_20260810.py와 동일
         44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662)  # 신규 무작위 10개
EPOCHS = (40,)
TRAIN_ROWS = (30000, 45000)
VAL_MDD_FLOOR = -8.0


def suffix(epochs: int, rows: int, seed: int) -> str:
    return f"h48qual_final12_h384_20260811_v2_e{epochs}_r{rows}_s{seed}"


def run_one(epochs: int, rows: int, seed: int, python: str) -> Path | None:
    cmd = [python, str(SCRIPT), "--epochs", str(epochs), "--max-train-rows", str(rows),
           "--seed", str(seed), "--device", "cuda", "--out-suffix", suffix(epochs, rows, seed)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED e{epochs} r{rows} s{seed}: {r.stderr.strip().splitlines()[-1][:160]}")
        return None
    hits = sorted(RUN_ROOT.glob(f"*{suffix(epochs, rows, seed)}"))
    print(f"  ok e{epochs} r{rows} s{seed}  ({time.time() - t0:.0f}s) -> {hits[-1].name if hits else '?'}")
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=PYTHON_BIN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    combos = list(product(EPOCHS, TRAIN_ROWS, SEEDS))
    print(f"=== ETH h48qual final12+h384 tuning: {len(combos)} runs "
          f"({len(EPOCHS)} epochs x {len(TRAIN_ROWS)} train_rows x {len(SEEDS)} seeds)")
    if args.dry_run:
        for e, r, s in combos:
            print("  ", suffix(e, r, s))
        return

    rows = []
    t0 = time.time()
    for i, (e, r, s) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] epochs={e} train_rows={r} seed={s}", flush=True)
        d = run_one(e, r, s, args.python)
        if d is None:
            continue
        rank = pd.read_csv(d / "quality_threshold_ranking.csv")
        for _, row in rank.iterrows():
            rows.append({"epochs": e, "train_rows": r, "seed": s, "dir": d.name,
                         "quality_threshold": float(row["quality_threshold"]),
                         "val_pnl": float(row["validation_pnl"]),
                         "val_mdd": float(row["validation_mdd"]),
                         "val_trades": int(row["validation_trades"]),
                         "oos_pnl": float(row["oos_pnl"]),
                         "oos_mdd": float(row["oos_mdd"])})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "tune_eth_h48qual_v2_runs.csv", index=False)

    agg = (df.groupby(["epochs", "train_rows", "quality_threshold"])
             .agg(val_pnl_mean=("val_pnl", "mean"), val_pnl_std=("val_pnl", "std"),
                  val_pnl_min=("val_pnl", "min"), val_mdd_mean=("val_mdd", "mean"),
                  val_mdd_worst=("val_mdd", "min"), val_trades_mean=("val_trades", "mean"),
                  oos_pnl_mean=("oos_pnl", "mean"), n=("seed", "count"))
             .reset_index())
    agg.to_csv(OUT_DIR / "tune_eth_h48qual_v2_aggregate.csv", index=False)

    eligible = agg[agg["val_mdd_mean"] >= VAL_MDD_FLOOR]
    print(f"\n=== {len(eligible)}/{len(agg)} configs meet mean VAL MDD >= {VAL_MDD_FLOOR}")
    show = (eligible if len(eligible) else agg).sort_values("val_pnl_mean", ascending=False)
    print(show.head(12).to_string(index=False,
          columns=["epochs", "train_rows", "quality_threshold", "val_pnl_mean", "val_pnl_std",
                   "val_mdd_mean", "val_mdd_worst", "val_trades_mean", "oos_pnl_mean", "n"],
          float_format=lambda v: f"{v:8.3f}"))
    if len(eligible):
        best = show.iloc[0]
        print(f"\nSELECTED (VAL only): epochs={int(best['epochs'])} "
              f"train_rows={int(best['train_rows'])} q={best['quality_threshold']:.2f}  "
              f"VAL pnl {best['val_pnl_mean']:.2f}+-{best['val_pnl_std']:.2f} "
              f"mdd {best['val_mdd_mean']:.2f}")
        (OUT_DIR / "tune_eth_h48qual_v2_selected.json").write_text(json.dumps({
            "epochs": int(best["epochs"]), "train_rows": int(best["train_rows"]),
            "quality_threshold": float(best["quality_threshold"]),
            "val_pnl_mean": float(best["val_pnl_mean"]),
            "val_mdd_mean": float(best["val_mdd_mean"]),
            "seeds": list(SEEDS), "selection": "mean VAL pnl s.t. mean VAL mdd >= -8, seeds averaged",
        }, indent=2))
    else:
        print("\nNO config meets the VAL MDD floor -- floor may not transfer from BTC's scale to "
              "this ETH config; falling back to best pnl from the full (unfiltered) grid above.")
    print(f"\ntotal {time.time() - t0:.0f}s  -> {OUT_DIR}/tune_eth_h48qual_v2_*.csv")


if __name__ == "__main__":
    main()
