#!/usr/bin/env python3
"""CUSUM aswa/bag(+hp 재확인용) 5시드 분석 -- eth_cusum_hp_variant_analysis_20260820.py의
버그수정(dense-cashfill 기반 앙상블 조건부정확도)을 그대로 물려받아 변형별로 반복.
DC variant_analysis(hp/aswa/bag 전부)와 동일 방법론, CUSUM 라벨/경로로 교체."""
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

SEEDS_BY_VARIANT = {
    "hp": [275556965, 560024039, 179071808, 605770441, 214282759],  # 재확인용(이미 분석완료, 비교표용 재사용)
    "aswa": [976276698, 378385158, 833303863, 320746188, 5770026],
    "bag": [724851692, 398400266, 7309143, 253857633, 748580650],
}

TRUE_LBL = pd.read_csv(
    ROOT / "tmp/eth_cusum_triple_barrier_labels_20260819/zigzag_action_labels_2026.csv",
    usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"],
).set_index("timestamp")
TRUE_LBL_DENSE = pd.read_csv(
    ROOT / "tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820/zigzag_action_labels_2026.csv",
    usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"],
).set_index("timestamp")


def _seed_dir(variant: str, seed: int) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_cusum_dense_cashfill_unified_single_model_{variant}_seed{seed}_20260820"


def _conditional_accuracy(variant: str, seed: int, variant_thr: str = "q045") -> dict:
    d = _seed_dir(variant, seed)
    pred = pd.read_csv(d / f"oos_predictions_{variant_thr}.csv", parse_dates=["timestamp"],
                        usecols=["timestamp", "omega1_regime3_expertdq_dir_action"])
    joined = pred.set_index("timestamp").join(TRUE_LBL, how="inner")
    y_true = joined["zigzag_action"].to_numpy()
    y_pred = joined["omega1_regime3_expertdq_dir_action"].to_numpy()
    both = (y_true != 0) & (y_pred != 0)
    acc = float((y_true[both] == y_pred[both]).mean() * 100) if both.sum() else float("nan")
    n_long = int((y_pred == 1).sum()); n_short = int((y_pred == 2).sum())
    return {"n_both": int(both.sum()), "cond_acc": acc, "raw_long_pct": n_long / max(n_long + n_short, 1) * 100}


def _report_json_pnl(variant: str, seed: int) -> dict:
    r = json.load(open(_seed_dir(variant, seed) / "report.json"))
    best = r["ranking_by_validation_pnl"][0]
    return {"threshold": best["quality_threshold"], "val_pnl": best["validation_pnl"], "oos_pnl": best["oos_pnl"],
            "epochs_ran": r["summaries"]["bull"]["epochs_ran"]}


def _deep_ensemble(variant: str, seeds: list[int]) -> dict:
    p_cols = ["omega1_regime3_expertdq_dir_p_cash", "omega1_regime3_expertdq_dir_p_long", "omega1_regime3_expertdq_dir_p_short"]
    frames = []
    for seed in seeds:
        df = pd.read_csv(_seed_dir(variant, seed) / "oos_predictions_q045.csv", parse_dates=["timestamp"], usecols=["timestamp", *p_cols])
        df = df.rename(columns={c: f"{c}__{seed}" for c in p_cols}).set_index("timestamp")
        frames.append(df)
    merged = pd.concat(frames, axis=1, join="inner")
    avg = pd.DataFrame(index=merged.index)
    for c in p_cols:
        avg[c] = merged[[f"{c}__{s}" for s in seeds]].mean(axis=1)
    ensemble_action = avg[p_cols].to_numpy().argmax(axis=1)

    joined = TRUE_LBL_DENSE.reindex(merged.index)
    if joined["zigzag_action"].isna().any():
        raise RuntimeError(f"{variant}: dense-cashfill 재색인 후 NaN {int(joined['zigzag_action'].isna().sum())}개")
    y_true = joined["zigzag_action"].to_numpy()
    both = (y_true != 0) & (ensemble_action != 0)
    cond_acc = float((y_true[both] == ensemble_action[both]).mean() * 100) if both.sum() else float("nan")
    n_long = int((ensemble_action == 1).sum()); n_short = int((ensemble_action == 2).sum())

    ohlc = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                        usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).set_index("timestamp")
    ohlc = ohlc.reindex(merged.index)
    dec = base_nt.build_dec(ensemble_action)
    m = omega._metrics(ohlc.reset_index(), dec, fee=base_nt._FEE, slip=base_nt._SLIP, cost_mult=base_nt.COST_MULTS["cost3"])
    return {"cond_acc": cond_acc, "long_pct": n_long / max(n_long + n_short, 1) * 100, "pnl": m["pnl"], "mdd": m["mdd"],
            "trades": m["trades"], "wr": m["wr"]}


def main() -> None:
    report = {}
    for variant, seeds in SEEDS_BY_VARIANT.items():
        print(f"\n{'='*20} CUSUM variant={variant} {'='*20}", flush=True)
        rows = []
        for seed in seeds:
            acc = _conditional_accuracy(variant, seed)
            pnl = _report_json_pnl(variant, seed)
            rows.append({"seed": seed, **acc, **pnl})
            print(f"  seed={seed}: epochs_ran={pnl['epochs_ran']} threshold={pnl['threshold']} "
                  f"OOS_pnl={pnl['oos_pnl']:+.2f} cond_acc={acc['cond_acc']:.1f}% raw_long%={acc['raw_long_pct']:.1f}%", flush=True)
        accs = [r["cond_acc"] for r in rows]
        pnls = [r["oos_pnl"] for r in rows]
        print(f"  [{variant} 개별5시드] cond_acc: {min(accs):.1f}~{max(accs):.1f}%  "
              f"OOS_pnl: {min(pnls):+.2f}~{max(pnls):+.2f} "
              f"부호일치={'YES' if all(p>0 for p in pnls) or all(p<0 for p in pnls) else 'NO(혼재)'}", flush=True)

        ens = _deep_ensemble(variant, seeds)
        print(f"  [{variant} 5시드 딥앙상블] LONG%={ens['long_pct']:.1f}% cond_acc={ens['cond_acc']:.1f}% "
              f"pnl={ens['pnl']:+.2f} mdd={ens['mdd']:+.2f} trades={ens['trades']} wr={ens['wr']:.3f}", flush=True)

        report[variant] = {"individual": rows, "individual_cond_acc_range": [min(accs), max(accs)],
                            "individual_oos_pnl_range": [min(pnls), max(pnls)], "ensemble": ens}

    out_path = ROOT / "tmp/eth_cusum_variant_analysis_20260820.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
