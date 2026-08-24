#!/usr/bin/env python3
"""레짐피쳐 제거+2024-2025 확장 축 5시드 분석 -- 개별 조건부 방향정확도 + 5시드 딥앙상블,
DC 원본(158피쳐/2025단독학습) 베이스라인과 비교. OOS 행수가 다름(57,601 vs 원본 51,746 --
이번 축은 REGIME3_CURRENT_2026 필터링이 필요없어져 전체 2026 파일을 그대로 씀)에 유의."""
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

SEEDS = [210016155, 310845945, 997351716, 35952628, 585635406]
TRUE_LBL = pd.read_csv(
    ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819/zigzag_action_labels_2026.csv",
    usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"],
).set_index("timestamp")


def _seed_dir(seed: int) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_dc_2024_2025_noregime_unified_single_model_seed{seed}_20260820"


def _conditional_accuracy(seed: int, variant_thr: str = "q045") -> dict:
    d = _seed_dir(seed)
    pred = pd.read_csv(d / f"oos_predictions_{variant_thr}.csv", parse_dates=["timestamp"],
                        usecols=["timestamp", "omega1_regime3_expertdq_dir_action"])
    joined = pred.set_index("timestamp").join(TRUE_LBL, how="inner")
    y_true = joined["zigzag_action"].to_numpy()
    y_pred = joined["omega1_regime3_expertdq_dir_action"].to_numpy()
    both = (y_true != 0) & (y_pred != 0)
    acc = float((y_true[both] == y_pred[both]).mean() * 100) if both.sum() else float("nan")
    n_long = int((y_pred == 1).sum()); n_short = int((y_pred == 2).sum())
    return {"n_both": int(both.sum()), "cond_acc": acc, "raw_long_pct": n_long / max(n_long + n_short, 1) * 100}


def _report_json_pnl(seed: int) -> dict:
    r = json.load(open(_seed_dir(seed) / "report.json"))
    best = r["ranking_by_validation_pnl"][0]
    return {"threshold": best["quality_threshold"], "val_pnl": best["validation_pnl"], "oos_pnl": best["oos_pnl"],
            "epochs_ran": r["summaries"]["bull"]["epochs_ran"], "base_feature_count": r["input_contract"]["base_feature_count"]}


def _deep_ensemble(seeds: list[int]) -> dict:
    p_cols = ["omega1_regime3_expertdq_dir_p_cash", "omega1_regime3_expertdq_dir_p_long", "omega1_regime3_expertdq_dir_p_short"]
    frames = []
    for seed in seeds:
        df = pd.read_csv(_seed_dir(seed) / "oos_predictions_q045.csv", parse_dates=["timestamp"], usecols=["timestamp", *p_cols])
        df = df.rename(columns={c: f"{c}__{seed}" for c in p_cols}).set_index("timestamp")
        frames.append(df)
    merged = pd.concat(frames, axis=1, join="inner")
    avg = pd.DataFrame(index=merged.index)
    for c in p_cols:
        avg[c] = merged[[f"{c}__{s}" for s in seeds]].mean(axis=1)
    ensemble_action = avg[p_cols].to_numpy().argmax(axis=1)

    joined = TRUE_LBL.reindex(merged.index)
    if joined["zigzag_action"].isna().any():
        raise RuntimeError(f"reindex 후 NaN {int(joined['zigzag_action'].isna().sum())}개 -- dense-cashfill 커버리지 불일치")
    y_true = joined["zigzag_action"].to_numpy()
    both = (y_true != 0) & (ensemble_action != 0)
    cond_acc = float((y_true[both] == ensemble_action[both]).mean() * 100) if both.sum() else float("nan")
    n_long = int((ensemble_action == 1).sum()); n_short = int((ensemble_action == 2).sum())

    ohlc = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                        usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"]).set_index("timestamp")
    ohlc = ohlc.reindex(merged.index)
    if ohlc["close"].isna().any():
        raise RuntimeError("OHLC 정렬 후 결측")
    dec = base_nt.build_dec(ensemble_action)
    m = omega._metrics(ohlc.reset_index(), dec, fee=base_nt._FEE, slip=base_nt._SLIP, cost_mult=base_nt.COST_MULTS["cost3"])
    return {"cond_acc": cond_acc, "long_pct": n_long / max(n_long + n_short, 1) * 100, "pnl": m["pnl"], "mdd": m["mdd"],
            "trades": m["trades"], "wr": m["wr"], "n_bars": int(len(merged))}


def main() -> None:
    rows = []
    for seed in SEEDS:
        acc = _conditional_accuracy(seed)
        pnl = _report_json_pnl(seed)
        rows.append({"seed": seed, **acc, **pnl})
        print(f"  seed={seed}: base_feat={pnl['base_feature_count']} epochs_ran={pnl['epochs_ran']} "
              f"threshold={pnl['threshold']} OOS_pnl={pnl['oos_pnl']:+.2f} cond_acc={acc['cond_acc']:.1f}% "
              f"raw_long%={acc['raw_long_pct']:.1f}%", flush=True)
    accs = [r["cond_acc"] for r in rows]
    pnls = [r["oos_pnl"] for r in rows]
    print(f"\n[2024-2025+noregime 개별5시드] cond_acc: {min(accs):.1f}~{max(accs):.1f}% "
          f"(DC원본(158피쳐/2025단독) 베이스라인: 48.2~51.4%)", flush=True)
    print(f"  OOS_pnl: {min(pnls):+.2f}~{max(pnls):+.2f} 부호일치={'YES' if all(p>0 for p in pnls) or all(p<0 for p in pnls) else 'NO(혼재)'}", flush=True)

    ens = _deep_ensemble(SEEDS)
    print(f"\n[2024-2025+noregime 5시드 딥앙상블] n_bars={ens['n_bars']:,} LONG%={ens['long_pct']:.1f}% "
          f"cond_acc={ens['cond_acc']:.1f}% pnl={ens['pnl']:+.2f} mdd={ens['mdd']:+.2f} "
          f"trades={ens['trades']} wr={ens['wr']:.3f}", flush=True)

    out_path = ROOT / "tmp/eth_candidate_2024_2025_noregime_analysis_20260820.json"
    out_path.write_text(json.dumps({"individual": rows, "ensemble": ens}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
