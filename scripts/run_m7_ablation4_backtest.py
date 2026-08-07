#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT.parent / "anaconda3" / "envs" / "quant_ai" / "bin" / "python"
DEFAULT_DROP_COLS = ["funding_roc_48", "funding_roc_12", "funding_z_score", "session_us"]


def _run(cmd: list[str], desc: str, pbar: tqdm) -> None:
    pbar.set_description(desc)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    pbar.update(1)


def _write_pruned_csv(src: Path, dst: Path, drop_cols: list[str]) -> None:
    df = pd.read_csv(src)
    cols = [c for c in drop_cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def _generate_nohdb_dataset(feature_path: Path, output_path: Path, drop_cols: list[str]) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from pipeline.augment_m7_dataset import (
        _derive_prereq_features,
        _load_frames,
        _write_meta,
        compute_new_elite_signals,
        compute_regime,
        compute_synthetic_alphas,
        compute_volatility_models,
        EliteSignals,
        row_to_market_row,
    )
    from ensemble.seven_model_ensemble import SevenModelEnsemble
    from features.registry import get_m7_columns
    from features.schema import build_rl_feature_keep

    rl_path = ROOT / "data" / "splits" / "year_oos" / "rl_base_2025.csv"
    rl_df, work_df = _load_frames(str(rl_path), str(feature_path), "timestamp")
    for col in drop_cols:
        if col not in work_df.columns:
            work_df[col] = 0.0

    work_df = _derive_prereq_features(work_df)
    work_df = compute_synthetic_alphas(work_df)
    work_df = compute_regime(work_df)
    work_df = compute_volatility_models(work_df)
    work_df = compute_new_elite_signals(work_df)

    elite = EliteSignals()
    if "smart_money_flow" in work_df.columns:
        smf_std = (
            work_df["smart_money_flow"]
            .rolling(window=576, min_periods=10)
            .std()
            .fillna(work_df["smart_money_flow"].expanding(min_periods=1).std())
            .fillna(1.0)
        )
    else:
        smf_std = pd.Series(1.0, index=work_df.index)

    keys = [
        "sig_whale",
        "sig_oi_divergence",
        "sig_ai_squeeze",
        "sig_orderblock",
        "sig_liq_squeeze",
        "sig_net_taker",
        "sig_hurst_ofi",
        "sig_funding_cascade",
        "sig_multifractal",
        "sig_cluster_fib",
        "sig_top_trader_squeeze",
        "sig_btc_corr_breakout",
        "sig_garch_regime",
        "sig_ou_mean_rev",
        "sig_jump_rebound",
        "sig_evt_tail",
    ]
    for k in keys:
        if k not in work_df.columns:
            work_df[k] = 0.0

    records = work_df.to_dict("records")
    for i in tqdm(range(len(records)), desc="elite-signals", unit="row"):
        cur = row_to_market_row(records[i])
        prev = row_to_market_row(records[i - 1]) if i > 0 else cur
        sigs = elite.compute_all(current=cur, prev=prev, smf_std=float(smf_std.iloc[i]))
        for k in keys:
            if k in sigs:
                work_df.at[i, k] = float(sigs[k])

    null_meta_dir = ROOT / "data" / "ensemble" / "_null_meta"
    null_meta_dir.mkdir(parents=True, exist_ok=True)
    null_hdb = null_meta_dir / "hdbscan_regime.json"
    if not null_hdb.exists():
        null_hdb.write_text("{}", encoding="utf-8")

    ensemble = SevenModelEnsemble(meta_paths={"hdbscan_regime": str(null_hdb)}, strict=False)
    m7 = ensemble.predict_batch(work_df)
    raw_m7_cols = list(m7.columns)
    drop_cols = [c for c in get_m7_columns("deprecated", include_entry_price=True) if c in m7.columns]
    if drop_cols:
        m7 = m7.drop(columns=drop_cols)

    base = rl_df.copy()
    overlap = [c for c in m7.columns if c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)

    rl_keep = set(build_rl_feature_keep(include_entry_price=False))
    passthrough_cols = [c for c in work_df.columns if c in rl_keep and c not in base.columns and c not in m7.columns]
    passthrough = work_df[passthrough_cols].reset_index(drop=True) if passthrough_cols else pd.DataFrame(index=base.index)
    out_df = pd.concat([base.reset_index(drop=True), passthrough, m7.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    _write_meta(
        output_path=str(output_path),
        rl_path=str(rl_path),
        feature_path=str(feature_path),
        row_count=len(out_df),
        col_count=len(out_df.columns),
        m7_cols=raw_m7_cols,
        dropped_cols=drop_cols,
    )
    return {"rows": len(out_df), "cols": len(out_df.columns), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain pruned M7 models and backtest with tqdm progress.")
    parser.add_argument("--xgb-trials", type=int, default=12)
    parser.add_argument("--quantile-trials", type=int, default=10)
    parser.add_argument("--unsup-trials", type=int, default=8)
    parser.add_argument("--vae-trials", type=int, default=6)
    parser.add_argument("--drop-cols", type=str, default=",".join(DEFAULT_DROP_COLS), help="Comma-separated feature columns to drop.")
    parser.add_argument("--tag", type=str, default="ablation4", help="Tag used in generated filenames.")
    args = parser.parse_args()
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    pbar = tqdm(total=11, desc="m7-ablation4", unit="step")

    feat2024 = ROOT / "data" / "splits" / "year_oos" / f"training_features_2024_pruned_{args.tag}.csv"
    feat2025 = ROOT / "data" / "splits" / "year_oos" / f"training_features_2025_pruned_{args.tag}.csv"
    _write_pruned_csv(ROOT / "data" / "splits" / "year_oos" / "training_features_2024.csv", feat2024, drop_cols)
    _write_pruned_csv(ROOT / "data" / "splits" / "year_oos" / "training_features_2025.csv", feat2025, drop_cols)
    pbar.update(1)

    _run(
        [str(PYTHON), "ensemble/supervised/train_entry_price_model.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv"],
        "2/11 entry_price",
        pbar,
    )
    _run(
        [str(PYTHON), "ensemble/supervised/train_trend_xgb.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv", "--n-trials", str(args.xgb_trials)],
        "3/11 trend_xgb",
        pbar,
    )

    backup_sup = ROOT / "data" / "ensemble" / "_feature_prune_backup_20260422" / "supervised"
    shutil.copy2(backup_sup / "multi_target_lgbm.json", ROOT / "data" / "ensemble" / "supervised" / "multi_target_lgbm.json")
    shutil.copy2(backup_sup / "multi_target_lgbm.pkl", ROOT / "data" / "ensemble" / "supervised" / "multi_target_lgbm.pkl")
    shutil.copy2(backup_sup / "multitarget_lgbm_training_results.json", ROOT / "data" / "ensemble" / "supervised" / "multitarget_lgbm_training_results.json")
    pbar.set_description("4/11 restore_mtl")
    pbar.update(1)

    _run(
        [str(PYTHON), "ensemble/supervised/train_quantile_forest.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv", "--n-trials", str(args.quantile_trials)],
        "5/11 quantile",
        pbar,
    )
    _run(
        [str(PYTHON), "ensemble/unsupervised/train_gmm_volatility.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv", "--n-trials", str(args.unsup_trials)],
        "6/11 gmm",
        pbar,
    )
    _run(
        [str(PYTHON), "ensemble/unsupervised/train_isolation_forest.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv", "--n-trials", str(args.unsup_trials)],
        "7/11 iso",
        pbar,
    )
    _run(
        [str(PYTHON), "ensemble/unsupervised/train_vae_anomaly.py", "--data-path", str(feat2024.relative_to(ROOT)), "--rl-path", "data/splits/year_oos/rl_base_2024.csv", "--n-trials", str(args.vae_trials)],
        "8/11 vae",
        pbar,
    )

    pbar.set_description("9/11 generate_nohdb")
    gen = _generate_nohdb_dataset(
        feature_path=feat2025,
        output_path=ROOT / "data" / "splits" / "year_oos" / f"rl_training_2025_m7_{args.tag}_nohdb.csv",
        drop_cols=drop_cols,
    )
    pbar.update(1)

    _run(
        [
            str(PYTHON),
            "scripts/backtest_m7_signal_only.py",
            "--csv",
            f"data/splits/year_oos/rl_training_2025_m7_{args.tag}_nohdb.csv",
            "--out",
            f"data/ensemble/reports/backtest_m7_signal_only_{args.tag}_nohdb_2025.json",
        ],
        "10/11 backtest",
        pbar,
    )

    pbar.set_description("11/11 summarize")
    out = ROOT / "data" / "ensemble" / "reports" / f"backtest_m7_signal_only_{args.tag}_nohdb_2025.json"
    summary = json.loads(out.read_text(encoding="utf-8"))
    print(json.dumps({"tag": args.tag, "drop_cols": drop_cols, "generated": gen, "backtest": summary}, ensure_ascii=False, indent=2))
    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
