#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.augment_m7_dataset import (
    _derive_prereq_features,
    _load_frames,
    compute_new_elite_signals,
    compute_regime,
    compute_synthetic_alphas,
    compute_volatility_models,
    EliteSignals,
    row_to_market_row,
)
from scripts.backtest_m7_signal_only import run_backtest
from ensemble.seven_model_ensemble import SevenModelEnsemble
from features.registry import get_m7_columns
from features.schema import build_rl_feature_keep
NULL_META_DIR = ROOT / "data" / "ensemble" / "_null_meta"
LOW_IMPORTANCE_FEATURES = [
    "is_hour_open",
    "session_europe",
    "session_us",
    "regime_trending",
    "mta_funding",
    "funding_roc_12",
    "dual_momentum",
]
FEATURE_GROUPS = {
    "low_imp_core": LOW_IMPORTANCE_FEATURES,
    "session_time": ["is_hour_open", "session_europe", "session_us", "hour_sin", "hour_cos", "minute_cos"],
    "funding_weak": ["mta_funding", "funding_roc_12", "funding_roc_48", "funding_roc_288", "funding_z_score"],
    "redundant_vol": ["garch_vol_z", "parkinson_vol", "garman_klass_vol"],
    "whale_flow": ["sig_whale", "whale_retail_ratio", "whale_conviction", "smart_money_flow"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search lightweight M7 config variants with progress bars.")
    parser.add_argument("--eval-rows", type=int, default=12000, help="Rows to actually evaluate/backtest.")
    parser.add_argument("--warmup-rows", type=int, default=3000, help="Extra historical rows kept for rolling features.")
    parser.add_argument("--config-limit", type=int, default=6, help="Number of configs to test from the built-in list.")
    parser.add_argument("--search-mode", choices=["preset", "features"], default="preset", help="Preset config search or staged feature ablation search.")
    parser.add_argument("--topk-features", type=int, default=5, help="Top single ablations to keep for exhaustive subset search.")
    return parser.parse_args()


def _prepare_work_frame(eval_rows: int, warmup_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rl_path = ROOT / "data" / "splits" / "year_oos" / "rl_base_2025.csv"
    feat_path = ROOT / "data" / "splits" / "year_oos" / "training_features_2025.csv"
    rl_df, work_df = _load_frames(str(rl_path), str(feat_path), "timestamp")
    keep_rows = max(int(eval_rows), 1) + max(int(warmup_rows), 0)
    if keep_rows < len(rl_df):
        rl_df = rl_df.tail(keep_rows).reset_index(drop=True)
        work_df = work_df.tail(keep_rows).reset_index(drop=True)
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
        "sig_whale", "sig_oi_divergence", "sig_ai_squeeze", "sig_orderblock",
        "sig_liq_squeeze", "sig_net_taker", "sig_hurst_ofi", "sig_funding_cascade",
        "sig_multifractal", "sig_cluster_fib", "sig_top_trader_squeeze", "sig_btc_corr_breakout",
        "sig_garch_regime", "sig_ou_mean_rev", "sig_jump_rebound", "sig_evt_tail",
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
    if eval_rows < len(rl_df):
        rl_df = rl_df.tail(eval_rows).reset_index(drop=True)
        work_df = work_df.tail(eval_rows).reset_index(drop=True)
    return rl_df, work_df


def _null_meta_path(name: str) -> str:
    NULL_META_DIR.mkdir(parents=True, exist_ok=True)
    p = NULL_META_DIR / f"{name}.json"
    if not p.exists():
        p.write_text("{}", encoding="utf-8")
    return str(p)


def _build_dataset(rl_df: pd.DataFrame, work_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    model_work_df = work_df.copy()
    zero_features = list(cfg.get("zero_features", []))
    for col in zero_features:
        if col in model_work_df.columns:
            model_work_df[col] = 0.0

    meta_paths = {}
    for name in cfg.get("omit_models", []):
        meta_paths[name] = _null_meta_path(name)
    ensemble = SevenModelEnsemble(
        meta_paths=meta_paths or None,
        weight_trend_xgb=float(cfg["weights"][0]),
        weight_multitarget=float(cfg["weights"][1]),
        weight_quantile=float(cfg["weights"][2]),
        strict=False,
    )
    m7 = ensemble.predict_batch(model_work_df)
    drop_cols = [c for c in get_m7_columns("deprecated", include_entry_price=True) if c in m7.columns]
    if drop_cols:
        m7 = m7.drop(columns=drop_cols)

    base = rl_df.copy()
    overlap = [c for c in m7.columns if c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)

    rl_keep = set(build_rl_feature_keep(include_entry_price=False))
    passthrough_cols = [c for c in model_work_df.columns if c in rl_keep and c not in base.columns and c not in m7.columns]
    passthrough = model_work_df[passthrough_cols].reset_index(drop=True) if passthrough_cols else pd.DataFrame(index=base.index)
    return pd.concat([base.reset_index(drop=True), passthrough, m7.reset_index(drop=True)], axis=1)


def _score(res: dict) -> float:
    pnl = float(res["pnl_pct"])
    sharpe = float(res["sharpe"])
    wr = float(res["win_rate_pct"])
    pf = float(res["profit_factor"])
    trades = float(res["trades"])
    return pnl + 0.03 * sharpe + 0.01 * wr + 0.5 * min(pf, 2.0) - 0.002 * abs(trades - 20000.0)


def _evaluate_configs(rl_df: pd.DataFrame, work_df: pd.DataFrame, configs: list[dict], run_desc: str) -> list[dict]:
    results = []
    for cfg in tqdm(configs, desc=run_desc, unit="cfg"):
        df = _build_dataset(rl_df, work_df, cfg)
        res = run_backtest(df, fee_bps=2.0, slip_bps=1.0)
        row = {
            "name": cfg["name"],
            "weights": list(cfg["weights"]),
            "omit_models": list(cfg["omit_models"]),
            "zero_features": list(cfg.get("zero_features", [])),
            **json.loads(json.dumps(res, default=lambda o: o.__dict__)),
        }
        row["score"] = _score(row)
        results.append(row)
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "pnl_pct": row["pnl_pct"],
                    "sharpe": row["sharpe"],
                    "trades": row["trades"],
                    "wr": row["win_rate_pct"],
                    "pf": row["profit_factor"],
                    "score": row["score"],
                    "zero_features": row["zero_features"],
                },
                ensure_ascii=False,
            )
        )
    return results


def _feature_search(rl_df: pd.DataFrame, work_df: pd.DataFrame, topk_features: int) -> dict:
    base_cfg = {"name": "baseline_no_hdb", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": []}
    candidate_features = [
        "is_hour_open",
        "session_us",
        "session_europe",
        "regime_trending",
        "oi_change_rate",
        "funding_roc_12",
        "dual_momentum",
        "mta_funding",
        "minute_sin",
        "minute_cos",
        "funding_roc_48",
        "funding_roc_288",
        "funding_z_score",
    ]
    single_configs = [base_cfg] + [
        {
            "name": f"drop_{feat}",
            "weights": base_cfg["weights"],
            "omit_models": list(base_cfg["omit_models"]),
            "zero_features": [feat],
        }
        for feat in candidate_features
    ]
    stage1 = _evaluate_configs(rl_df, work_df, single_configs, "single-feature-search")
    baseline = next(r for r in stage1 if r["name"] == "baseline_no_hdb")
    improved = [r for r in stage1 if r["name"] != "baseline_no_hdb" and r["score"] > baseline["score"]]
    improved = sorted(improved, key=lambda x: x["score"], reverse=True)
    top_features = [r["zero_features"][0] for r in improved[: max(1, topk_features)]]
    if not top_features:
        top_features = [r["zero_features"][0] for r in sorted(stage1[1:], key=lambda x: x["score"], reverse=True)[: max(1, topk_features)]]

    subset_configs = [base_cfg]
    for size in range(1, len(top_features) + 1):
        for subset in combinations(top_features, size):
            subset_configs.append(
                {
                    "name": "subset_" + "__".join(subset),
                    "weights": base_cfg["weights"],
                    "omit_models": list(base_cfg["omit_models"]),
                    "zero_features": list(subset),
                }
            )
    stage2 = _evaluate_configs(rl_df, work_df, subset_configs, "subset-search")
    stage2 = sorted(stage2, key=lambda x: x["score"], reverse=True)
    return {
        "search_mode": "features",
        "candidate_features": candidate_features,
        "top_features": top_features,
        "single_feature_results": sorted(stage1, key=lambda x: x["score"], reverse=True),
        "subset_results": stage2,
        "best": deepcopy(stage2[0]) if stage2 else None,
    }


def main() -> None:
    args = _parse_args()
    rl_df, work_df = _prepare_work_frame(eval_rows=args.eval_rows, warmup_rows=args.warmup_rows)

    configs = [
        {"name": "baseline", "weights": (0.45, 0.35, 0.20), "omit_models": [], "zero_features": []},
        {"name": "no_hdb", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": []},
        {"name": "lowimp_zero", "weights": (0.45, 0.35, 0.20), "omit_models": [], "zero_features": FEATURE_GROUPS["low_imp_core"]},
        {"name": "no_hdb_lowimp", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["low_imp_core"]},
        {"name": "no_hdb_sessions", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["session_time"]},
        {"name": "no_hdb_fundingweak", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["funding_weak"]},
        {"name": "trend_heavy_lowimp", "weights": (0.60, 0.25, 0.15), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["low_imp_core"]},
        {"name": "quant_heavy_lowimp", "weights": (0.35, 0.20, 0.45), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["low_imp_core"]},
        {"name": "no_hdb_redundant_vol", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["redundant_vol"]},
        {"name": "no_hdb_whale_flow", "weights": (0.45, 0.35, 0.20), "omit_models": ["hdbscan_regime"], "zero_features": FEATURE_GROUPS["whale_flow"]},
    ]
    configs = configs[: max(1, min(args.config_limit, len(configs)))]

    print(
        json.dumps(
            {
                "eval_rows": len(rl_df),
                "warmup_rows": args.warmup_rows,
                "config_count": len(configs),
                "search_mode": args.search_mode,
            },
            ensure_ascii=False,
        )
    )
    if args.search_mode == "features":
        out = _feature_search(rl_df, work_df, topk_features=args.topk_features)
    else:
        results = _evaluate_configs(rl_df, work_df, configs, "config-search")
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        out = {
            "search_mode": "preset",
            "results": results,
            "best": deepcopy(results[0]) if results else None,
        }
    out_path = ROOT / "data" / "ensemble" / "reports" / "m7_config_search_2025.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
