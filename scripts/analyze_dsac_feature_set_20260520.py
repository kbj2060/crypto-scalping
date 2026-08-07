#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/llewyn/crypto-scalping")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CSV_2025 = ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519/rl_training_2025_direction_router.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_analysis_20260520"


def _configure_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_ALL_FEATURES_ENABLE"] = "1"
    os.environ["DSAC_RECURRENT_ENABLE"] = "0"
    os.environ["DSAC_ATTN_STACK_ENABLE"] = "1"
    os.environ["DSAC_STACK_N"] = "2"


def _feature_group(name: str) -> str:
    if name.startswith("m7_"):
        return "m7"
    if name.startswith("regime_"):
        return "regime"
    if name.startswith("cvp_"):
        return "cvp"
    if name.startswith("funding_"):
        return "funding"
    if name.startswith("sig_"):
        return "elite_signal"
    if name.startswith("ai_"):
        return "ai_overlay"
    if name.startswith("timesnet_"):
        return "timesnet"
    if name.startswith("dlinear_"):
        return "dlinear"
    if name.startswith("clean_regime4_"):
        return "clean_regime4"
    if name.startswith("ret_"):
        return "lag_return"
    if name in {"open", "high", "low", "close", "volume", "quote_volume", "trades", "close_btc", "volume_btc", "quote_volume_btc"}:
        return "raw_market"
    if name in {"hour_sin", "hour_cos", "minute_sin", "minute_cos", "session_europe", "session_us", "is_hour_open"}:
        return "time"
    return "other"


def main() -> None:
    _configure_env()
    from ensemble import train_rl_dsac_agent as dsac

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_2025)

    all_cols = list(dsac.DSAC_ALL_FEATURE_COLS)
    present = [c for c in all_cols if c in df.columns]

    close = pd.to_numeric(df["close"], errors="coerce")
    fwd_ret_12 = close.shift(-12) / close - 1.0
    fwd_ret_48 = close.shift(-48) / close - 1.0

    rows = []
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.notna()
        miss = float(1.0 - valid.mean())
        if valid.any():
            sv = s[valid].astype(float)
            std = float(sv.std(ddof=0))
            mean = float(sv.mean())
            zero_rate = float((np.isclose(sv.to_numpy(), 0.0)).mean())
            nunique = int(sv.nunique(dropna=True))
            rho12 = float(sv.corr(fwd_ret_12, method="spearman"))
            rho48 = float(sv.corr(fwd_ret_48, method="spearman"))
        else:
            std = mean = zero_rate = rho12 = rho48 = 0.0
            nunique = 0
        rows.append(
            {
                "feature": col,
                "group": _feature_group(col),
                "missing_rate": miss,
                "zero_rate": zero_rate,
                "nunique": nunique,
                "mean": mean,
                "std": std,
                "spearman_fwd12": rho12,
                "abs_spearman_fwd12": abs(rho12),
                "spearman_fwd48": rho48,
                "abs_spearman_fwd48": abs(rho48),
            }
        )

    score_df = pd.DataFrame(rows).sort_values(
        ["abs_spearman_fwd48", "abs_spearman_fwd12", "std"],
        ascending=[False, False, False],
    )
    score_df.to_csv(OUT_DIR / "feature_scores.csv", index=False)

    corr_cols = [c for c in present if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.95]
    corr_frame = df[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_frame.corr(method="spearman").abs()
    redundant = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            v = corr.loc[c1, c2]
            if np.isfinite(v) and v >= 0.97:
                redundant.append(
                    {
                        "feature_a": c1,
                        "feature_b": c2,
                        "abs_spearman": float(v),
                        "group_a": _feature_group(c1),
                        "group_b": _feature_group(c2),
                    }
                )
    redundant_df = pd.DataFrame(redundant).sort_values("abs_spearman", ascending=False)
    redundant_df.to_csv(OUT_DIR / "redundant_pairs.csv", index=False)

    group_summary = (
        score_df.groupby("group")
        .agg(
            feature_count=("feature", "count"),
            median_abs_spearman_fwd12=("abs_spearman_fwd12", "median"),
            median_abs_spearman_fwd48=("abs_spearman_fwd48", "median"),
            mean_missing_rate=("missing_rate", "mean"),
            mean_zero_rate=("zero_rate", "mean"),
        )
        .sort_values("median_abs_spearman_fwd48", ascending=False)
        .reset_index()
    )
    group_summary.to_csv(OUT_DIR / "group_summary.csv", index=False)

    summary = {
        "csv_path": str(CSV_2025),
        "all_feature_count": len(all_cols),
        "present_feature_count": len(present),
        "top_fwd48": score_df[["feature", "group", "spearman_fwd48", "abs_spearman_fwd48"]].head(20).to_dict(orient="records"),
        "top_fwd12": score_df[["feature", "group", "spearman_fwd12", "abs_spearman_fwd12"]].head(20).to_dict(orient="records"),
        "redundant_pair_count": int(len(redundant)),
        "top_redundant_pairs": redundant_df.head(30).to_dict(orient="records"),
        "group_summary": group_summary.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "present_feature_count": len(present), "redundant_pair_count": len(redundant)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
