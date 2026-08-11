"""Knockoff-gate -> mRMR-compress feature analysis for ETH zig075's oracle direction label,
reproducing (independently, in this session) the cross-review in
docs/experiments/eth_knockoff_feature_comparison_h48qual_vs_zig075_20260811.md, which found that
the Step-B fixed-threshold AUC screen (63/138 "passed") has no FDR control and no redundancy
check. Applies that review's three recommendations:

  1. Decontaminate raw funding/whale/squeeze features before testing them (diff1 for
     funding_pressure; rolling(288)-demean for the other seven flagged as price-trend-correlated).
  2. Gate with a Model-X knockoff filter (knockpy, fstat=randomforest, ksampler=gaussian,
     Ledoit-Wolf shrinkage) at FDR 0.10 and 0.20, for both the direction and tradeability
     sub-tasks, instead of the earlier fixed |AUC-0.5|>=0.02 threshold.
  3. Compress the gated set with mRMR (Peng-Long-Ding, mutual-information based) ranked against
     the direction task, then a hard |r|>0.5 dedup pass on the mRMR order.

TRAIN window (2024-06-01..2025-06-30) matches Step B and the h48qual session's reconstruction, so
the FDR-controlled counts are directly comparable to that review's zig075-direction /
zig075-tradeability columns.

Outputs: tmp/eth_zig075_oracle_label_check_20260811/knockoff_mrmr_result.json (+ printed summary).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from knockpy import KnockoffFilter
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]

TECH_PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
REGIME3_PATHS = [
    ROOT / f"data/ensemble/supervised/eth_regime3_current_hmm_jmredesign_20260810_{yr}_maskedname.csv"
    for yr in ("2024", "2025", "2026")
]
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
LABEL_PATHS = [LABEL_DIR / f"zigzag_action_labels_{yr}.csv" for yr in ("2024", "2025", "2026")]

OUT_PATH = ROOT / "tmp/eth_zig075_oracle_label_check_20260811/knockoff_mrmr_result.json"

RAW_LEVEL_COLS = {
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "close_btc", "volume_btc", "quote_volume_btc",
}
DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")

DIFF1_COLS = ["funding_pressure"]
DT288_COLS = [
    "last_funding_rate", "squeeze_power", "long_squeeze_risk", "funding_abs",
    "whale_retail_ratio", "count_long_short_ratio", "sum_toptrader_long_short_ratio",
]

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-06-30")
FDR_LEVELS = (0.10, 0.20)
GATE_FDR = 0.20
MRMR_TOP_K = 20
DEDUP_R = 0.5
SEED = 20260811


def _forbidden(name: str) -> bool:
    low = name.lower()
    return name.startswith(DENY_PREFIXES) or any(tok in low for tok in DENY_TOKENS)


def load_pool(decontaminate: bool = True) -> tuple[pd.DataFrame, list[str]]:
    tech = pd.read_csv(TECH_PANEL_PATH, low_memory=False)
    tech["timestamp"] = pd.to_datetime(tech["timestamp"])
    regime3 = pd.concat([pd.read_csv(p) for p in REGIME3_PATHS], ignore_index=True)
    regime3["timestamp"] = pd.to_datetime(regime3["timestamp"])
    labels = pd.concat(
        [pd.read_csv(p, usecols=["timestamp", "zigzag_action"]) for p in LABEL_PATHS], ignore_index=True
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    df = labels.merge(tech, on="timestamp", how="inner").merge(regime3, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if not decontaminate:
        feat_cols = [c for c in tech.columns if c != "timestamp" and c not in RAW_LEVEL_COLS and not _forbidden(c)]
        feat_cols += [c for c in regime3.columns if c != "timestamp" and not _forbidden(c)]
        return df, feat_cols

    extra = {}
    for col in DIFF1_COLS:
        extra[f"{col}_diff1"] = df[col].diff(1)
    for col in DT288_COLS:
        extra[f"{col}_dt288"] = df[col] - df[col].rolling(288, min_periods=288).mean()
    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    contaminated = set(DIFF1_COLS) | set(DT288_COLS)
    feat_cols = [c for c in tech.columns if c != "timestamp" and c not in RAW_LEVEL_COLS
                 and c not in contaminated and not _forbidden(c)]
    feat_cols += [c for c in regime3.columns if c != "timestamp" and not _forbidden(c)]
    feat_cols += [f"{c}_diff1" for c in DIFF1_COLS] + [f"{c}_dt288" for c in DT288_COLS]
    return df, feat_cols


def make_task_xy(df: pd.DataFrame, feat_cols: list[str], task: str, mask: np.ndarray):
    sub = df.loc[mask]
    action = sub["zigzag_action"].to_numpy()
    if task == "direction":
        keep = action != 0
        y = (action[keep] == 1).astype(int)
    else:
        keep = np.ones(len(sub), dtype=bool)
        y = (action != 0).astype(int)
    x = sub.loc[keep, feat_cols].replace([np.inf, -np.inf], np.nan)
    finite = x.notna().all(axis=1).to_numpy()
    return x.loc[finite].to_numpy(dtype=np.float64), y[finite], int(finite.sum()), int(len(finite))


def run_knockoff_all_fdr(x: np.ndarray, y: np.ndarray, fdr_levels: tuple[float, ...], seed: int,
                          n_estimators: int = 200) -> dict[float, np.ndarray]:
    """Fit the knockoff sampler + feature statistic once (fdr-independent); derive selections at
    every requested FDR level from the same W-statistics instead of refitting per level."""
    kfilter = KnockoffFilter(fstat="randomforest", ksampler="gaussian")
    kfilter.forward(
        X=x, y=y, fdr=fdr_levels[0], shrinkage="ledoitwolf",
        fstat_kwargs={"n_jobs": 4, "random_state": seed, "n_estimators": n_estimators},
    )
    out = {}
    for fdr in fdr_levels:
        rej = kfilter.make_selections(kfilter.W, fdr)
        out[fdr] = np.asarray(rej).astype(bool)
    return out


def mrmr_rank(x: np.ndarray, y: np.ndarray, feat_names: list[str], k: int, seed: int) -> list[str]:
    relevance = mutual_info_classif(x, y, random_state=seed)
    p = x.shape[1]
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    selected: list[int] = []
    remaining = set(range(p))
    redundancy_sum = np.zeros(p)
    for _ in range(min(k, p)):
        if not selected:
            scores = relevance.copy()
        else:
            scores = relevance - redundancy_sum / len(selected)
        for s in selected:
            scores[s] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        remaining.discard(best)
        redundancy_sum += np.abs(corr[best])
    return [feat_names[i] for i in selected]


def hard_dedup(ranked: list[str], x_cols: list[str], corr_lookup: pd.DataFrame, r_thresh: float) -> list[str]:
    kept: list[str] = []
    for f in ranked:
        if all(abs(corr_lookup.loc[f, k]) <= r_thresh for k in kept):
            kept.append(f)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-decontaminate", action="store_true",
                     help="control run: keep raw whale/funding/squeeze features, skip diff1/dt288")
    ap.add_argument("--n-estimators", type=int, default=200, help="RF trees for the knockoff feature statistic")
    ap.add_argument("--out-suffix", default="", help="extra suffix for the output filename")
    args = ap.parse_args()
    decontaminate = not args.no_decontaminate
    tag = ("" if decontaminate else "_raw_control") + args.out_suffix
    out_path = OUT_PATH.with_name(f"knockoff_mrmr_result{tag}.json")

    df, feat_cols = load_pool(decontaminate=decontaminate)
    print(f"candidate pool size (decontaminate={decontaminate}, n_estimators={args.n_estimators}): "
          f"{len(feat_cols)}", flush=True)

    train_mask = ((df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)).to_numpy()

    gate_results: dict[str, dict] = {}
    task_xy = {}
    for task in ("direction", "tradeability"):
        x, y, n_finite, n_total = make_task_xy(df, feat_cols, task, train_mask)
        task_xy[task] = (x, y)
        print(f"[{task}] rows finite={n_finite}/{n_total}  positive_rate={y.mean():.3f}", flush=True)
        rej_by_fdr = run_knockoff_all_fdr(x, y, FDR_LEVELS, SEED, n_estimators=args.n_estimators)
        gate_results[task] = {}
        for fdr, rej in rej_by_fdr.items():
            passed = [feat_cols[i] for i in range(len(feat_cols)) if rej[i]]
            gate_results[task][f"fdr_{fdr}"] = passed
            print(f"[{task}] fdr={fdr}: {len(passed)}/{len(feat_cols)} passed", flush=True)

    gate_set = sorted(set(gate_results["tradeability"][f"fdr_{GATE_FDR}"])
                       | set(gate_results["direction"][f"fdr_{GATE_FDR}"]))
    print(f"gate (tradeability@{GATE_FDR} U direction@{GATE_FDR}): {len(gate_set)} features", flush=True)

    x_dir, y_dir = task_xy["direction"]
    gate_idx = [feat_cols.index(f) for f in gate_set]
    x_gate = x_dir[:, gate_idx]
    ranked = mrmr_rank(x_gate, y_dir, gate_set, MRMR_TOP_K, SEED)
    print(f"mRMR top-{MRMR_TOP_K} (direction target): {ranked}", flush=True)

    corr_df = pd.DataFrame(x_gate, columns=gate_set).corr()
    final = hard_dedup(ranked, gate_set, corr_df, DEDUP_R)
    print(f"final after |r|>{DEDUP_R} hard dedup: {len(final)} -> {final}", flush=True)

    reference = [
        "cvp_regime", "funding_roc_288", "ou_halflife", "vwap_dist_24", "funding_roc_48",
        "breakout_strength", "regime3_current_sensitive_wide24_chop_prob",
    ]
    overlap = sorted(set(final) & set(reference))
    print(f"overlap with h48qual-session reference 7: {overlap}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "decontaminate": decontaminate,
        "candidate_pool_size": len(feat_cols),
        "decontaminated_features": ([f"{c}_diff1" for c in DIFF1_COLS] + [f"{c}_dt288" for c in DT288_COLS]
                                     if decontaminate else []),
        "knockoff_gate_results": {t: {k: v for k, v in d.items()} for t, d in gate_results.items()},
        "gate_fdr_used": GATE_FDR,
        "gate_set": gate_set,
        "mrmr_top_k": ranked,
        "final_after_dedup": final,
        "reference_from_h48qual_session": reference,
        "overlap_with_reference": overlap,
    }, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
