"""
4-step feature audit for the BTC CUSUM-trailing quality model's 156-feature
set (5m base + 1h overlay + metrics4 + regime3), following the session's
literature review (clustered feature importance / substitution effects,
adversarial validation for distribution shift, SHAP-stability across splits).

1. Cluster-level permutation importance (addresses substitution effect --
   correlated features dilute each other's individual importance, as seen
   with regime3's 6 sub-columns).
2. Within-cluster redundancy pruning (|corr| > 0.9 pairs).
3. Adversarial validation (train-vs-OOS classifier; features that easily
   separate the two periods are distribution-shift risks -- the same
   failure mode behind BTC's repeated decay/sign-flip findings this project
   has hit before).
4. SHAP-style (LightGBM native pred_contrib, no extra package needed)
   importance-rank stability across train/VAL/OOS.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402
from build_btc_cusum_trailing_final_model_20260803 import build_trailing_targets, EXCLUDE_COLS, VAL_START, OOS_START, OOS_END  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_regime3_2024_2026.parquet"

CLUSTER_MAP = {
    "1h_overlay": ["ts_action", "ts_t_value", "ts_opt_L", "rsi_14", "rvol_6", "rvol_12", "rvol_24", "rvol_48", "atr_pct", "bb_width", "bb_pos"],
    "regime3": ["bull_prob", "bear_prob", "chop_prob", "confidence", "entropy", "margin"],
    "metrics4": ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio", "taker_vol_ratio_z", "count_toptrader_ratio_z", "toptrader_count_size_divergence", "sig_whale", "sig_oi_divergence"],
    "volatility": ["garman_klass_vol", "rogers_satchell_vol", "parkinson_vol", "bb_width_z", "realized_skewness", "amihud_illiquidity_z"],
    "trend_structure": ["rsi", "macd_hist", "mtf_trend_1h", "mtf_trend_4h", "turtle_signal", "squeeze_power", "chop_index", "wick_ratio", "fvg_dist"],
    "cvp_profile": ["cvp_poc_dist", "cvp_cluster_position", "cvp_volume_imbalance", "cvp_regime", "volume_profile_signal"],
    "flow_whale": ["smart_money_flow", "whale_retail_ratio", "whale_conviction", "long_squeeze_risk"],
    "state_compression": ["regime_persistence", "cross_scale_curvature", "liquidity_vacuum", "crowding_pressure", "execution_quality"],
    "btc_relative": ["btc_corr_60", "eth_btc_ratio_change"],
    "oi_funding": ["last_funding_rate", "funding_price_divergence"],
}


def assign_cluster(col: str) -> str:
    for cluster, keywords in CLUSTER_MAP.items():
        for kw in keywords:
            if kw in col:
                return cluster
    return "other"


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < n - 48 - 2]
    targets = build_trailing_targets(frame, events)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    event_feats = frame.loc[targets["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([targets.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

    train = data[data["timestamp"] < VAL_START].reset_index(drop=True)
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)].reset_index(drop=True)
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)].reset_index(drop=True)

    clusters = {}
    for c in feat_cols:
        clusters.setdefault(assign_cluster(c), []).append(c)
    print("=== feature -> cluster counts ===")
    for k, v in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:20s} n={len(v)}")

    model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    model.fit(train[feat_cols], train["long_net"])
    base_r2 = r2_score(oos["long_net"], model.predict(oos[feat_cols]))
    print(f"\n=== STEP 1: cluster permutation importance (OOS R2 drop, base R2={base_r2:.4f}) ===")
    rng = np.random.default_rng(0)
    cluster_drops = []
    for cname, cols in clusters.items():
        oos_perm = oos.copy()
        for c in cols:
            oos_perm[c] = rng.permutation(oos_perm[c].to_numpy())
        r2_perm = r2_score(oos["long_net"], model.predict(oos_perm[feat_cols]))
        cluster_drops.append((cname, len(cols), base_r2 - r2_perm))
    for cname, ncols, drop in sorted(cluster_drops, key=lambda t: -t[2]):
        print(f"  {cname:20s} n_cols={ncols:3d}  R2_drop={drop:+.4f}")

    print("\n=== STEP 2: within-cluster redundancy (|corr| > 0.9) ===")
    for cname, cols in clusters.items():
        if len(cols) < 2:
            continue
        corr = train[cols].corr().abs()
        pairs = [(cols[i], cols[j], corr.iloc[i, j]) for i in range(len(cols)) for j in range(i + 1, len(cols)) if corr.iloc[i, j] > 0.9]
        if pairs:
            print(f"  [{cname}]")
            for a, b, v in pairs:
                print(f"    {a} <-> {b}  corr={v:.3f}")

    print("\n=== STEP 3: adversarial validation (train vs OOS separability) ===")
    adv_data = pd.concat([train.assign(_is_oos=0), oos.assign(_is_oos=1)], ignore_index=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aucs, importances = [], np.zeros(len(feat_cols))
    for tr_idx, te_idx in skf.split(adv_data[feat_cols], adv_data["_is_oos"]):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05, verbosity=-1, random_state=0)
        m.fit(adv_data.loc[tr_idx, feat_cols], adv_data.loc[tr_idx, "_is_oos"])
        p = m.predict_proba(adv_data.loc[te_idx, feat_cols])[:, 1]
        aucs.append(roc_auc_score(adv_data.loc[te_idx, "_is_oos"], p))
        importances += m.feature_importances_
    print(f"  adversarial AUC (0.5=indistinguishable, 1.0=trivially separable): mean={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    imp_series = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
    print("  top 15 features driving train/OOS distribution shift (highest risk of decay):")
    for c, v in imp_series.head(15).items():
        print(f"    {c:45s} [{assign_cluster(c)}]  imp={v:.0f}")

    print("\n=== STEP 4: SHAP-style (LightGBM pred_contrib) importance rank stability across splits ===")
    def mean_abs_contrib(split_df):
        contrib = model.predict(split_df[feat_cols], pred_contrib=True)
        contrib = np.asarray(contrib)[:, :-1]  # drop bias column
        return pd.Series(np.abs(contrib).mean(axis=0), index=feat_cols)

    shap_train = mean_abs_contrib(train)
    shap_val = mean_abs_contrib(val)
    shap_oos = mean_abs_contrib(oos)
    rank_df = pd.DataFrame({"train_rank": shap_train.rank(ascending=False), "val_rank": shap_val.rank(ascending=False), "oos_rank": shap_oos.rank(ascending=False)})
    rank_df["max_rank_shift"] = (rank_df["train_rank"] - rank_df["oos_rank"]).abs()
    rank_df["cluster"] = [assign_cluster(c) for c in rank_df.index]
    unstable = rank_df.sort_values("max_rank_shift", ascending=False).head(15)
    print("  top 15 LEAST stable features (train->OOS rank shift, higher = more suspect):")
    print(unstable.to_string())
    stable_top20 = rank_df[rank_df["oos_rank"] <= 20].sort_values("oos_rank")
    print("\n  top 20 OOS-important features and their rank stability:")
    print(stable_top20.to_string())


if __name__ == "__main__":
    main()
