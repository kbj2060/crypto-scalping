"""Omega-Scalp: a multi-component architecture mirroring Omega4.6.1's composed-model structure
(direction parent + quality/risk heads + regime routing + duration gate), adapted to 1-minute
scalping. Every individual component below (meta-quality gate, TabM, DP/trend-scan labels) was
already tested ALONE this session and lost to the simple baseline
(scalp_1m_tune_maker_realistic_20260716.json, OOS +3.74%) -- this script tests the hypothesis
that COMPOSING several weak components produces a net win the way Omega's router-combine does,
not that any single component here is individually better.

Components:
  1. REGIME FILTER  -- rule-based, mirrors Sigma6's single strongest 1h lever (a "not-chop"
     veto) which has NOT been tried yet at 1m. chop_index (below train-fit 33rd pct = trending)
     + hurst_48 (above train-fit 50th pct = persistent, not mean-reverting) -> hard veto, trades
     are dropped entirely regardless of every other component's score.
  2. DIRECTION MODEL -- unchanged HGB primary classifier (the already-validated baseline).
  3. META-QUALITY GATE -- purged-OOF binary classifier predicting realized profitability
     (same construction as train_eval_scalp_1m_meta_label_20260716.py).
  4. DURATION MODEL -- HGB regressor predicting scalp_resolution_offset (bars until the
     triple-barrier resolves) -- fast-resolving signals historically carry more of the edge
     (this is the same intuition build_scalp_1m_tb_labels_weighted_20260716.py's uniqueness
     weighting captures, exposed here as a router feature instead of a training weight).
  5. RISK SIDECAR -- HGB regressor predicting expected net PnL magnitude (continuous, not
     binary) for a taken trade -- lets the router prioritize by expected edge size, not just
     win/loss probability.
  6. ROUTER -- a small HGB binary classifier stacked on [primary_proba, meta_quality_proba,
     duration_pred, risk_sidecar_pred] (regime already applied as a hard veto before this
     stage) -- the final composite decision, trained on the same purged-OOF rows as its inputs.

KNOWN LIMITATION (stated explicitly rather than hidden): components 3-5 and the router are all
trained on the SAME single-pass 5-block purged-OOF set, i.e. the router sees in-sample
predictions from the meta/duration/risk models rather than a further-nested OOF layer -- a true
stacking architecture would nest CV again at the router level. This is a pragmatic scope
compromise; if it caused material overfitting it would show up as OOS underperformance, which is
checked below rather than assumed away.

Output: data/ensemble/reports/scalp_1m_omega_style_20260716.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, HGB_PARAMS
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, ROUND_TRIP_FEE
from train_eval_scalp_1m_meta_label_20260716 import (
    generate_oof_primary, fit_primary, primary_predict, N_BLOCKS,
)

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_WEIGHTED_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels_weighted.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
ROUTER_THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

REGIME_COLS = ['chop_index', 'hurst_48']
DURATION_HGB_PARAMS = dict(
    loss='squared_error', learning_rate=0.05, max_iter=200, max_depth=4,
    l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=100,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=0,
)
META_HGB_PARAMS = dict(
    loss='log_loss', learning_rate=0.05, max_iter=250, max_depth=4,
    l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=100,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    class_weight='balanced', random_state=0,
)
ROUTER_HGB_PARAMS = dict(
    loss='log_loss', learning_rate=0.05, max_iter=150, max_depth=3,
    l2_regularization=1.0, max_leaf_nodes=15, min_samples_leaf=50,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
    class_weight='balanced', random_state=0,
)


def fit_regime_cutoffs(train: pd.DataFrame) -> dict:
    return {
        'chop_index_p33': float(train['chop_index'].quantile(0.33)),
        'hurst_48_p50': float(train['hurst_48'].quantile(0.50)),
    }


def regime_pass(df: pd.DataFrame, cutoffs: dict) -> np.ndarray:
    """Hard veto: only 'trending, not choppy' bars pass."""
    return ((df['chop_index'] <= cutoffs['chop_index_p33']) &
             (df['hurst_48'] >= cutoffs['hurst_48_p50'])).to_numpy()


def build_meta_and_risk_targets(oof: pd.DataFrame, df: pd.DataFrame,
                                 long_sim: pd.DataFrame, short_sim: pd.DataFrame) -> pd.DataFrame:
    non_cash = oof[oof['oof_pred'] != 'CASH'].copy()
    is_long = non_cash['oof_pred'] == 'LONG'
    filled = np.where(is_long, long_sim.loc[non_cash.index, 'filled'], short_sim.loc[non_cash.index, 'filled'])
    move = np.where(is_long, long_sim.loc[non_cash.index, 'realized_move'], short_sim.loc[non_cash.index, 'realized_move'])
    net_pnl = np.where(filled, move - ROUND_TRIP_FEE, -ROUND_TRIP_FEE)  # unfilled treated as a wash loss of the fee opportunity
    non_cash['filled'] = filled
    non_cash['net_pnl'] = net_pnl
    non_cash['meta_y'] = ((filled) & (net_pnl > 0)).astype(int)
    # df already has scalp_resolution_offset merged in (same row-position index space as
    # oof/long_sim/short_sim, all derived from this same post-merge df) -- pull directly rather
    # than re-joining a separately-indexed copy of the labels file.
    non_cash = non_cash.join(df.loc[non_cash.index, ['scalp_resolution_offset']], how='left')
    return non_cash


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + weighted labels...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    wlabels = pd.read_csv(LABELS_WEIGHTED_CSV, parse_dates=['timestamp'])
    df = df.merge(wlabels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    train = df[df['timestamp'] <= TRAIN_END].reset_index(drop=False).rename(columns={'index': 'orig_idx'})
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")

    print("\n--- Component 1: regime filter (rule-based, train-fit cutoffs) ---")
    regime_cutoffs = fit_regime_cutoffs(train)
    print(f"  cutoffs: {regime_cutoffs}")

    print("\n--- Purged 5-block OOF primary predictions (shared by components 3-5) ---")
    oof = generate_oof_primary(train[['timestamp'] + feat_cols + ['scalp_action', 'scalp_uniqueness_weight']], feat_cols)
    oof['orig_idx'] = train['orig_idx'].to_numpy()
    oof = oof.set_index('orig_idx')

    print("\n--- Component 3+5 targets: meta-quality (binary) + risk sidecar (continuous net_pnl) ---")
    rows = build_meta_and_risk_targets(oof, df, long_sim, short_sim)
    rows = rows.join(df.loc[rows.index, feat_cols])
    print(f"  OOF non-CASH rows: {len(rows):,}, meta positive rate={rows['meta_y'].mean():.3f}, "
          f"mean net_pnl={rows['net_pnl'].mean():.5f}")

    router_feat_cols = feat_cols + ['oof_proba']
    X_router_train = rows[router_feat_cols].fillna(0.0)

    print("\n--- Component 4: duration model (predict resolution offset) ---")
    duration_model = HistGradientBoostingRegressor(**DURATION_HGB_PARAMS)
    duration_model.fit(X_router_train, rows['scalp_resolution_offset'].fillna(20.0))

    print("--- Component 3: meta-quality classifier ---")
    meta_model = HistGradientBoostingClassifier(**META_HGB_PARAMS)
    meta_model.fit(X_router_train, rows['meta_y'])

    print("--- Component 5: risk sidecar regressor ---")
    risk_model = HistGradientBoostingRegressor(**DURATION_HGB_PARAMS)
    risk_model.fit(X_router_train, rows['net_pnl'])

    print("\n--- Component 6: router (stacks components 2-5's OOF outputs) ---")
    rows['duration_pred'] = duration_model.predict(X_router_train)
    rows['meta_pred'] = meta_model.predict_proba(X_router_train)[:, 1]
    rows['risk_pred'] = risk_model.predict(X_router_train)
    router_input_cols = ['oof_proba', 'duration_pred', 'meta_pred', 'risk_pred']
    router = HistGradientBoostingClassifier(**ROUTER_HGB_PARAMS)
    router.fit(rows[router_input_cols], rows['meta_y'])

    print("\n--- Final (deployed) primary model on full purged training period ---")
    final_primary = fit_primary(train, feat_cols)

    def score_split(split_df: pd.DataFrame):
        regime_ok = regime_pass(split_df, regime_cutoffs)
        X = split_df[feat_cols].fillna(0.0)
        pred, proba = primary_predict(final_primary, X)
        non_cash = pred != 'CASH'
        eligible = regime_ok & non_cash

        router_scores = np.zeros(len(split_df))
        if eligible.any():
            X_elig = pd.concat([split_df.loc[eligible, feat_cols], pd.Series(proba[eligible], index=split_df.index[eligible], name='oof_proba')], axis=1)
            dur_pred = duration_model.predict(X_elig[router_feat_cols])
            meta_pred = meta_model.predict_proba(X_elig[router_feat_cols])[:, 1]
            risk_pred = risk_model.predict(X_elig[router_feat_cols])
            router_X = pd.DataFrame({'oof_proba': proba[eligible], 'duration_pred': dur_pred,
                                      'meta_pred': meta_pred, 'risk_pred': risk_pred})
            router_scores[eligible.to_numpy() if hasattr(eligible, 'to_numpy') else eligible] = router.predict_proba(router_X)[:, 1]
        final_pred = np.where(eligible, pred, 'CASH')
        return final_pred, router_scores

    print("\n--- Sweeping router threshold on val ---")
    val_pred, val_router_score = score_split(val)
    val_idx = val.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)

    sweep = []
    for thr in ROUTER_THRESHOLDS:
        gated = np.where(val_router_score >= thr, val_pred, 'CASH')
        bt = backtest_maker(gated, long_sim_val, short_sim_val)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val router_thr={thr}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_filled'] and s['n_filled'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else sweep[0]
    best_thr = best['threshold']
    print(f"-> best val router threshold: {best_thr}")

    print("\n--- OOS evaluation ---")
    oos_pred, oos_router_score = score_split(oos)
    oos_idx = oos.index
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)
    gated_oos = np.where(oos_router_score >= best_thr, oos_pred, 'CASH')
    bt_oos = backtest_maker(gated_oos, long_sim_oos, short_sim_oos)
    print(f"[oos @ router_thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
          f"fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} total_pnl_pct={bt_oos['total_pnl_pct']}")

    # component-ablation reference: regime-veto-only (no router), for diagnosing which piece helps
    regime_only_pred = np.where(regime_pass(oos, regime_cutoffs) & (oos_pred != 'CASH'), oos_pred, 'CASH')
    bt_regime_only = backtest_maker(regime_only_pred, long_sim_oos, short_sim_oos)
    print(f"[oos regime-veto-only, no router] signals={bt_regime_only['n_signals']:,} "
          f"total_pnl_pct={bt_regime_only['total_pnl_pct']}")

    result = {
        'components': ['regime_filter', 'direction_hgb', 'meta_quality', 'duration_model', 'risk_sidecar', 'router'],
        'regime_cutoffs': regime_cutoffs,
        'val_sweep': sweep,
        'chosen_router_threshold': best_thr,
        'oos_at_chosen_threshold': bt_oos,
        'oos_regime_veto_only_no_router': bt_regime_only,
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json',
            'oos_total_pnl_pct': 3.7390646402123644,
        },
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'known_limitation': ('Components 3-5 and the router are trained on the same single-pass '
                              'purged-OOF set (no further router-level CV nesting) -- see module '
                              'docstring.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_omega_style_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_omega_style_20260716.json")


if __name__ == '__main__':
    main()
