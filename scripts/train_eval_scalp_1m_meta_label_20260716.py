"""Lever A-1/B-2: meta-labeling. Replaces the fixed confidence threshold (0.55, chosen once by
eyeballing a val sweep) with a LEARNED gate: a second binary classifier that predicts whether a
given primary LONG/SHORT signal will actually be net-profitable after realistic maker-fill costs,
trained on purged out-of-fold (OOF) primary predictions so it never sees the primary model's own
memorized (in-sample) confidence -- using in-sample confidence directly would let the meta-model
learn "the primary model is confident" rather than "the trade will actually work," which is a
subtle but real leakage mode in meta-labeling per Lopez de Prado (Advances in Financial Machine
Learning, ch.3).

Pipeline:
  1. Purged 5-block CV over the training period only: for each of 5 contiguous chronological
     blocks, fit a primary HGB on the other 4 blocks (with HORIZON-minute purge at both
     boundaries of the held-out block, since a training row's label can look forward past a
     block boundary) and predict on the held-out block -> OOF primary class + max-probability for
     every training row.
  2. Meta target: for OOF rows where the primary predicted LONG/SHORT, use the realistic
     maker-fill simulation's realized outcome (same fill model as
     simulate_maker_entry_scalp_1m_20260716.py) -> binary target = 1 if filled AND net PnL > 0
     after fees, else 0 (unfilled counts as a "no" -- a gate that also learns to avoid
     signals unlikely to get a good fill is exactly what we want).
  3. Meta model: binary HGB classifier on [primary OOF max-proba + full feature set] -> full
     features let the meta-model learn conditions under which the primary's confidence is/isn't
     trustworthy, not just recalibrate the raw number.
  4. Final (deployed) primary model: fit fresh on the FULL purged training period (not
     fold-restricted) -- standard meta-labeling practice, OOF is only for training the meta-model
     without leakage.
  5. At val: score final-primary's predictions with the meta-model, sweep the META probability
     threshold (not the primary's own confidence) on val-realized PnL, apply the chosen threshold
     unchanged to OOS.

Output: data/ensemble/reports/scalp_1m_meta_label_20260716.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from train_eval_scalp_1m_hgb_20260716 import BASE_CSV, feature_cols_for, HGB_PARAMS
from simulate_maker_entry_scalp_1m_20260716 import simulate_maker_fills, backtest_maker, ROUND_TRIP_FEE

DATA_DIR = os.path.join(_ROOT_DIR, 'data')
LABELS_WEIGHTED_CSV = os.path.join(DATA_DIR, 'training_features_1m_scalp_labels_weighted.csv')
REPORT_DIR = os.path.join(DATA_DIR, 'ensemble', 'reports')

HORIZON_MINUTES = 20
N_BLOCKS = 5
TRAIN_END = '2026-04-30'
VAL_END = '2026-05-31'
OOS_END = '2026-07-12'
META_THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

META_HGB_PARAMS = dict(
    loss='log_loss', learning_rate=0.05, max_iter=250, max_depth=4,
    l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=100,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
    class_weight='balanced', random_state=0,
)


def fit_primary(train: pd.DataFrame, feat_cols: list[str]) -> HistGradientBoostingClassifier:
    X = train[feat_cols].fillna(0.0)
    y = train['scalp_action']
    w = train['scalp_uniqueness_weight']
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(X, y, sample_weight=w)
    return clf


def primary_predict(clf, X: pd.DataFrame):
    proba = clf.predict_proba(X)
    classes = clf.classes_
    max_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), max_idx]
    pred = classes[max_idx]
    return pred, max_proba


def generate_oof_primary(train: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Purged 5-block CV: returns a DataFrame indexed like `train` with oof_pred/oof_proba."""
    train = train.sort_values('timestamp').reset_index(drop=True)
    n = len(train)
    block_bounds = np.array_split(np.arange(n), N_BLOCKS)
    purge_delta = pd.Timedelta(minutes=HORIZON_MINUTES)

    oof_pred = np.full(n, 'CASH', dtype=object)
    oof_proba = np.zeros(n)

    for b_i, block_idx in enumerate(block_bounds, 1):
        block_start_ts = train['timestamp'].iloc[block_idx[0]]
        block_end_ts = train['timestamp'].iloc[block_idx[-1]]
        is_block = np.zeros(n, dtype=bool)
        is_block[block_idx] = True
        # purge: drop 'other' rows within HORIZON minutes of either block boundary, since their
        # label windows could overlap into the held-out block.
        near_boundary = (
            (train['timestamp'] >= block_start_ts - purge_delta) & (train['timestamp'] < block_start_ts)
        ) | (
            (train['timestamp'] > block_end_ts) & (train['timestamp'] <= block_end_ts + purge_delta)
        )
        other = train[~is_block & ~near_boundary.to_numpy()]
        print(f"    OOF block {b_i}/{N_BLOCKS}: held-out n={len(block_idx):,}, "
              f"fit-on n={len(other):,} (purged {int(near_boundary.sum()):,})")

        clf = fit_primary(other, feat_cols)
        X_block = train.iloc[block_idx][feat_cols].fillna(0.0)
        pred, proba = primary_predict(clf, X_block)
        oof_pred[block_idx] = pred
        oof_proba[block_idx] = proba

    out = train[['timestamp']].copy()
    out['oof_pred'] = oof_pred
    out['oof_proba'] = oof_proba
    return out


def build_meta_target(rows: pd.DataFrame, pred_col: str, long_sim: pd.DataFrame, short_sim: pd.DataFrame) -> pd.DataFrame:
    """rows must be aligned (same index) with long_sim/short_sim (indexed by absolute row position
    in the full dataset). Returns rows filtered to non-CASH predictions with a binary meta_y."""
    non_cash = rows[rows[pred_col] != 'CASH'].copy()
    is_long = non_cash[pred_col] == 'LONG'
    filled = np.where(is_long, long_sim.loc[non_cash.index, 'filled'], short_sim.loc[non_cash.index, 'filled'])
    move = np.where(is_long, long_sim.loc[non_cash.index, 'realized_move'], short_sim.loc[non_cash.index, 'realized_move'])
    net_pnl = move - ROUND_TRIP_FEE
    non_cash['filled'] = filled
    non_cash['meta_y'] = ((filled) & (net_pnl > 0)).astype(int)
    return non_cash


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading data + weighted labels (Experiment A: full-history, price-only)...")
    df = pd.read_csv(BASE_CSV, parse_dates=['timestamp'])
    labels = pd.read_csv(LABELS_WEIGHTED_CSV, parse_dates=['timestamp'])
    df = df.merge(labels, on='timestamp', how='inner')
    df = df[df['scalp_has_full_horizon']].reset_index(drop=True)
    print(f"  {len(df):,} rows")

    print("Simulating maker-entry fills (long + short, once for the whole dataset)...")
    long_sim = simulate_maker_fills(df, 'long')
    short_sim = simulate_maker_fills(df, 'short')

    feat_cols = feature_cols_for(df, [])
    purge_cutoff = pd.Timestamp(TRAIN_END) - pd.Timedelta(minutes=HORIZON_MINUTES)
    train = df[df['timestamp'] <= purge_cutoff].reset_index(drop=False).rename(columns={'index': 'orig_idx'})
    val = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)]
    oos = df[(df['timestamp'] > VAL_END) & (df['timestamp'] <= OOS_END)]
    print(f"Train={len(train):,} Val={len(val):,} OOS={len(oos):,}")

    print("\n--- Step 1: purged 5-block OOF primary predictions ---")
    oof = generate_oof_primary(train[['timestamp'] + feat_cols + ['scalp_action', 'scalp_uniqueness_weight']], feat_cols)
    oof['orig_idx'] = train['orig_idx'].to_numpy()
    oof = oof.set_index('orig_idx')

    print("\n--- Step 2: build meta target from OOF predictions + maker-fill sim ---")
    meta_train_rows = build_meta_target(oof, 'oof_pred', long_sim, short_sim)
    meta_train_rows = meta_train_rows.join(df.loc[meta_train_rows.index, feat_cols])
    print(f"  Meta training rows (OOF non-CASH signals): {len(meta_train_rows):,}, "
          f"positive rate={meta_train_rows['meta_y'].mean():.3f}")

    print("\n--- Step 3: train meta model ---")
    meta_feat_cols = feat_cols + ['oof_proba']
    X_meta = meta_train_rows[meta_feat_cols].fillna(0.0)
    y_meta = meta_train_rows['meta_y']
    meta_clf = HistGradientBoostingClassifier(**META_HGB_PARAMS)
    meta_clf.fit(X_meta, y_meta)

    print("\n--- Step 4: train final (deployed) primary model on full purged training period ---")
    final_primary = fit_primary(train, feat_cols)

    print("\n--- Step 5: sweep meta-probability threshold on val ---")

    def score_split(split_df: pd.DataFrame):
        X = split_df[feat_cols].fillna(0.0)
        pred, proba = primary_predict(final_primary, X)
        scored = pd.DataFrame({'pred': pred, 'primary_proba': proba}, index=split_df.index)
        non_cash_mask = scored['pred'] != 'CASH'
        meta_X = pd.concat([split_df.loc[non_cash_mask, feat_cols],
                             scored.loc[non_cash_mask, 'primary_proba'].rename('oof_proba')], axis=1)
        meta_proba = np.zeros(len(split_df))
        if non_cash_mask.any():
            meta_proba_nc = meta_clf.predict_proba(meta_X[meta_feat_cols])[:, 1]
            meta_proba[non_cash_mask.to_numpy()] = meta_proba_nc
        return scored['pred'].to_numpy(), meta_proba

    val_pred, val_meta_proba = score_split(val)
    val_idx = val.index
    long_sim_val, short_sim_val = long_sim.loc[val_idx].reset_index(drop=True), short_sim.loc[val_idx].reset_index(drop=True)

    sweep = []
    for thr in META_THRESHOLDS:
        gated_pred = np.where(val_meta_proba >= thr, val_pred, 'CASH')
        bt = backtest_maker(gated_pred, long_sim_val, short_sim_val)
        sweep.append({'threshold': thr, **bt})
        print(f"  [val meta_thr={thr}] signals={bt['n_signals']:,} filled={bt['n_filled']:,} "
              f"fill_rate={bt['fill_rate']} hit_rate={bt['hit_rate']} total_pnl_pct={bt['total_pnl_pct']}")

    viable = [s for s in sweep if s['n_filled'] and s['n_filled'] >= 20]
    best = max(viable, key=lambda s: s['total_pnl_pct']) if viable else sweep[0]
    best_thr = best['threshold']
    print(f"-> best val meta threshold: {best_thr}")

    print("\n--- OOS evaluation ---")
    oos_pred, oos_meta_proba = score_split(oos)
    oos_idx = oos.index
    long_sim_oos, short_sim_oos = long_sim.loc[oos_idx].reset_index(drop=True), short_sim.loc[oos_idx].reset_index(drop=True)
    gated_pred_oos = np.where(oos_meta_proba >= best_thr, oos_pred, 'CASH')
    bt_oos = backtest_maker(gated_pred_oos, long_sim_oos, short_sim_oos)
    print(f"[oos @ meta_thr={best_thr}] signals={bt_oos['n_signals']:,} filled={bt_oos['n_filled']:,} "
          f"fill_rate={bt_oos['fill_rate']} hit_rate={bt_oos['hit_rate']} total_pnl_pct={bt_oos['total_pnl_pct']}")

    # For direct comparison, also report OOS with NO meta-gate (primary argmax only, all non-CASH).
    bt_oos_ungated = backtest_maker(oos_pred, long_sim_oos, short_sim_oos)
    print(f"[oos ungated primary-only] signals={bt_oos_ungated['n_signals']:,} "
          f"filled={bt_oos_ungated['n_filled']:,} total_pnl_pct={bt_oos_ungated['total_pnl_pct']}")

    result = {
        'experiment': 'A_meta_labeling',
        'n_blocks_oof': N_BLOCKS,
        'meta_train_n': len(meta_train_rows),
        'meta_train_positive_rate': float(meta_train_rows['meta_y'].mean()),
        'val_sweep': sweep,
        'chosen_meta_threshold': best_thr,
        'oos_at_chosen_meta_threshold': bt_oos,
        'oos_ungated_primary_only': bt_oos_ungated,
        'baseline_for_comparison': {
            'report': 'scalp_1m_tune_maker_realistic_20260716.json',
            'method': 'fixed primary confidence threshold=0.55, no meta-model',
            'oos_total_pnl_pct': 3.7390646402123644,
        },
        'compliance': {
            'fresh_forward_bar_by_bar': True, 'trade_ledgers_used_as_input': False,
            'saved_parent_exit_timestamps_used': False, 'future_rows_used_for_entry': False,
        },
        'note': ('Meta-model trained on purged 5-block OOF primary predictions (never sees primary\'s '
                 'in-sample/memorized confidence). Meta threshold chosen on val by realized maker-fill '
                 'PnL, applied unchanged to OOS.'),
    }
    with open(os.path.join(REPORT_DIR, 'scalp_1m_meta_label_20260716.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved scalp_1m_meta_label_20260716.json")


if __name__ == '__main__':
    main()
